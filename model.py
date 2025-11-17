import os
import re
import glob
import pandas as pd
import numpy as np
import optuna
import torch

from darts import TimeSeries
from darts.models import NHiTSModel
from darts.metrics import mae
from darts.dataprocessing.transformers import Scaler
from darts.logging import raise_log

# --- User settings ---
PRODUCTS_DIR = "products_csv"         # folder with product CSVs
OUTPUT_DIR = "nhits_models"           # folder to save per-product models
COMMON_MODEL_PATH = "common_nhits_model.h5"  # final shared model save path
N_TRIALS = 20                         # Optuna trials per product
HORIZON = 6                           # forecast horizon (months) to optimize for
MIN_SERIES_LENGTH = 12                # Minimum data points needed: 6 (ICL) + 6 (Horizon)
MAX_EPOCHS = 80                       # training epochs
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs("forecasts", exist_ok=True) # Ensure forecasts directory exists

# --- Helpers ---
def safe_name(s):
    """Sanitizes product name for use in filenames."""
    return re.sub(r'[^A-Za-z0-9]+', '_', s).strip('_')

def load_product_series(csv_path):
    """Loads CSV into a Darts TimeSeries, handling date frequency and missing values."""
    df = pd.read_csv(csv_path)
    # Expect columns: date, product, quantity
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    
    # CRITICAL: Set monthly frequency ('MS' for Month Start) and fill missing dates with 0.0
    ts = TimeSeries.from_dataframe(
        df, 
        'date', 
        'quantity',
        freq='MS',                
        fill_missing_dates=True,  
        fillna_value=0.0          
    )
    return ts, df['product'].iloc[0]  # series, product_name

# --- Optuna objective for a single series ---
def make_optuna_objective(train_ts, val_ts, input_chunk_hint=24, horizon=HORIZON):
    """
    Returns an objective(trial) that builds a NHiTS model with trial hyperparams,
    fits it and returns validation MAE.
    """
    num_stacks = 3
    max_train_len = len(train_ts)
    max_input_len_allowed = max_train_len - horizon
    
    # Define bounds for input_chunk_len sampling
    min_icl_bound = 6
    max_icl_bound = min(max_input_len_allowed, max(24, input_chunk_hint))

    if max_icl_bound < min_icl_bound:
        # This condition should ideally be caught by the pre-filter, but kept for safety
        print(f"Skipping Optuna: Training series length ({max_train_len}) is too short to support a minimum ICL of 6 + OCL of {horizon}.")
        def failed_objective(trial):
            return 1e9
        return failed_objective
        
    def objective(trial):
        # sample hyperparameters
        input_chunk_len = trial.suggest_int("input_chunk_len", min_icl_bound, max_icl_bound)
        num_blocks = trial.suggest_int("num_blocks", 1, 4)
        num_layers = trial.suggest_int("num_layers", 1, 3)
        layer_size = trial.suggest_int("layer_size", 32, 512, log=True)
        dropout = trial.suggest_float("dropout", 0.0, 0.3)
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])

        model = NHiTSModel(
            input_chunk_length = input_chunk_len,
            output_chunk_length = horizon,
            n_epochs = 20,                          
            batch_size = batch_size,
            num_blocks = num_blocks,
            num_layers = num_layers,
            layer_widths = [layer_size] * num_stacks, 
            dropout = dropout,
            optimizer_kwargs = {"lr": lr},
            random_state = 42,
            likelihood = None,
            pl_trainer_kwargs = {"accelerator": "gpu"} if DEVICE.startswith("cuda") else {}
        )

        try:
            model.fit(train_ts, verbose=False)
            pred = model.predict(n=horizon)
            eval_len = min(len(val_ts), len(pred))
            score = mae(val_ts[-eval_len:], pred[:eval_len])
        except Exception as e:
            # If training fails, return a large loss
            score = 1e9
        # free GPU memory
        del model
        torch.cuda.empty_cache()
        return float(score)
    return objective

# --- Pre-filter and Load Function ---
def filter_and_load_series(products_dir, min_length):
    """
    Reads all CSVs, loads them into TimeSeries, and filters out short series.
    
    Returns: list of (TimeSeries, product_name, csv_path) tuples
    """
    suitable_products = []
    
    for csv_path in glob.glob(os.path.join(products_dir, "*.csv")):
        print(f"Checking: {os.path.basename(csv_path)}", end="\r")
        try:
            ts, prod_name = load_product_series(csv_path)
            
            if len(ts) >= min_length:
                suitable_products.append((ts, prod_name, csv_path))
            else:
                print(f"Skipping: {prod_name} (Length: {len(ts)}). Too short (Needs >= {min_length}).")

        except ValueError as e:
            # Catch the ValueError for index frequency (should be rare with the fix)
            print(f"Skipping: {os.path.basename(csv_path)} due to TimeSeries creation error: {e}")
        except Exception as e:
            print(f"Skipping: {os.path.basename(csv_path)} due to unexpected error: {e}")
            
    return suitable_products

# --- Per-product tuning / training function ---
def tune_and_train_product(ts: TimeSeries, product_name: str):
    """Runs Optuna tuning and trains the final NHiTS model on one product."""
    
    # 1. Split for Tuning (last HORIZON months for validation)
    train_ts, val_ts = ts[:-HORIZON], ts[-HORIZON:]

    # 2. Scale series
    scaler = Scaler()
    train_scaled = scaler.fit_transform(train_ts)
    val_scaled = scaler.transform(val_ts)

    # 3. Optuna Hyperparameter Tuning
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
    input_chunk_hint = min(36, max(12, len(train_ts)//2))
    objective = make_optuna_objective(train_scaled, val_scaled, input_chunk_hint=input_chunk_hint)
    
    if hasattr(objective, '__name__') and objective.__name__ == 'failed_objective':
        return None, None

    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

    print(f"Best params for {product_name}: {study.best_params}, best_val_mae={study.best_value}")
    
    # 4. Use Best Params or Defaults
    if study.best_value == 1e9:
        print("WARNING: All Optuna trials failed. Using default parameters.")
        best = {}
    else:
        best = study.best_params

    # 5. Build and Fit Final Model
    num_stacks = 3
    final_input_chunk_len = best.get("input_chunk_len", 24)
    
    # Ensure ICL works for the full series 
    full_len = len(ts)
    if final_input_chunk_len + HORIZON > full_len:
        final_input_chunk_len = max(6, full_len - HORIZON)
        print(f"Adjusting final input_chunk_len to {final_input_chunk_len} due to short series length ({full_len}).")

    final_model = NHiTSModel(
        input_chunk_length = final_input_chunk_len,
        output_chunk_length = HORIZON,
        n_epochs = MAX_EPOCHS,
        batch_size = best.get("batch_size", 32),
        num_blocks = best.get("num_blocks", 2),
        num_layers = best.get("num_layers", 2),
        layer_widths = [best.get("layer_size", 128)] * num_stacks,
        dropout = best.get("dropout", 0.0),
        optimizer_kwargs = {"lr": best.get("lr", 1e-3)},
        random_state = 42,
        pl_trainer_kwargs = {"accelerator": "gpu"} if DEVICE.startswith("cuda") else {}
    )

    # Fit on the entire series (train+val)
    # The scaler used here will be saved and used for inverse transforming 
    scaler = Scaler()
    scaled_full = scaler.fit_transform(ts)
    final_model.fit(scaled_full, verbose=True)

    # 6. Save Model and Scaler
    filename = os.path.join(OUTPUT_DIR, safe_name(product_name) + "_nhits.h5")
    final_model.save(filename)
    scaler_path = os.path.join(OUTPUT_DIR, safe_name(product_name) + "_scaler.pkl")
    pd.to_pickle(scaler, scaler_path)

    print(f"Saved model: {filename} and scaler: {scaler_path}")
    return final_model, scaler

# ====================================================================
# --- Main Pipeline Execution ---
# ====================================================================

# 1. Pre-filter and Load all suitable series
print(f"--- Stage 1: Filtering CSVs (Requires >= {MIN_SERIES_LENGTH} months of data) ---")
all_data = filter_and_load_series(PRODUCTS_DIR, MIN_SERIES_LENGTH)

if not all_data:
    print("No products found with sufficient data length. Cannot proceed with NHiTS training.")
else:
    print(f"\nFound {len(all_data)} products suitable for NHiTS training.")

    # Unpack suitable data
    suitable_series = [item[0] for item in all_data]
    product_names = [item[1] for item in all_data]
    
    # Lists to hold results for global model training and ensemble forecasting
    scaled_series_list = []

    print("\n--- Stage 2: Per-Product NHiTS Training (Optuna + Final Fit) ---")
    # 2. Per-Product Training Loop
    for i, (ts, prod_name, _) in enumerate(all_data):
        print(f"\nProcessing Product {i+1}/{len(all_data)}: {prod_name}")
        
        # Scale the series for global model training (using a dedicated scaler for the global list)
        global_scaler = Scaler()
        scaled_ts = global_scaler.fit_transform(ts)
        scaled_series_list.append(scaled_ts)

        # Retrain model and save in per-product function
        # Note: tune_and_train_product handles its own scaling for saving the individual model/scaler
        tune_and_train_product(ts, prod_name)


    # 3. Train a common (global) NHITS model on all suitable series
    print("\n" + "="*60)
    print("--- Stage 3: Training Global NHITS Model on All Suitable Products ---")
    print("="*60)
    
    # The global model must use parameters compatible with the *shortest* suitable series (Length 12).
    # Since ICL + HORIZON = 12, the maximum safe ICL is 6.
    GLOBAL_ICL = 6 
    
    num_stacks_global = 3 
    global_model = NHiTSModel(
        input_chunk_length = GLOBAL_ICL, # Fixed to 6 (max safe ICL)
        output_chunk_length = HORIZON,
        n_epochs = MAX_EPOCHS,
        batch_size = 32, 
        num_blocks = 2,
        num_layers = 2,
        layer_widths = [128] * num_stacks_global, 
        dropout = 0.05,
        optimizer_kwargs = {"lr": 1e-3},
        random_state = 42,
        pl_trainer_kwargs = {"accelerator": "gpu"} if DEVICE.startswith("cuda") else {}
    )

    # Fit on list of scaled TimeSeries
    global_model.fit(scaled_series_list, verbose=True)

    # Save the final common model
    global_model.save(COMMON_MODEL_PATH)
    print("Saved common global model to:", COMMON_MODEL_PATH)

    # 4. Generate Ensemble Forecasts
    print("\n" + "="*60)
    print("--- Stage 4: Generating Ensemble Forecasts ---")
    print("="*60)
    
    # Pass the scaled historical data used for training for multi-series prediction.
    all_common_preds = global_model.predict(n=HORIZON, series=scaled_series_list)
    
    FORECAST_OUTPUT = "forecasts"

    for i, prod_name in enumerate(product_names):
        per_product_path = os.path.join(OUTPUT_DIR, safe_name(prod_name) + "_nhits.h5")
        scaler_path = os.path.join(OUTPUT_DIR, safe_name(prod_name) + "_scaler.pkl")

        try:
            # Load per-product model and scaler
            per_model = NHiTSModel.load(per_product_path)
            # This scaler is the one used to transform the individual product series
            scaler = pd.read_pickle(scaler_path) 
            
            # Get the corresponding common forecast for this product
            common_pred = all_common_preds[i]
            
            # Generate per-product forecast 
            per_pred = per_model.predict(n=HORIZON, series=suitable_series[i])

        except Exception as e:
            print(f"Failed loading per-product model or prediction for {prod_name}. Skipping ensemble: {e}")
            continue

        # Inverse transform
        per_inv = scaler.inverse_transform(per_pred)
        # Note: common_pred is already scaled from scaled_series_list
        common_inv = scaler.inverse_transform(common_pred) 

        # Average and save
        # FIX: Changed 'pd_dataframe()' to 'to_dataframe()'
        per_df = per_inv.to_dataframe().rename(columns={per_inv.to_dataframe().columns[0]:"per_product"})
        com_df = common_inv.to_dataframe().rename(columns={common_inv.to_dataframe().columns[0]:"common"})
        merged = pd.concat([per_df, com_df], axis=1)
        merged['ensemble'] = merged.mean(axis=1)

        outpath = os.path.join(FORECAST_OUTPUT, safe_name(prod_name) + "_forecast.csv")
        merged.to_csv(outpath, index=True)
        print(f"Saved forecast for {prod_name}: {outpath}")

print("\nPipeline execution complete.")