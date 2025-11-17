

# 📦 Sales Forecasting System (NHITS + Optuna + Streamlit)

This project provides a complete **end-to-end automated sales forecasting pipeline** using:

* ⏳ **N-HiTS deep learning models**
* 🎯 **Hyperparameter tuning with Optuna**
* 📊 **Per-product + Global Ensemble Forecasting**
* 📈 **Interactive Streamlit Forecasting App**
* 🧹 **Automated Excel → per-product CSV conversion**

---

## 🚀 Project Workflow

### **1. Data Preparation**

Handled by **`data_split.py`**.
This script:

* Loads `data.xlsx`
* Converts wide-format monthly data into long format
* Splits into **one CSV per product**
* Saves them into:

```
products_csv/
    ProductA.csv
    ProductB.csv
    ...
```

Each CSV contains:

```
date, product, quantity
2022-01-01, Product A, 120
```

---

### **2. Model Training**

Forecasting logic is implemented in:

* **`model.py`** — main training pipeline
* **`model_test.py`** — extended training with accuracy metrics

#### **Per-Product N-HiTS Training**

For each product:

* Converts to monthly frequency (`MS`)
* Fills missing months automatically
* Runs **Optuna (20 trials)** to find best hyperparameters
* Trains final model with optimized settings
* Saves:

```
nhits_models/
    ProductA_nhits.h5
    ProductA_scaler.pkl
```

#### **Global N-HiTS Training**

A shared model trained across all product series:

```
common_nhits_model.h5
```

#### **Ensemble Forecasting**

For each product:

* Per-product forecast
* Global-model forecast
* Averaged **ensemble forecast**

Saved at:

```
forecasts/ProductA_forecast.csv
```

Columns:

```
per_product, common, ensemble
```

#### **Accuracy Reporting**

The extended script (`model_test.py`) also computes:

* Per-product MAE
* Overall MAE scores

---

## 🎨 Streamlit Forecasting App

Implemented in **`app.py`**.

Features:

✔ Upload CSV (`date,product,quantity`)
✔ Automatic monthly resampling
✔ Uses **global N-HiTS model**
✔ Shows:

* Next-month forecast
* Trend plot (Plotly)
* Forecast table

Run the app:

```bash
streamlit run app.py
```

---

## 📂 Project Structure

```
.
├── data_split.py            # Convert Excel → product CSVs
├── model.py                 # NHITS + Optuna training pipeline
├── model_test.py            # Extended training & accuracy metrics
├── app.py                   # Streamlit forecasting UI
├── products_csv/            # Generated product-level CSVs
├── nhits_models/            # Saved per-product models
├── forecasts/               # Saved predictions
├── common_nhits_model.h5    # Global shared model
├── requirements.txt         # Dependencies
└── .gitignore
```

---

## 🔧 Installation

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Usage Steps

### **1. Prepare data**

Place your Excel file as:

```
data.xlsx
```

Then run:

```bash
python data_split.py
```

### **2. Train Models**

Main pipeline:

```bash
python model.py
```

Extended (with MAE):

```bash
python model_test.py
```

### **3. Run the Streamlit App**

```bash
streamlit run app.py
```

Upload a CSV in:

```
date,product,quantity
```

format.

---

## 🔮 Outputs

Each forecast file (per product):

```
date | per_product | common | ensemble
```

Example:

| date       | per_product | common | ensemble |
| ---------- | ----------- | ------ | -------- |
| 2025-02-01 | 180         | 165    | 173      |

---

## 🧠 Notes

* Minimum **12 months** of data required for model stability.
* Ensemble forecasts generally provide more robust predictions.
* Streamlit app uses only the **global model** for simplicity.

---

If you want, I can also generate:

✅ Architecture diagram
✅ Badges
✅ Setup script
✅ Dockerfile
