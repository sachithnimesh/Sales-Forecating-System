import streamlit as st
import pandas as pd
import numpy as np
import re
import os
import torch
import plotly.express as px

from darts import TimeSeries
from darts.models import NHiTSModel

MODEL_PATH = r"common_nhits_model.h5"
HORIZON = 6

def safe_name(s):
    return re.sub(r'[^A-Za-z0-9]+', '_', s).strip('_')

@st.cache_resource
def load_common_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file not found: {MODEL_PATH}")
        return None
    try:
        model = NHiTSModel.load(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Failed loading model: {e}")
        return None

@st.cache_data
def load_and_preprocess_data(uploaded_file):
    try:
        df = pd.read_csv(uploaded_file)
        required = {'date', 'product', 'quantity'}
        if not required.issubset(df.columns):
            st.error(f"CSV missing required columns: {required}")
            return None
        df['date'] = pd.to_datetime(df['date'], dayfirst=True, errors='coerce')
        df = df.dropna(subset=['date'])
        df['product'] = df['product'].astype(str).str.strip()
        df = df.sort_values(['product', 'date'])
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

def make_monthly_series(df_prod):
    dfp = df_prod.set_index('date').copy()
    dfp['quantity'] = pd.to_numeric(dfp['quantity'], errors='coerce').fillna(0.0)
    monthly = dfp['quantity'].resample('MS').sum().to_frame(name='quantity')
    return monthly

def convert_to_timeseries(monthly_df):
    try:
        ts = TimeSeries.from_dataframe(
            monthly_df,
            time_col=None,
            value_cols='quantity',
            freq='MS',
            fill_missing_dates=False,
            fillna_value=0.0
        )
        return ts
    except Exception as e:
        st.error(f"Error creating TimeSeries: {e}")
        return None

def predict_and_clip(model, series):
    try:
        # Use the model directly, no inverse scaling
        fc_ts = model.predict(n=HORIZON, series=series, verbose=False)
        fc_df = fc_ts.to_dataframe().rename(columns={fc_ts.columns[0]: 'Forecast'})
        fc_df.index = pd.to_datetime(fc_df.index)
        fc_df['Forecast'] = fc_df['Forecast'].clip(lower=0.0)
        next_date = fc_df.index[0]
        next_qty = float(fc_df.iloc[0]['Forecast'])
        return fc_df, next_date, next_qty
    except Exception as e:
        st.error(f"Forecasting error: {e}")
        return None, None, None

def main_app():
    st.set_page_config(layout="wide", page_title="Sales Forecast with Single Model")
    st.title("Sales Forecast (No External Scaler)")

    uploaded = st.sidebar.file_uploader("Upload CSV (product, date, quantity)", type=['csv'])
    df = load_and_preprocess_data(uploaded) if uploaded else None
    if df is None or df.empty:
        st.info("Upload valid data to proceed.")
        return

    products = sorted(df['product'].unique().tolist())
    selected = st.sidebar.selectbox("Select product", products)
    if not selected:
        return

    dfp = df[df['product'] == selected].copy()
    if dfp.empty:
        st.error("No data for that product.")
        return

    monthly_df = make_monthly_series(dfp)
    if monthly_df is None or monthly_df.empty:
        st.error("Monthly series is invalid.")
        return

    st.write("Monthly data sample:", monthly_df.head())

    series = convert_to_timeseries(monthly_df)
    if series is None:
        return

    model = load_common_model()
    if model is None:
        return

    with st.spinner("Forecasting..."):
        fc_df, nd, nq = predict_and_clip(model, series)

    if fc_df is None:
        return

    st.subheader("Next Month Forecast")
    st.metric(label=f"{nd.strftime('%B %Y')}", value=f"{nq:,.0f} units")

    hist_df = monthly_df.rename(columns={'quantity': 'Historical'})
    plot_df = pd.concat([hist_df, fc_df], axis=0)
    plot_df = plot_df.reset_index().rename(columns={'index': 'date'})
    plot_df['Type'] = np.where(plot_df['Forecast'].notna(), 'Forecast', 'Historical')
    plot_df['Quantity'] = plot_df['Historical'].combine_first(plot_df['Forecast'])
    plot_df['date'] = pd.to_datetime(plot_df['date'], errors='coerce')
    plot_df = plot_df.dropna(subset=['date'])

    st.write("Plot df columns:", plot_df.columns.tolist())
    st.write(plot_df.head())

    try:
        fig = px.line(
            plot_df,
            x='date',
            y='Quantity',
            color='Type',
            title=f"Historical vs Forecast ({HORIZON} months)",
            markers=True
        )
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Plotly plot failed: {e}")
        try:
            temp = plot_df.set_index('date')['Quantity']
            st.line_chart(temp)
        except Exception as e2:
            st.error(f"Fallback line chart failed: {e2}")
            st.write("Cols:", plot_df.columns.tolist())
            st.write(plot_df.head())

    st.subheader("Forecast Table")
    st.dataframe(fc_df.style.format("{:,.0f}"))

if __name__ == "__main__":
    main_app()
