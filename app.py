import os

# ===================== SaaS / Linux Server Configuration =====================
os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"
os.environ["STREAMLIT_SERVER_ENABLE_CORS"] = "false"
os.environ["STREAMLIT_SERVER_PORT"] = "8501"
# ============================================================================

import io
import base64
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt

from datetime import timedelta
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
from xgboost import XGBRegressor

st.set_page_config(
    page_title="Retail Demand Forecasting",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)


def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    if not np.any(mask):
        return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def preprocess_data(df, date_col, target_col, product_col=None, selected_product=None):
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col, target_col])

    if product_col is not None and selected_product is not None:
        df = df[df[product_col] == selected_product]

    df = df.sort_values(date_col)
    df = df.groupby(date_col)[target_col].sum().reset_index()
    df = df.rename(columns={date_col: "ds", target_col: "y"})

    inferred_freq = pd.infer_freq(df["ds"])
    if inferred_freq is None:
        inferred_freq = "D"

    df = df.set_index("ds").asfreq(inferred_freq)

    y = df["y"].copy()
    stockout_mask = (y == 0) & (y.shift(1) > 0) & (y.shift(-1) > 0)
    y[stockout_mask] = np.nan
    y = y.interpolate(method="linear").ffill().bfill()

    q1, q3 = y.quantile([0.25, 0.75])
    iqr = q3 - q1
    if iqr > 0:
        low_cap = q1 - 1.5 * iqr
        high_cap = q3 + 1.5 * iqr
        y = y.clip(lower=low_cap, upper=high_cap)

    df["y"] = y
    df = df.reset_index()
    return df


def plot_time_series(df):
    fig = px.line(df, x="ds", y="y", title="Sales Over Time")
    fig.update_layout(template="plotly_white")
    return fig


def plot_distribution(df):
    fig = px.histogram(df, x="y", nbins=30, title="Sales Distribution")
    fig.update_layout(template="plotly_white")
    return fig


def plot_day_month_patterns(df):
    temp = df.copy()
    temp["day_of_week"] = temp["ds"].dt.day_name()
    temp["month"] = temp["ds"].dt.month_name().str.slice(stop=3)

    fig1 = px.box(temp, x="day_of_week", y="y", title="Sales by Day of Week")
    fig2 = px.box(temp, x="month", y="y", title="Sales by Month")
    fig1.update_layout(template="plotly_white")
    fig2.update_layout(template="plotly_white")
    return fig1, fig2


def plot_month_year_heatmap(df):
    temp = df.copy()
    temp["year"] = temp["ds"].dt.year
    temp["month"] = temp["ds"].dt.month
    pivot = temp.pivot_table(index="year", columns="month", values="y", aggfunc="sum")

    fig = go.Figure(
        data=go.Heatmap(
            z=pivot.values,
            x=[f"{m:02d}" for m in pivot.columns],
            y=pivot.index.astype(str),
            colorscale="Purples",
        )
    )
    fig.update_layout(
        title="Month vs Year Sales Heatmap",
        xaxis_title="Month",
        yaxis_title="Year",
        template="plotly_white",
    )
    return fig


def seasonal_decomposition_plot(df):
    series = df.set_index("ds")["y"].asfreq("D").interpolate()
    result = seasonal_decompose(series, model="additive", period=7)

    fig, axes = plt.subplots(4, 1, figsize=(12, 8), sharex=True)
    result.observed.plot(ax=axes[0], title="Observed")
    result.trend.plot(ax=axes[1], title="Trend")
    result.seasonal.plot(ax=axes[2], title="Seasonal")
    result.resid.plot(ax=axes[3], title="Residual")
    plt.tight_layout()
    return fig


def train_test_split_time_series(df, test_size):
    split_point = len(df) - test_size
    return df.iloc[:split_point], df.iloc[split_point:]


def train_arima(train):
    model = SARIMAX(train["y"], order=(1, 1, 1), enforce_stationarity=False, enforce_invertibility=False)
    return model.fit(disp=False)


def forecast_arima(model, train, steps):
    pred = model.get_forecast(steps=steps)
    conf = pred.conf_int()
    future_dates = pd.date_range(train["ds"].iloc[-1] + timedelta(days=1), periods=steps, freq="D")

    return pd.DataFrame({
        "ds": future_dates,
        "yhat": pred.predicted_mean.values,
        "yhat_lower": conf.iloc[:, 0].values,
        "yhat_upper": conf.iloc[:, 1].values,
    })


def train_prophet(train):
    m = Prophet(daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=True)
    m.fit(train[["ds", "y"]])
    return m


def forecast_prophet(model, train, periods):
    future = model.make_future_dataframe(periods=periods, freq="D")
    forecast = model.predict(future)
    return forecast.tail(periods)[["ds", "yhat", "yhat_lower", "yhat_upper"]]


def compute_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return mae, rmse, mape(y_true, y_pred)


def get_table_download_link(df, filename="forecast.csv"):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    return f'<a href="data:file/csv;base64,{b64}" download="{filename}">📥 Download Forecast</a>'


def main():
    st.sidebar.title("⚙️ Configuration")
    uploaded_file = st.sidebar.file_uploader("Upload Sales CSV", type=["csv"])
    model_choice = st.sidebar.selectbox("Model", ["ARIMA", "Prophet"])
    horizon = st.sidebar.slider("Forecast Horizon (days)", 7, 60, 30)
    test_size = st.sidebar.slider("Test Size (days)", 7, 90, 30)
    run = st.sidebar.button("🚀 Train & Forecast")

    st.title("🛒 Retail Demand Forecasting System")

    if uploaded_file is None:
        st.info("Upload a CSV file to start.")
        return

    df_raw = pd.read_csv(uploaded_file)
    st.subheader("Dataset Preview")
    st.dataframe(df_raw.head())

    date_col = st.selectbox("Date Column", df_raw.columns)
    target_col = st.selectbox("Sales Column", df_raw.columns)

    df = preprocess_data(df_raw, date_col, target_col)

    st.plotly_chart(plot_time_series(df), use_container_width=True)

    train, test = train_test_split_time_series(df, test_size)

    if run:
        if model_choice == "ARIMA":
            model = train_arima(train)
            forecast_df = forecast_arima(model, df, horizon)
        else:
            model = train_prophet(train)
            forecast_df = forecast_prophet(model, train, horizon)

        st.subheader("📈 Forecast")
        st.dataframe(forecast_df)
        st.markdown(get_table_download_link(forecast_df), unsafe_allow_html=True)


# ===================== SaaS ENTRY POINT =====================
def app():
    main()
if __name__=="__main__":
    app()
# ===========================================================
