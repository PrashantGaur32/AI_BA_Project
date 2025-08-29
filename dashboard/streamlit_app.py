
# Streamlit Dashboard - AI-Powered Business Performance (Advanced with Prophet)
# Run: streamlit run streamlit_app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os

st.set_page_config(page_title="AI Business Performance (Advanced)", layout="wide")

@st.cache_data
def load_data():
    sales = pd.read_csv("../data/sales.csv", parse_dates=["date"])
    customers = pd.read_csv("../data/customers.csv", parse_dates=["join_date"])
    competitors = pd.read_csv("../data/competitors.csv")
    # prophet forecast may be present
    prophet_path = "../report/prophet_forecast.csv"
    prophet = pd.read_csv(prophet_path) if os.path.exists(prophet_path) else None
    lr_path = "../report/model_forecast.csv"
    lr = pd.read_csv(lr_path) if os.path.exists(lr_path) else None
    return sales, customers, competitors, prophet, lr

sales, customers, competitors, prophet, lr = load_data()

st.title("AI-Powered Business Performance & Market Insights Dashboard (Advanced)")

# Tabs for layout
tab1, tab2, tab3 = st.tabs(["Overview", "Prophet Forecast", "Comparison (LR vs Prophet)"])

with tab1:
    # KPI row
    col1, col2, col3, col4 = st.columns(4)
    total_revenue = sales['revenue'].sum()
    total_profit = sales['profit'].sum()
    avg_clv = customers['clv'].mean()
    churn_rate = customers['churn_flag'].mean()

    col1.metric("Total Revenue", f"{total_revenue:,.0f}")
    col2.metric("Total Profit", f"{total_profit:,.0f}")
    col3.metric("Avg CLV", f"{avg_clv:,.2f}")
    col4.metric("Churn Rate", f"{churn_rate:.2%}")

    st.markdown("### Revenue Trend (Monthly)")
    monthly = sales.groupby(sales['date'].dt.to_period('M')).agg({'revenue':'sum'}).reset_index()
    monthly['date'] = monthly['date'].dt.to_timestamp()
    fig, ax = plt.subplots(figsize=(10,3))
    ax.plot(monthly['date'], monthly['revenue'], label='Actual')
    ax.set_ylabel("Revenue")
    ax.legend()
    st.pyplot(fig)

    st.markdown("### Top Products by Revenue")
    prod = sales.groupby('product').agg({'revenue':'sum'}).sort_values('revenue', ascending=False).reset_index()
    st.bar_chart(prod.rename(columns={"revenue":"value"}).set_index('product'))

    st.markdown("### Customer Segments")
    seg = customers.groupby('segment').agg({'clv':'mean','churn_flag':'mean','customer_id':'count'}).rename(columns={'customer_id':'count'}).reset_index()
    st.dataframe(seg)

with tab2:
    st.markdown("## Prophet Forecast (Advanced)")
    horizon = st.selectbox("Forecast Horizon (months)", options=[3,6,12], index=1)
    # Inform user to run analysis if no prophet forecast present
    if prophet is None or (prophet.shape[1] > 0 and 'yhat' not in prophet.columns):
        st.warning("Prophet forecast not found or Prophet not installed. Run `python notebooks/analysis.py` locally after installing 'prophet' to generate advanced forecast.")
    else:
        # Read prophet forecast and plot horizon months
        pf = prophet.copy()
        pf['ds'] = pd.to_datetime(pf['ds'])
        # choose the last 'horizon' months of forecast after the last actual
        pf_plot = pf.tail(horizon)
        fig, ax = plt.subplots(figsize=(10,4))
        ax.plot(pf_plot['ds'], pf_plot['yhat'], label='Prophet yhat')
        ax.fill_between(pf_plot['ds'], pf_plot['yhat_lower'], pf_plot['yhat_upper'], alpha=0.2)
        ax.set_title("Prophet Forecast (with confidence interval)")
        ax.legend()
        st.pyplot(fig)
        st.dataframe(pf_plot[['ds','yhat','yhat_lower','yhat_upper']].rename(columns={'ds':'date'}))

with tab3:
    st.markdown("## Comparison: Linear Regression vs Prophet")
    # Show both forecasts if available
    monthly = sales.groupby(sales['date'].dt.to_period('M')).agg({'revenue':'sum'}).reset_index()
    monthly['date'] = pd.to_datetime(monthly['month'].astype(str) + "-01")
    fig, ax = plt.subplots(figsize=(12,4))
    ax.plot(monthly['date'], monthly['revenue'], label='Actual', linewidth=1)
    if lr is not None and 'forecast_revenue' in lr.columns:
        # create future dates for lr forecast based on last monthly date
        last = monthly['date'].max()
        future_dates = pd.date_range(start=last + pd.offsets.MonthBegin(1), periods=len(lr), freq='M')
        ax.plot(future_dates, lr['forecast_revenue'], label='LinearRegression Forecast', linestyle='--')
    if prophet is not None and 'yhat' in prophet.columns:
        pf = prophet.copy()
        pf['ds'] = pd.to_datetime(pf['ds'])
        ax.plot(pf['ds'], pf['yhat'], label='Prophet Forecast', linestyle=':')
        # show confidence band for entire prophet forecast
        ax.fill_between(pf['ds'], pf['yhat_lower'], pf['yhat_upper'], alpha=0.1)
    ax.set_ylabel("Revenue")
    ax.legend()
    st.pyplot(fig)
    st.markdown("### Forecast Tables (for reference)")
    cols = st.columns(2)
    with cols[0]:
        st.write("Linear Regression Forecast")
        if lr is not None:
            st.dataframe(lr)
        else:
            st.info("Run `python notebooks/analysis.py` to generate linear regression forecast.")
    with cols[1]:
        st.write("Prophet Forecast")
        if prophet is not None and 'yhat' in prophet.columns:
            st.dataframe(prophet[['ds','yhat','yhat_lower','yhat_upper']])
        else:
            st.info("Prophet forecast not available. Install `prophet` and run analysis.py locally to generate it.")
