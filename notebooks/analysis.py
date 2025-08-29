"""
AI-Powered Business Performance - Analysis Script (with Prophet forecast)
Run: python analysis.py
Produces: summary_kpis.csv, model_forecast.csv (linear regression), prophet_forecast.csv (if prophet installed)
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import os

sales = pd.read_csv("../data/sales.csv", parse_dates=["date"])
customers = pd.read_csv("../data/customers.csv", parse_dates=["join_date"])
competitors = pd.read_csv("../data/competitors.csv")

# --- KPI calculations ---
sales['month'] = sales['date'].dt.to_period('M').astype(str)
monthly = sales.groupby('month').agg({'revenue':'sum','profit':'sum','quantity':'sum'}).reset_index()
monthly['profit_margin'] = monthly['profit'] / monthly['revenue']
monthly['revenue_yoy_growth'] = monthly['revenue'].pct_change().fillna(0)

# Customer KPIs
total_customers = customers.shape[0]
churn_rate = customers['churn_flag'].mean()
avg_clv = customers['clv'].mean()

kpis = {
    'total_customers': total_customers,
    'churn_rate': churn_rate,
    'avg_clv': avg_clv,
    'latest_month_revenue': monthly.iloc[-1]['revenue'],
    'latest_month_profit': monthly.iloc[-1]['profit']
}

os.makedirs("../report", exist_ok=True)
pd.DataFrame([kpis]).to_csv("../report/summary_kpis.csv", index=False)

# --- Simple forecasting: Linear Regression baseline ---
monthly_lr = monthly.copy().reset_index(drop=True)
monthly_lr['month_index'] = range(len(monthly_lr))
X = monthly_lr[['month_index']].values
y = monthly_lr['revenue'].values
model = LinearRegression()
model.fit(X,y)
future_index = np.arange(len(monthly_lr), len(monthly_lr)+3).reshape(-1,1)
forecast = model.predict(future_index)
forecast_df = pd.DataFrame({
    'month_index': future_index.flatten(),
    'forecast_revenue': forecast
})
forecast_df.to_csv("../report/model_forecast.csv", index=False)

# --- Prophet forecasting (advanced) ---
try:
    from prophet import Prophet
    prophet_available = True
except Exception as e:
    Prophet = None
    prophet_available = False

if prophet_available:
    # Prepare data for Prophet (ds, y)
    prophet_df = monthly[['month','revenue']].copy()
    prophet_df['ds'] = pd.to_datetime(prophet_df['month'] + "-01")
    prophet_df = prophet_df[['ds','revenue']].rename(columns={'revenue':'y'})

    m = Prophet()
    m.fit(prophet_df)

    # Forecast horizon (default 6 months)
    future = m.make_future_dataframe(periods=6, freq='M')
    fcst = m.predict(future)
    # Keep relevant columns and save
    out = fcst[['ds','yhat','yhat_lower','yhat_upper']].copy()
    out['month'] = out['ds'].dt.to_period('M').astype(str)
    out.to_csv("../report/prophet_forecast.csv", index=False)
    print("Prophet forecast generated and saved to ../report/prophet_forecast.csv")
else:
    # Create a small placeholder CSV noting that Prophet isn't installed
    placeholder = pd.DataFrame([{"note":"prophet_not_installed","message":"Install 'prophet' to generate advanced forecasts locally."}])
    placeholder.to_csv("../report/prophet_forecast.csv", index=False)
    print("Prophet is not installed in this environment. A placeholder was saved to ../report/prophet_forecast.csv")
    
print("Analysis complete. Outputs in /report: summary_kpis.csv, model_forecast.csv, prophet_forecast.csv")
