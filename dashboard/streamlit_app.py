import os
import pandas as pd
import streamlit as st
from io import StringIO

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # project root
DATA_DIR = os.path.join(BASE_DIR, "data")

# Sample data (for Render demo if files not found)
SAMPLE_SALES = """date,sales
2023-01-01,100
2023-01-02,120
2023-01-03,130
"""

SAMPLE_CUSTOMERS = """id,name,spend
1,Alice,200
2,Bob,350
3,Charlie,150
"""

SAMPLE_COMPETITORS = """date,competitor_sales
2023-01-01,90
2023-01-02,140
2023-01-03,110
"""

@st.cache_data
def load_data():
    try:
        # Try loading from data/ folder
        sales = pd.read_csv(os.path.join(DATA_DIR, "sales.csv"), parse_dates=["date"])
        customers = pd.read_csv(os.path.join(DATA_DIR, "customers.csv"))
        competitors = pd.read_csv(os.path.join(DATA_DIR, "competitors.csv"))
    except FileNotFoundError:
        # Fallback for Render demo
        st.warning("⚠️ Local data files not found. Using sample data.")
        sales = pd.read_csv(StringIO(SAMPLE_SALES), parse_dates=["date"])
        customers = pd.read_csv(StringIO(SAMPLE_CUSTOMERS))
        competitors = pd.read_csv(StringIO(SAMPLE_COMPETITORS))

    return sales, customers, competitors, None, None

