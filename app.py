# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from sklearn.ensemble import RandomForestRegressor
try:
    from xgboost import XGBRegressor
    XGB_AVAILABLE = True
except:
    XGB_AVAILABLE = False

st.set_page_config(page_title="Crust & Bloom Dashboard", layout="wide")
st.title("🥐 Crust & Bloom Production & Revenue Dashboard")

# ---- Load Data ----
st.sidebar.header("Upload your Excel file")
data_file = st.sidebar.file_uploader("Upload your dataset", type=["xlsx"])

if data_file is None:
    st.warning("Please upload your dataset to proceed.")
    st.stop()

df_raw = pd.read_excel(data_file, parse_dates=["Date"])
st.subheader("Raw Data Sample")
st.dataframe(df_raw.head())

# ---- Prepare Data ----
st.markdown("### Data Preparation & Feature Engineering")
df = df_raw.copy()

# Target: Units Sold
y = df["Units Sold"]

# Create features
df["Revenue_per_Unit"] = df["Revenue"] / df["Units Sold"].replace(0, 1)
df["Waste_Rate"] = df["Waste"] / df["Units Produced"].replace(0,1)
df["Month"] = df["Date"].dt.month
df["DayOfWeek"] = df["Date"].dt.dayofweek

# Categorical columns: One-hot encoding
categorical_cols = ["Product Type", "Customer Type", "Ad Campaign Source", "Time"]
df_final = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Features
features = [c for c in df_final.columns if c not in ["Date", "Units Sold", "Revenue", "Revenue Attributed to Each Campaign"]]
X = df_final[features]

# ---- Train Model ----
st.markdown("### Train Units Sold Prediction Model")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

if XGB_AVAILABLE:
    model = XGBRegressor(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        enable_categorical=False
    )
    st.info("Training XGBoost Regressor...")
else:
    model = RandomForestRegressor(n_estimators=300, random_state=42)
    st.info("Training RandomForest Regressor...")

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
st.success("Model trained successfully!")
from sklearn.metrics import mean_squared_error
import numpy as np

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
st.write(f"RMSE on test set: {rmse:.2f}")


# ---- Dashboard KPIs ----
st.markdown("### KPIs Overview")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Records", df_final.shape[0])
col2.metric("Total Units Sold", int(df["Units Sold"].sum()))
col3.metric("Total Revenue", f"${df['Revenue'].sum():.2f}")
col4.metric("Average Waste Rate", f"{df['Waste_Rate'].mean()*100:.1f}%")

# ---- Revenue by Product Type ----
st.markdown("---")
st.subheader("Revenue by Product Type")
prod_cols = [c for c in df_final.columns if c.startswith("Product Type_")]
prod_names = [c.replace("Product Type_","") for c in prod_cols]
prod_rev = pd.DataFrame({
    "Product Type": prod_names,
    "Revenue": [df_final[df_final[c]==1]["Revenue"].sum() for c in prod_cols]
})
fig1 = px.bar(prod_rev, x="Product Type", y="Revenue", text_auto=True, title="Revenue by Product Type")
st.plotly_chart(fig1, use_container_width=True)

# ---- Predict Units Sold for New Production ----
st.markdown("---")
st.subheader("Predict Units Sold for New Production")

prod_type = st.selectbox("Product Type", df_raw["Product Type"].unique())
time_of_day = st.selectbox("Time", df_raw["Time"].unique())
cust_type = st.selectbox("Customer Type", df_raw["Customer Type"].unique())
campaign = st.selectbox("Ad Campaign Source", df_raw["Ad Campaign Source"].unique())
ad_spend = st.number_input("Ad Spend", min_value=0, value=int(df["Ad Spend"].mean()))
units_produced = st.number_input("Units Produced", min_value=0, value=int(df["Units Produced"].mean()))
revenue_per_unit = st.number_input("Revenue per Unit", min_value=0.0, value=float(df["Revenue_per_Unit"].mean()))
waste_rate = st.number_input("Waste Rate", min_value=0.0, max_value=1.0, value=float(df["Waste_Rate"].mean()))

# Create new row for prediction
# القيم الرقمية
row_dict = {
    "Units Produced": df_final["Units Produced"].mean(),
    "Revenue": df_final["Revenue"].mean(),
    "Waste": df_final["Waste"].mean(),
    "Ad Spend": df_final["Ad Spend"].mean(),
    "Revenue Attributed to Each Campaign": df_final["Revenue Attributed to Each Campaign"].mean(),
    "Revenue_per_Unit": df_final["Revenue"].mean() / df_final["Units Produced"].mean(),
    "Waste_Rate": df_final["Waste"].mean() / df_final["Units Produced"].mean(),
    "Month": pd.Timestamp.now().month,
    "DayOfWeek": pd.Timestamp.now().weekday()
}

# القيم الفئوية
for col in [c for c in features if c.startswith("Time_")]:
    row_dict[col] = 1 if col == f"Time_{time_of_day}" else 0
for col in [c for c in features if c.startswith("Product Type_")]:
    row_dict[col] = 1 if col == f"Product Type_{prod_type}" else 0
for col in [c for c in features if c.startswith("Customer Type_")]:
    row_dict[col] = 1 if col == f"Customer Type_{cust_type}" else 0
for col in [c for c in features if c.startswith("Ad Campaign Source_")]:
    row_dict[col] = 1 if col == f"Ad Campaign Source_{campaign}" else 0

row = pd.DataFrame([row_dict])

# التنبؤ
pred_units = model.predict(row[features])[0]
st.metric("Predicted Units Sold", f"{pred_units:.0f}")
