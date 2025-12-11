# app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

st.set_page_config(page_title="Retention & Marketing Dashboard", layout="wide")
st.title("💄 Client Retention & Marketing Dashboard")

# ---- Load Data ----
st.sidebar.header("Upload your Excel file")
data_file = st.sidebar.file_uploader("Upload data.xlsx", type=["xlsx"])

if data_file is None:
    st.warning("Please upload data.xlsx file to proceed.")
    st.stop()

df_raw = pd.read_excel(data_file, parse_dates=["Date"])
st.subheader("Raw Data Sample")
st.dataframe(df_raw.head())

# ---- Prepare Data ----
st.markdown("### Data Preparation & Feature Engineering")
df = df_raw.copy()
df['Return_Visit'] = (df['Conversions'] > 0).astype(int)

rfm = df.groupby("Customer Type").agg(
    last_purchase=("Date", "max"),
    frequency=("Date", "count"),
    total_spent=("Revenue", "sum"),
    avg_spent=("Revenue", "mean")
).reset_index()
reference_date = df["Date"].max()
rfm["recency_days"] = (reference_date - rfm["last_purchase"]).dt.days

df_final = df.merge(rfm, on="Customer Type", how="left")
df_final["purchase_month"] = df_final["Date"].dt.month
df_final["purchase_dayofweek"] = df_final["Date"].dt.dayofweek

categorical_cols = ["Channel", "Service Type", "Time of Day"]
df_final = pd.get_dummies(df_final, columns=categorical_cols, drop_first=True)

st.subheader("Prepared Dataset Sample")
st.dataframe(df_final.head())

# ---- Train Model ----
st.markdown("### Train Retention Model")
all_features = [c for c in df_final.columns if c not in ["Customer Type", "Date", "Return_Visit", "last_purchase"]]
X = df_final[all_features]
y = df_final["Return_Visit"]

model = None
if len(y.unique()) < 2:
    st.warning("Cannot train model: Return_Visit has only one class present in the data.")
else:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
    st.info("Training RandomForest...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    st.success("Model trained successfully!")
    st.write(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
    st.write(f"ROC AUC: {roc_auc_score(y_test, model.predict_proba(X_test)[:,1]):.3f}")

# ---- KPIs ----
st.markdown("### KPIs Overview")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Records", df_final.shape[0])
col2.metric("Overall Return Rate", f"{df_final['Return_Visit'].mean()*100:.1f}%")
col3.metric("Avg Revenue", f"${df_final['Revenue'].mean():.2f}")
col4.metric("Avg Frequency", f"{df_final['frequency'].mean():.1f}")

# ---- Revenue by Channel ----
st.markdown("---")
st.subheader("Revenue by Channel")
channel_cols = [c for c in df_final.columns if c.startswith("Channel_")]
channel_names = [c.replace("Channel_", "") for c in channel_cols]
channel_rev = pd.DataFrame({
    "Channel": channel_names,
    "Revenue": [df_final[df_final[c]==1]["Revenue"].sum() for c in channel_cols]
})
st.bar_chart(channel_rev.set_index("Channel"))

# ---- Return Rate by Service Type ----
st.subheader("Return Rate by Service Type")
service_cols = [c for c in df_final.columns if c.startswith("Service Type_")]
service_names = [c.replace("Service Type_", "") for c in service_cols]
service_rate = pd.DataFrame({
    "Service Type": service_names,
    "Return Rate": [df_final[df_final[c]==1]["Return_Visit"].mean() for c in service_cols]
})
st.bar_chart(service_rate.set_index("Service Type"))

# ---- Predict Return Probability ----
st.markdown("---")
st.subheader("Predict Return Probability")
if model is not None:
    cust_type = st.selectbox("Customer Type", df_final["Customer Type"].unique())
    service_type = st.selectbox("Service Type", ["Deluxe","Premium"])
    time_of_day = st.selectbox("Time of Day", ["Morning","Afternoon","Evening"])
    channel = st.selectbox("Channel", ["Online", "Phone", "Walk-in", "Social Media"])

    row_dict = {
        "Ad Spend": df_final["Ad Spend"].mean(),
        "Conversions": df_final["Conversions"].mean(),
        "Revenue": df_final["Revenue"].mean(),
        "Email_Engagement": 1,
        "Discount_Used": 0,
        "frequency": df_final["frequency"].mean(),
        "total_spent": df_final["total_spent"].mean(),
        "avg_spent": df_final["avg_spent"].mean(),
        "recency_days": df_final["recency_days"].mean(),
        "purchase_month": pd.Timestamp.now().month,
        "purchase_dayofweek": pd.Timestamp.now().weekday()
    }

    for col in all_features:
        if col.startswith("Channel_"):
            row_dict[col] = 1 if col == f"Channel_{channel}" else 0
        elif col.startswith("Service Type_"):
            row_dict[col] = 1 if col == f"Service Type_{service_type}" else 0
        elif col.startswith("Time of Day_"):
            row_dict[col] = 1 if col == f"Time of Day_{time_of_day}" else 0
        else:
            if col not in row_dict:
                row_dict[col] = 0

    row = pd.DataFrame([row_dict])
    proba = model.predict_proba(row[all_features])[0][1]
    st.metric("Predicted return probability", f"{proba*100:.1f}%")
else:
    st.info("Model not trained due to insufficient class diversity in Return_Visit.")
