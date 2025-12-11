import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Retention & Marketing Dashboard", layout="wide")
st.title("💄 Client Retention & Marketing Dashboard")

# ---- Upload Data ----
data_file = st.file_uploader("Upload your Excel file", type=["xlsx"])
if data_file is None:
    st.warning("Please upload data.xlsx file to proceed.")
    st.stop()

df = pd.read_excel(data_file, parse_dates=["Date"])
st.subheader("Raw Data Sample")
st.dataframe(df.head())

# ---- Feature Engineering ----
df['Return_Visit'] = (df['Conversions'] > 0).astype(int)

rfm = df.groupby("Customer Type").agg(
    last_purchase=("Date", "max"),
    frequency=("Date", "count"),
    total_spent=("Revenue", "sum"),
    avg_spent=("Revenue", "mean")
).reset_index()
rfm["recency_days"] = (df["Date"].max() - rfm["last_purchase"]).dt.days

df_final = df.merge(rfm, on="Customer Type", how="left")

# ---- KPIs ----
st.markdown("### KPIs Overview")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Records", df_final.shape[0])
col2.metric("Overall Return Rate", f"{df_final['Return_Visit'].mean()*100:.1f}%")
col3.metric("Avg Revenue", f"${df_final['Revenue'].mean():.2f}")
col4.metric("Avg Frequency", f"{df_final['frequency'].mean():.1f}")

# ---- Revenue by Channel ----
st.markdown("### Revenue by Channel")
channels = df['Channel'].unique()
channel_rev = {ch: df[df['Channel']==ch]['Revenue'].sum() for ch in channels}
st.bar_chart(pd.DataFrame.from_dict(channel_rev, orient='index', columns=['Revenue']))

# ---- Return Rate by Service Type ----
st.markdown("### Return Rate by Service Type")
services = df['Service Type'].unique()
service_rate = {s: df[df['Service Type']==s]['Return_Visit'].mean() for s in services}
st.bar_chart(pd.DataFrame.from_dict(service_rate, orient='index', columns=['Return Rate']))

# ---- Predict Return Probability (RFM-based simple rule) ----
st.markdown("### Predict Return Probability (RFM-based)")
cust_type = st.selectbox("Customer Type", df_final["Customer Type"].unique())

cust_rfm = rfm[rfm['Customer Type']==cust_type].iloc[0]
# قاعدة بسيطة: Recency منخفض + Frequency عالي => احتمال العودة أعلى
prob = min(1.0, max(0.0, 0.5 + (cust_rfm['frequency']/df_final['frequency'].max()*0.3) - 
                        (cust_rfm['recency_days']/df_final['recency_days'].max()*0.3)))
st.metric("Predicted return probability", f"{prob*100:.1f}%")
