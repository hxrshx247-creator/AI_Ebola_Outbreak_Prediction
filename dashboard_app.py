"""
Ebola Outbreak Prediction Dashboard
Member 2 & 3 combined prototype: visualizes historical data, model predictions,
and evaluation for the Random Forest / Gradient Boosting models trained on
NewCases (incident case) forecasting.

Run with:  streamlit run app.py
Expects these files in the SAME folder:
    ebola_environmental_merged.csv
    rf_model_newcases.pkl
    gb_model_newcases.pkl
    feature_cols.pkl
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os

st.set_page_config(page_title="Ebola Outbreak Prediction", layout="wide")

DATA_PATH = "ebola_environmental_merged.csv"
RF_PATH = "rf_model_newcases.pkl"
GB_PATH = "gb_model_newcases.pkl"
FEATURES_PATH = "feature_cols.pkl"


# -----------------------------
# Data + feature pipeline (mirrors the training notebook)
# -----------------------------

@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.rename(columns={
        "Cumulative no. of confirmed, probable and suspected cases": "Cases",
        "Cumulative no. of confirmed, probable and suspected deaths": "Deaths",
    })
    return df.sort_values(["Country", "Date"]).reset_index(drop=True)


@st.cache_data
def build_features(df, lags=(1, 2, 3, 7), rolling_windows=(3, 7, 14)):
    out_frames = []
    for country, g in df.groupby("Country"):
        g = g.sort_values("Date").reset_index(drop=True)
        g["NewCases"] = g["Cases"].diff().clip(lower=0)
        g["NewDeaths"] = g["Deaths"].diff().clip(lower=0)

        base_cols = ["Cases", "Deaths", "NewCases", "NewDeaths",
                     "Temperature", "Rainfall", "Relative_Humidity"]
        for col in base_cols:
            for lag in lags:
                g[f"{col}_lag{lag}"] = g[col].shift(lag)

        for col in ["Cases", "Deaths", "NewCases", "Temperature", "Rainfall"]:
            shifted = g[col].shift(1)
            for w in rolling_windows:
                g[f"{col}_rollmean{w}"] = shifted.rolling(w).mean()
                g[f"{col}_rollstd{w}"] = shifted.rolling(w).std()

        g["days_since_prev"] = g["Date"].diff().dt.days
        out_frames.append(g)
    return pd.concat(out_frames, ignore_index=True)


@st.cache_resource
def load_models():
    missing = [p for p in [RF_PATH, GB_PATH, FEATURES_PATH] if not os.path.exists(p)]
    if missing:
        return None, None, None, missing
    rf = joblib.load(RF_PATH)
    gb = joblib.load(GB_PATH)
    feature_cols = joblib.load(FEATURES_PATH)
    return rf, gb, feature_cols, []


# -----------------------------
# App
# -----------------------------

st.title("🦠 Ebola Outbreak Prediction Dashboard")
st.caption("Guinea & Sierra Leone, 2014–2016 · Random Forest & Gradient Boosting forecasting new cases")

if not os.path.exists(DATA_PATH):
    st.error(f"Couldn't find `{DATA_PATH}`. Put it in the same folder as this app.")
    st.stop()

df = load_data(DATA_PATH)
feat_df = build_features(df)
rf, gb, feature_cols, missing = load_models()

countries = sorted(df["Country"].unique())
st.sidebar.header("Filters")
selected_countries = st.sidebar.multiselect("Country", countries, default=countries)
date_min, date_max = df["Date"].min(), df["Date"].max()
date_range = st.sidebar.date_input("Date range", (date_min, date_max),
                                    min_value=date_min, max_value=date_max)

if len(date_range) == 2:
    start, end = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
else:
    start, end = date_min, date_max

view = df[(df["Country"].isin(selected_countries)) &
          (df["Date"] >= start) & (df["Date"] <= end)]

tab_overview, tab_trends, tab_predict, tab_importance, tab_about = st.tabs(
    ["📊 Overview", "📈 Historical Trends", "🔮 Predictions", "🌟 Feature Importance", "ℹ️ About"]
)

# --- Overview ---
with tab_overview:
    col1, col2, col3, col4 = st.columns(4)
    latest = view.sort_values("Date").groupby("Country").tail(1)
    col1.metric("Countries shown", len(selected_countries))
    col2.metric("Total cumulative cases (latest)", int(latest["Cases"].sum()))
    col3.metric("Total cumulative deaths (latest)", int(latest["Deaths"].sum()))
    col4.metric("Date range", f"{start.date()} → {end.date()}")

    fig = px.line(view, x="Date", y="Cases", color="Country",
                   title="Cumulative Cases Over Time")
    st.plotly_chart(fig, use_container_width=True)

    fig2 = px.line(view, x="Date", y="Deaths", color="Country",
                    title="Cumulative Deaths Over Time")
    st.plotly_chart(fig2, use_container_width=True)

# --- Historical trends ---
with tab_trends:
    metric = st.selectbox("Metric", ["Temperature", "Rainfall", "Relative_Humidity"])
    fig3 = px.line(view, x="Date", y=metric, color="Country", title=f"{metric} Over Time")
    st.plotly_chart(fig3, use_container_width=True)

    new_cases_view = feat_df[(feat_df["Country"].isin(selected_countries)) &
                              (feat_df["Date"] >= start) & (feat_df["Date"] <= end)]
    fig4 = px.bar(new_cases_view, x="Date", y="NewCases", color="Country",
                   title="New (Incident) Cases per Report", barmode="group")
    st.plotly_chart(fig4, use_container_width=True)

# --- Predictions ---
with tab_predict:
    if missing:
        st.warning(f"Model files not found: {', '.join(missing)}. "
                    f"Place them in the same folder as this app to enable predictions.")
    else:
        st.subheader("Model vs. Actual — New Cases (test period)")

        eval_df = feat_df.dropna(subset=feature_cols + ["NewCases"]).copy()
        eval_df["target"] = eval_df.groupby("Country")["NewCases"].shift(-1)
        eval_df = eval_df.dropna(subset=["target"])

        # last 20% per country = same test split logic as training
        test_frames = []
        for country, g in eval_df.groupby("Country"):
            g = g.sort_values("Date")
            split_idx = int(len(g) * 0.8)
            test_frames.append(g.iloc[split_idx:])
        test = pd.concat(test_frames).sort_values("Date")
        test = test[test["Country"].isin(selected_countries)]

        if len(test) == 0:
            st.info("No test-period data for the selected filters — widen the date range or country selection.")
        else:
            X_test = test[feature_cols]
            rf_pred = rf.predict(X_test)
            gb_pred = gb.predict(X_test)

            fig5 = go.Figure()
            fig5.add_trace(go.Scatter(x=test["Date"], y=test["target"], name="Actual", mode="lines+markers"))
            fig5.add_trace(go.Scatter(x=test["Date"], y=rf_pred, name="Random Forest", mode="lines+markers"))
            fig5.add_trace(go.Scatter(x=test["Date"], y=gb_pred, name="Gradient Boosting", mode="lines+markers"))
            fig5.update_layout(title="Actual vs. Predicted New Cases (t+1)", xaxis_title="Date", yaxis_title="New Cases")
            st.plotly_chart(fig5, use_container_width=True)

            mae_rf = np.mean(np.abs(test["target"].values - rf_pred))
            mae_gb = np.mean(np.abs(test["target"].values - gb_pred))
            naive_mae = np.mean(np.abs(test["target"].values - test["NewCases"].values))

            c1, c2, c3 = st.columns(3)
            c1.metric("Random Forest MAE", f"{mae_rf:.2f}")
            c2.metric("Gradient Boosting MAE", f"{mae_gb:.2f}")
            c3.metric("Naive baseline MAE", f"{naive_mae:.2f}",
                      help="Predicting 'same as last report' — the sanity-check baseline")

            st.caption("Note: models don't currently beat the naive baseline on this small dataset — "
                       "see the project write-up for why this is an expected, honest finding.")

# --- Feature importance ---
with tab_importance:
    if missing:
        st.warning("Model files not found — see the Predictions tab for details.")
    else:
        importances = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False).head(15)
        fig6 = px.bar(importances[::-1], orientation="h",
                       title="Top 15 Feature Importances (Random Forest)",
                       labels={"value": "Importance", "index": "Feature"})
        st.plotly_chart(fig6, use_container_width=True)

# --- About ---
with tab_about:
    st.markdown("""
    ### About this dashboard
    - **Member 1** collected and cleaned the case/death and environmental (temperature, rainfall) data.
    - **Member 2** engineered lag/rolling features and trained the Random Forest and Gradient
      Boosting models shown here, evaluated with MAE, RMSE, R², and MASE.
    - **Member 3** built this dashboard to surface the predictions and data for exploration.

    **Target being predicted:** `NewCases` — the incident (new) case count at the next report,
    derived from the raw cumulative case totals. This target was chosen over predicting cumulative
    cases directly because cumulative counts barely change day-to-day, making a trivial
    "predict yesterday's value" baseline misleadingly accurate.
    """)
