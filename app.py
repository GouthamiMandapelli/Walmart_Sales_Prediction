"""
Walmart Sales Prediction — Streamlit Application
Stage 7

Predicts weekly sales for a Walmart store/department using the
approved Stage-1 model (Lag-52 Baseline).

The Lag-52 Baseline predicts this week's sales as the same
week's sales from 52 weeks ago (same store, same department).
It is the cross-validated best model by WMAE.

Usage:
    streamlit run app.py
"""

import pickle
from datetime import date, timedelta

import numpy as np
import pandas as pd
import streamlit as st

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Walmart Sales Prediction",
    page_icon="🛒",
    layout="centered",
)

# ── Constants ──────────────────────────────────────────────────────────────────
KNOWN_HOLIDAY_WEEKS = {6, 36, 47, 52}   # Super Bowl, Labor Day, Thanksgiving, Christmas

TYPE_ENCODING = {"A": 0, "B": 1, "C": 2}

FEATURES = [
    "Store", "Dept",
    "Year", "Month", "Week", "Quarter",
    "Type",
    "IsHoliday_int", "Is_Known_Holiday",
    "Temperature", "Fuel_Price",
    "MarkDown1", "MarkDown2", "MarkDown3", "MarkDown4", "MarkDown5",
    "MarkDown1_active", "MarkDown2_active", "MarkDown3_active",
    "MarkDown4_active", "MarkDown5_active",
    "CPI", "Unemployment",
    "Size",
    "lag_52",
]

# ── Cached data loaders ────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_train():
    return pd.read_csv("train_features.csv", parse_dates=["Date"])


@st.cache_data(show_spinner=False)
def load_stores():
    return pd.read_csv("stores.csv")


@st.cache_resource(show_spinner=False)
def load_model():
    with open("model_outputs/selected_model.pkl", "rb") as f:
        return pickle.load(f)


@st.cache_resource(show_spinner=False)
def load_meta():
    with open("model_outputs/model_meta.pkl", "rb") as f:
        return pickle.load(f)


# ── Helper functions ───────────────────────────────────────────────────────────
def get_lag52(train_df: pd.DataFrame, store: int, dept: int,
              target_date: pd.Timestamp) -> float | None:
    """
    Look up Weekly_Sales for (store, dept) exactly 52 weeks before target_date.
    Returns None if no record found (handled by imputation below).
    """
    lag_date = target_date - pd.DateOffset(weeks=52)
    # Allow ±3 day tolerance for weekly Friday alignment
    tolerance = pd.Timedelta(days=3)
    mask = (
        (train_df["Store"] == store) &
        (train_df["Dept"] == dept) &
        (train_df["Date"] >= lag_date - tolerance) &
        (train_df["Date"] <= lag_date + tolerance)
    )
    subset = train_df[mask]
    if len(subset) > 0:
        return float(subset.iloc[0]["Weekly_Sales"])
    return None


def impute_lag52_from_train(train_df: pd.DataFrame, store: int,
                             dept: int) -> float:
    """
    Fallback imputation when no exact lag-52 record exists.
    Uses the per-(Store, Dept) median from the training set.
    Falls back to dept median, then global median.
    """
    pair = train_df[(train_df["Store"] == store) & (train_df["Dept"] == dept)]
    if len(pair) > 0 and pair["lag_52"].notna().any():
        return float(pair["lag_52"].median())

    dept_vals = train_df[train_df["Dept"] == dept]["lag_52"].dropna()
    if len(dept_vals) > 0:
        return float(dept_vals.median())

    return float(train_df["lag_52"].median())


def derive_calendar_features(input_date: date) -> dict:
    """Derive Year, Month, Week, Quarter, Is_Known_Holiday from a date."""
    ts = pd.Timestamp(input_date)
    week = int(ts.isocalendar().week)
    return {
        "Year":             ts.year,
        "Month":            ts.month,
        "Week":             week,
        "Quarter":          ts.quarter,
        "Is_Known_Holiday": int(week in KNOWN_HOLIDAY_WEEKS),
    }


def build_feature_row(
    store, dept, input_date, is_holiday,
    temperature, fuel_price,
    md1, md2, md3, md4, md5,
    cpi, unemployment, store_type, store_size,
    lag52_value,
) -> pd.DataFrame:
    """Assemble a single-row DataFrame of model features."""
    cal = derive_calendar_features(input_date)
    row = {
        "Store":           store,
        "Dept":            dept,
        "Year":            cal["Year"],
        "Month":           cal["Month"],
        "Week":            cal["Week"],
        "Quarter":         cal["Quarter"],
        "Type":            TYPE_ENCODING[store_type],
        "IsHoliday_int":   int(is_holiday),
        "Is_Known_Holiday": cal["Is_Known_Holiday"],
        "Temperature":     temperature,
        "Fuel_Price":      fuel_price,
        "MarkDown1":       md1,
        "MarkDown2":       md2,
        "MarkDown3":       md3,
        "MarkDown4":       md4,
        "MarkDown5":       md5,
        "MarkDown1_active": int(md1 > 0),
        "MarkDown2_active": int(md2 > 0),
        "MarkDown3_active": int(md3 > 0),
        "MarkDown4_active": int(md4 > 0),
        "MarkDown5_active": int(md5 > 0),
        "CPI":             cpi,
        "Unemployment":    unemployment,
        "Size":            store_size,
        "lag_52":          lag52_value,
    }
    return pd.DataFrame([row])[FEATURES]


# ── Load resources ─────────────────────────────────────────────────────────────
with st.spinner("Loading data and model …"):
    train_df = load_train()
    stores_df = load_stores()
    model    = load_model()
    meta     = load_meta()

# ── Header ─────────────────────────────────────────────────────────────────────
st.title("🛒 Walmart Sales Prediction")
st.markdown(
    """
    This application predicts **weekly sales** for a Walmart store and
    department using historical patterns.

    The model uses the **Lag-52 Baseline** — the same-week sales from
    52 weeks prior — which achieved the best cross-validated
    **Weighted MAE** (WMAE) across three chronological validation folds.

    Fill in the inputs below and click **Predict** to get a weekly
    sales estimate.
    """
)

st.divider()

# ── Store selector auto-fills Type and Size ────────────────────────────────────
store_type_map = dict(zip(stores_df["Store"], stores_df["Type"]))
store_size_map = dict(zip(stores_df["Store"], stores_df["Size"]))
dept_options   = sorted(train_df["Dept"].unique().tolist())

# ── Input form ─────────────────────────────────────────────────────────────────
st.subheader("Input Parameters")

col1, col2 = st.columns(2)

with col1:
    store = st.selectbox(
        "Store", options=sorted(train_df["Store"].unique().tolist()),
        help="Walmart store number (1–45)"
    )
    dept = st.selectbox(
        "Department", options=dept_options,
        help="Department number within the store"
    )
    input_date = st.date_input(
        "Week Date",
        value=date(2012, 10, 26),
        min_value=date(2010, 1, 1),
        max_value=date(2015, 12, 31),
        help="The Friday of the sales week"
    )
    is_holiday = st.checkbox("Is Holiday Week", value=False)

with col2:
    # Auto-fill Type and Size from stores.csv, allow override
    default_type = store_type_map.get(store, "A")
    default_size = store_size_map.get(store, 150000)
    store_type = st.selectbox(
        "Store Type", options=["A", "B", "C"],
        index=["A", "B", "C"].index(default_type),
        help="Store type: A (largest) → C (smallest)"
    )
    store_size = st.number_input(
        "Store Size (sq ft)", min_value=10000, max_value=300000,
        value=int(default_size), step=1000,
        help="Total store floor area in square feet"
    )

st.subheader("Economic & Environmental")
col3, col4 = st.columns(2)
with col3:
    temperature  = st.number_input("Temperature (°F)", value=60.0,
                                   min_value=-30.0, max_value=120.0, step=0.1)
    fuel_price   = st.number_input("Fuel Price ($/gal)", value=3.50,
                                   min_value=1.0, max_value=6.0, step=0.01)
with col4:
    cpi          = st.number_input("CPI", value=211.0,
                                   min_value=100.0, max_value=250.0, step=0.1)
    unemployment = st.number_input("Unemployment Rate (%)", value=8.0,
                                   min_value=1.0, max_value=20.0, step=0.1)

st.subheader("MarkDown Promotions (enter 0 if not active)")
col5, col6, col7 = st.columns(3)
with col5:
    md1 = st.number_input("MarkDown1", value=0.0, min_value=0.0, step=100.0)
    md2 = st.number_input("MarkDown2", value=0.0, min_value=0.0, step=100.0)
with col6:
    md3 = st.number_input("MarkDown3", value=0.0, min_value=0.0, step=100.0)
    md4 = st.number_input("MarkDown4", value=0.0, min_value=0.0, step=100.0)
with col7:
    md5 = st.number_input("MarkDown5", value=0.0, min_value=0.0, step=100.0)

st.divider()

# ── Predict ────────────────────────────────────────────────────────────────────
if st.button("Predict Weekly Sales", type="primary", use_container_width=True):

    target_ts = pd.Timestamp(input_date)

    # Resolve lag_52
    lag52 = get_lag52(train_df, store, dept, target_ts)
    lag52_source = "exact historical lookup"
    if lag52 is None:
        lag52 = impute_lag52_from_train(train_df, store, dept)
        lag52_source = "training-set median imputation"

    # Build feature row
    X = build_feature_row(
        store=store, dept=dept, input_date=input_date,
        is_holiday=is_holiday,
        temperature=temperature, fuel_price=fuel_price,
        md1=md1, md2=md2, md3=md3, md4=md4, md5=md5,
        cpi=cpi, unemployment=unemployment,
        store_type=store_type, store_size=store_size,
        lag52_value=lag52,
    )

    # Make prediction
    # Lag-52 Baseline: prediction = lag_52 directly
    if isinstance(model, dict) and model.get("model") == "Lag-52 Baseline":
        prediction = lag52
    else:
        # Fallback for any sklearn-compatible model saved in future stages
        prediction = float(model.predict(X)[0])

    # ── Results card ───────────────────────────────────────────────────────────
    st.subheader("Prediction Result")

    res_col1, res_col2, res_col3 = st.columns(3)
    res_col1.metric("Store", f"#{store}")
    res_col2.metric("Department", f"#{dept}")
    res_col3.metric("Week", input_date.strftime("%b %d, %Y"))

    st.success(f"### Predicted Weekly Sales:  **${prediction:,.2f}**")

    # Context details
    cal = derive_calendar_features(input_date)
    with st.expander("Prediction details", expanded=True):
        detail_col1, detail_col2 = st.columns(2)
        with detail_col1:
            st.markdown(f"**Model:** {meta['selected_model_name']}")
            st.markdown(f"**lag_52 value:** ${lag52:,.2f}")
            st.markdown(f"**lag_52 source:** {lag52_source}")
            st.markdown(f"**Is Holiday Week:** {'Yes' if is_holiday else 'No'}")
            st.markdown(f"**Is Known Holiday Week:** {'Yes' if cal['Is_Known_Holiday'] else 'No'}")
        with detail_col2:
            st.markdown(f"**Store Type:** {store_type}")
            st.markdown(f"**Store Size:** {store_size:,} sq ft")
            st.markdown(f"**Year/Month/Week/Quarter:** "
                        f"{cal['Year']} / {cal['Month']} / "
                        f"{cal['Week']} / Q{cal['Quarter']}")
            st.markdown(f"**CPI:** {cpi:.1f}  |  **Unemployment:** {unemployment:.1f}%")

    st.caption(
        "The Lag-52 Baseline predicts this week's sales as the same "
        "store/department sales from exactly 52 weeks ago. "
        "It achieved the best average WMAE (2,278) across three "
        "chronological walk-forward validation folds."
    )

# ── Sidebar — model info ───────────────────────────────────────────────────────
with st.sidebar:
    st.header("Model Information")
    st.markdown(f"**Selected Model:** {meta['selected_model_name']}")

    try:
        val_df = pd.read_csv("model_outputs/model_validation_results.csv")
        st.markdown("**Cross-Validation WMAE**")
        display_cols = ["Model", "Fold 1", "Fold 2", "Fold 3", "Avg_CV_WMAE"]
        available = [c for c in display_cols if c in val_df.columns]
        st.dataframe(val_df[available].set_index("Model"), use_container_width=True)
        st.caption("Lower WMAE = better. Holiday weeks weighted 5×.")
    except FileNotFoundError:
        st.info("Validation results not found.")

    st.divider()
    st.markdown("**Feature Set (Stage 1)**")
    st.caption("  \n".join(meta["features"]))

    st.divider()
    st.markdown("**Data**")
    st.caption(
        f"Training: 421,570 rows  \n"
        f"Stores: 45  |  Depts: 81  \n"
        f"Date range: 2010-02-05 → 2012-10-26"
    )
