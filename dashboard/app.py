import os
import requests
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta, timezone

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
IST = timezone(timedelta(hours=5, minutes=30))

st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon="🛡️",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Helper – safe API call
# ---------------------------------------------------------------------------

def api_get(endpoint, params=None):
    """GET request to the FastAPI backend. Returns (json, error_message)."""
    try:
        r = requests.get(f"{API_URL}{endpoint}", params=params, timeout=5)
        r.raise_for_status()
        return r.json(), None
    except requests.ConnectionError:
        return None, (
            "🔴 **Cannot reach the API.**  "
            "Make sure the FastAPI server is running in another terminal:  \n"
            "`uvicorn api.main:app --reload`"
        )
    except Exception as e:
        return None, f"🔴 API error: {e}"


def api_post(endpoint, payload):
    """POST request to the FastAPI backend. Returns (json, error_message)."""
    try:
        r = requests.post(f"{API_URL}{endpoint}", json=payload, timeout=10)
        r.raise_for_status()
        return r.json(), None
    except requests.ConnectionError:
        return None, (
            "🔴 **Cannot reach the API.**  "
            "Make sure the FastAPI server is running in another terminal:  \n"
            "`uvicorn api.main:app --reload`"
        )
    except requests.HTTPError as e:
        detail = ""
        try:
            detail = e.response.json().get("detail", "")
        except Exception:
            pass
        return None, f"🔴 API error ({e.response.status_code}): {detail or e}"
    except Exception as e:
        return None, f"🔴 API error: {e}"

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("🛡️ Fraud Detection Dashboard")
st.markdown(
    "Monitor your fraud detection model in real-time. "
    "Check individual transactions, review recent predictions, "
    "and track system health — all powered by the FastAPI prediction API."
)
st.divider()

# ---------------------------------------------------------------------------
# 1. System Status
# ---------------------------------------------------------------------------
st.header("⚙️ System Status")

health_data, health_err = api_get("/health")

if health_err:
    st.error(health_err)
else:
    col1, col2, col3 = st.columns(3)
    with col1:
        ok = health_data.get("status") == "ok"
        st.metric("API", "✅ Online" if ok else "❌ Offline")
    with col2:
        loaded = health_data.get("model_loaded", False)
        st.metric("Model", "✅ Loaded" if loaded else "❌ Not Loaded")
    with col3:
        db = health_data.get("db_connected", False)
        st.metric("Database", "✅ Connected" if db else "❌ Disconnected")

st.divider()

# ---------------------------------------------------------------------------
# 2. Check a Transaction
# ---------------------------------------------------------------------------
st.header("🔍 Check a Transaction")

SAMPLE_NORMAL = {
    "type": "CASH_OUT",
    "step": 14,
    "amount": 1000.0,
    "oldbalanceOrg": 5000.0,
    "newbalanceOrig": 4000.0,
    "oldbalanceDest": 10000.0,
    "newbalanceDest": 11000.0,
}
SAMPLE_FRAUD = {
    "type": "TRANSFER",
    "step": 3,
    "amount": 500000.0,
    "oldbalanceOrg": 500000.0,
    "newbalanceOrig": 0.0,
    "oldbalanceDest": 0.0,
    "newbalanceDest": 0.0,
}

# Sample buttons
btn_col1, btn_col2, _ = st.columns([1, 1, 3])
with btn_col1:
    if st.button("📗 Fill Normal Sample"):
        st.session_state["sample"] = SAMPLE_NORMAL
with btn_col2:
    if st.button("📕 Fill Fraud Sample"):
        st.session_state["sample"] = SAMPLE_FRAUD

defaults = st.session_state.get("sample", SAMPLE_NORMAL)

with st.form("predict_form"):
    form_col1, form_col2 = st.columns(2)
    with form_col1:
        tx_type = st.selectbox(
            "Transaction Type",
            ["TRANSFER", "CASH_OUT"],
            index=["TRANSFER", "CASH_OUT"].index(defaults["type"]),
        )
        step = st.number_input("Step (time in hours)", min_value=0, value=defaults["step"])
        amount = st.number_input("Amount", min_value=0.0, value=defaults["amount"], format="%.2f")
        oldbalanceOrg = st.number_input(
            "Sender Balance (before)", min_value=0.0, value=defaults["oldbalanceOrg"], format="%.2f"
        )
    with form_col2:
        newbalanceOrig = st.number_input(
            "Sender Balance (after)", min_value=0.0, value=defaults["newbalanceOrig"], format="%.2f"
        )
        oldbalanceDest = st.number_input(
            "Recipient Balance (before)", min_value=0.0, value=defaults["oldbalanceDest"], format="%.2f"
        )
        newbalanceDest = st.number_input(
            "Recipient Balance (after)", min_value=0.0, value=defaults["newbalanceDest"], format="%.2f"
        )

    submitted = st.form_submit_button("🚀 Predict", use_container_width=True)

if submitted:
    payload = {
        "step": step,
        "type": tx_type,
        "amount": amount,
        "oldbalanceOrg": oldbalanceOrg,
        "newbalanceOrig": newbalanceOrig,
        "oldbalanceDest": oldbalanceDest,
        "newbalanceDest": newbalanceDest,
    }
    result, err = api_post("/predict", payload)
    if err:
        st.error(err)
    else:
        prob = result["fraud_probability"]
        fraud = result["is_fraud"]
        threshold = result["threshold_used"]
        logged = result.get("logged", False)

        res_col1, res_col2, res_col3 = st.columns(3)
        with res_col1:
            st.metric("Fraud Probability", f"{prob:.2%}")
        with res_col2:
            if fraud:
                st.error("🚨 **FRAUD**")
            else:
                st.success("✅ **NOT FRAUD**")
        with res_col3:
            st.metric("Threshold", f"{threshold:.2f}")

        if logged:
            st.caption("✅ Prediction saved to database.")
        else:
            st.caption("⚠️ Prediction was NOT saved (database may be offline).")

st.divider()

# ---------------------------------------------------------------------------
# 3. Recent Predictions
# ---------------------------------------------------------------------------
st.header("📊 Recent Predictions")

refresh_col, _ = st.columns([1, 5])
with refresh_col:
    refresh = st.button("🔄 Refresh")

preds_data, preds_err = api_get("/predictions", params={"limit": 100})

if preds_err:
    st.warning(preds_err)
elif not preds_data:
    st.info("No predictions yet. Use the form above to make your first prediction!")
else:
    # --- Build DataFrame ---
    rows = []
    for p in preds_data:
        tx = p.get("transaction", {})
        ts_raw = p.get("timestamp")

        # Parse timestamp – handle ISO strings from the API
        if isinstance(ts_raw, str):
            # Strip trailing 'Z' and parse
            ts_raw = ts_raw.replace("Z", "+00:00")
            try:
                ts_utc = datetime.fromisoformat(ts_raw)
            except ValueError:
                ts_utc = None
        else:
            ts_utc = None

        # Convert to IST for display
        if ts_utc is not None:
            if ts_utc.tzinfo is None:
                ts_utc = ts_utc.replace(tzinfo=timezone.utc)
            ts_ist = ts_utc.astimezone(IST)
            time_str = ts_ist.strftime("%d %b %Y %I:%M:%S %p")
        else:
            time_str = "—"

        rows.append({
            "Time (IST)": time_str,
            "Type": tx.get("type", "—"),
            "Amount": tx.get("amount", 0),
            "Fraud Probability": p.get("fraud_probability", 0),
            "Result": "🚨 FRAUD" if p.get("is_fraud") else "✅ NOT FRAUD",
            "_ts_utc": ts_utc,  # kept for charting, hidden from table
        })

    df = pd.DataFrame(rows)

    # --- Summary Metrics ---
    total = len(df)
    fraud_count = sum(1 for r in rows if "FRAUD" in r["Result"] and "NOT" not in r["Result"])
    fraud_rate = fraud_count / total if total else 0
    avg_prob = df["Fraud Probability"].mean()

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Predictions", total)
    m2.metric("Flagged as Fraud", fraud_count)
    m3.metric("Fraud Rate", f"{fraud_rate:.1%}")
    m4.metric("Avg Fraud Prob", f"{avg_prob:.2%}")

    # --- Table ---
    st.subheader("Prediction Log")
    display_df = df.drop(columns=["_ts_utc"])
    st.dataframe(
        display_df,
        width="stretch",
        hide_index=True,
        column_config={
            "Amount": st.column_config.NumberColumn(format="%,.2f"),
            "Fraud Probability": st.column_config.ProgressColumn(
                min_value=0, max_value=1, format="%.2f"
            ),
        },
    )

    # --- Charts ---
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.subheader("Fraud vs Not Fraud")
        count_df = pd.DataFrame({
            "Category": ["Not Fraud", "Fraud"],
            "Count": [total - fraud_count, fraud_count],
            "Color": ["#2ecc71", "#e74c3c"],
        })
        st.bar_chart(
            count_df,
            x="Category",
            y="Count",
            color="Color",
            horizontal=False,
        )

    with chart_col2:
        st.subheader("Fraud Probability Over Time")
        chart_df = df[df["_ts_utc"].notna()].copy()
        if not chart_df.empty:
            chart_df = chart_df.sort_values("_ts_utc")
            chart_df["Time"] = chart_df["_ts_utc"].apply(
                lambda t: t.astimezone(IST).strftime("%H:%M:%S") if t else ""
            )
            st.line_chart(chart_df, x="Time", y="Fraud Probability")
        else:
            st.info("Not enough data to chart yet.")

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.divider()
st.caption(
    "Fraud Detection Dashboard • Powered by FastAPI + Random Forest + MongoDB"
)
