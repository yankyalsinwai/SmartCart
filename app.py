"""
SmartCart -- Online Shopper Purchase Intent Predictor
------------------------------------------------------
Streamlit front-end for the tuned Random Forest pipeline trained in
SmartCart_Online_Shoppers_Program_Code_Executed.ipynb

This app ONLY loads and serves the already-trained scikit-learn Pipeline
(ColumnTransformer + RandomForestClassifier) saved as smartcart_model.pkl.
No training/fitting logic lives here.

Run with:  streamlit run app.py
"""

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# --------------------------------------------------------------------------
# Page configuration
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="SmartCart | Purchase Intent Predictor",
    page_icon="🛒",
    layout="wide",
)

MODEL_PATH = "smartcart_model.pkl"

# --------------------------------------------------------------------------
# Model loading (cached so it only loads once per session)
# --------------------------------------------------------------------------
@st.cache_resource
def load_model(path: str):
    return joblib.load(path)


try:
    model = load_model(MODEL_PATH)
    model_loaded = True
except FileNotFoundError:
    model = None
    model_loaded = False

# --------------------------------------------------------------------------
# Reference option lists (matching categories seen in the training data)
# --------------------------------------------------------------------------
MONTH_OPTIONS = ["Feb", "Mar", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
VISITOR_TYPE_OPTIONS = ["Returning_Visitor", "New_Visitor", "Other"]
OS_OPTIONS = list(range(1, 9))          # OperatingSystems: 1-8
BROWSER_OPTIONS = list(range(1, 14))    # Browser: 1-13
REGION_OPTIONS = list(range(1, 10))     # Region: 1-9
TRAFFIC_TYPE_OPTIONS = list(range(1, 21))  # TrafficType: 1-20

# --------------------------------------------------------------------------
# Feature engineering (must mirror add_engineered_features() in the notebook)
# --------------------------------------------------------------------------
def add_engineered_features(row: dict) -> dict:
    row = dict(row)
    row["TotalPages"] = (
        row["Administrative"] + row["Informational"] + row["ProductRelated"]
    )
    row["TotalDuration"] = (
        row["Administrative_Duration"]
        + row["Informational_Duration"]
        + row["ProductRelated_Duration"]
    )
    row["ProductDurationPerPage"] = row["ProductRelated_Duration"] / (
        row["ProductRelated"] + 1
    )
    row["ExitBounceGap"] = row["ExitRates"] - row["BounceRates"]
    return row


def build_input_dataframe(raw_inputs: dict) -> pd.DataFrame:
    engineered = add_engineered_features(raw_inputs)
    return pd.DataFrame([engineered])


# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------
st.title("🛒 SmartCart")
st.caption(
    "Predict, in real time, whether a browsing session will end in a purchase -- "
    "so marketing and UX teams can act on high-intent shoppers before they leave."
)

if not model_loaded:
    st.error(
        f"Could not find **{MODEL_PATH}**. Please run the notebook's final cell "
        "to export the tuned pipeline with `joblib.dump(tuned_rf, "
        f"'{MODEL_PATH}')`, then place the file next to this app and refresh."
    )

st.divider()

left, right = st.columns([1, 1.3], gap="large")

# --------------------------------------------------------------------------
# LEFT: Session details input form
# --------------------------------------------------------------------------
with left:
    st.subheader("Session details")

    with st.form("session_form"):
        st.markdown("**Browsing activity**")
        c1, c2 = st.columns(2)
        with c1:
            administrative = st.slider("Administrative pages viewed", 0, 30, 2)
            informational = st.slider("Informational pages viewed", 0, 25, 0)
            product_related = st.slider("Product-related pages viewed", 0, 700, 30)
        with c2:
            administrative_duration = st.number_input(
                "Administrative duration (sec)", 0.0, 3400.0, 40.0, step=10.0
            )
            informational_duration = st.number_input(
                "Informational duration (sec)", 0.0, 2500.0, 0.0, step=10.0
            )
            product_related_duration = st.number_input(
                "Product-related duration (sec)", 0.0, 64000.0, 600.0, step=10.0
            )

        st.markdown("**Engagement signals**")
        c3, c4 = st.columns(2)
        with c3:
            bounce_rates = st.slider("Bounce rate", 0.0, 0.2, 0.02, step=0.001, format="%.3f")
            exit_rates = st.slider("Exit rate", 0.0, 0.2, 0.04, step=0.001, format="%.3f")
        with c4:
            page_values = st.slider("Page value", 0.0, 360.0, 5.0, step=1.0)
            special_day = st.slider("Closeness to a special day", 0.0, 1.0, 0.0, step=0.1)

        st.markdown("**Visit context**")
        c5, c6 = st.columns(2)
        with c5:
            month = st.selectbox("Month", MONTH_OPTIONS, index=MONTH_OPTIONS.index("Nov"))
            visitor_type = st.selectbox("Visitor type", VISITOR_TYPE_OPTIONS)
            weekend = st.toggle("Weekend session", value=False)
        with c6:
            operating_systems = st.selectbox("Operating system (code)", OS_OPTIONS, index=1)
            browser = st.selectbox("Browser (code)", BROWSER_OPTIONS, index=0)
            region = st.selectbox("Region (code)", REGION_OPTIONS, index=0)
        traffic_type = st.selectbox("Traffic type (code)", TRAFFIC_TYPE_OPTIONS, index=1)

        submitted = st.form_submit_button(
            "🔮 Predict purchase intent", use_container_width=True, disabled=not model_loaded
        )

# --------------------------------------------------------------------------
# RIGHT: Prediction result
# --------------------------------------------------------------------------
with right:
    st.subheader("Prediction")

    if submitted and model_loaded:
        raw_inputs = {
            "Administrative": administrative,
            "Administrative_Duration": administrative_duration,
            "Informational": informational,
            "Informational_Duration": informational_duration,
            "ProductRelated": product_related,
            "ProductRelated_Duration": product_related_duration,
            "BounceRates": bounce_rates,
            "ExitRates": exit_rates,
            "PageValues": page_values,
            "SpecialDay": special_day,
            "Month": month,
            "OperatingSystems": operating_systems,
            "Browser": browser,
            "Region": region,
            "TrafficType": traffic_type,
            "VisitorType": visitor_type,
            "Weekend": weekend,
        }

        input_df = build_input_dataframe(raw_inputs)

        probability = float(model.predict_proba(input_df)[:, 1][0])
        will_purchase = probability >= 0.50

        if will_purchase:
            st.success("### ✅ Likely to Purchase")
        else:
            st.warning("### ⚪ Unlikely to Purchase")

        m1, m2 = st.columns(2)
        m1.metric("Purchase probability", f"{probability * 100:.1f}%")
        m2.metric("Predicted outcome", "Purchase" if will_purchase else "No purchase")

        st.progress(min(max(probability, 0.0), 1.0))

        st.markdown("**Recommended action**")
        if probability >= 0.70:
            st.info(
                "High intent -- consider a timely incentive (e.g., a limited-time "
                "discount or free-shipping nudge) to convert this session now."
            )
        elif probability >= 0.50:
            st.info(
                "Moderate-to-high intent -- a light-touch prompt (e.g., a helpful "
                "product recommendation) may tip this session toward purchase."
            )
        else:
            st.info(
                "Low intent -- avoid costly interventions; a passive experience "
                "is likely sufficient for this session."
            )

        with st.expander("View engineered features used by the model"):
            engineered_only = {
                k: v
                for k, v in input_df.iloc[0].to_dict().items()
                if k in ["TotalPages", "TotalDuration", "ProductDurationPerPage", "ExitBounceGap"]
            }
            st.json({k: round(v, 3) if isinstance(v, float) else v for k, v in engineered_only.items()})

    elif not model_loaded:
        st.info("Load the trained model file to enable predictions.")
    else:
        st.info(
            "Fill in the session details on the left and click **Predict purchase "
            "intent** to see a result here."
        )

st.divider()

with st.expander("ℹ️ About this model"):
    st.markdown(
        """
- **Task:** Binary classification -- will this browsing session end in a purchase (`Revenue`)?
- **Data:** UCI Online Shoppers Purchasing Intention dataset (session-level behavioural and technical features).
- **Model:** Random Forest classifier, tuned via `RandomizedSearchCV` (3 values per hyperparameter:
  `n_estimators`, `max_depth`), with `class_weight="balanced"` to address class imbalance.
- **Feature engineering:** four extra engagement features are derived automatically from your
  inputs -- total pages viewed, total time on site, average time per product page, and the
  gap between exit and bounce rates.
- **Evaluation focus:** F1-score and ROC-AUC/PR-AUC (accuracy alone is misleading here, since
  most sessions do not end in a purchase).
        """
    )
