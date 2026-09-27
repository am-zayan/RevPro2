import base64
import html
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


# ============================================================
# REVPRO — E-COMMERCE PURCHASE PREDICTION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_CANDIDATES = [
    BASE_DIR / "shopper_revenue_model.pkl",
    BASE_DIR / "shopper_revenue_model(1).pkl",
]
LOGO_CANDIDATES = [
    BASE_DIR / "87019-Photoroom.svg",
    BASE_DIR / "revpro_logo.svg",
    BASE_DIR / "logo.svg",
]

st.set_page_config(
    page_title="RevPro — Purchase Prediction",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# STYLING
# ============================================================

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Inter:wght@400;500;600;700;800&display=swap');

:root {
    --black: #000000;
    --near-black: #080808;
    --panel: #111111;
    --panel-2: #181818;
    --line: #2b2b2b;
    --muted: #8b8b8b;
    --soft: #cfcfcf;
    --white: #f7f7f7;
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at 78% 4%, rgba(255,255,255,0.045), transparent 24%),
        linear-gradient(rgba(255,255,255,0.018) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255,255,255,0.018) 1px, transparent 1px),
        #050505;
    background-size: auto, 32px 32px, 32px 32px;
    color: var(--white);
}

.block-container {
    max-width: 1240px;
    padding-top: 1.4rem;
    padding-bottom: 4rem;
}

header[data-testid="stHeader"] {
    background: transparent;
}

[data-testid="stToolbar"] {
    display: none;
}

footer {
    visibility: hidden;
}

.revpro-nav {
    display: flex;
    align-items: center;
    justify-content: space-between;
    border-bottom: 1px solid var(--line);
    padding: 10px 0 18px 0;
    margin-bottom: 40px;
}

.brand {
    font-family: 'DM Mono', monospace;
    font-size: 17px;
    font-weight: 500;
    letter-spacing: 0.16em;
}

.brand small {
    color: #777;
    font-size: 10px;
    letter-spacing: 0.08em;
    margin-left: 9px;
}

.eyebrow {
    font-family: 'DM Mono', monospace;
    color: #8d8d8d;
    font-size: 11px;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    margin-bottom: 12px;
}

.hero-title {
    font-size: clamp(42px, 6vw, 78px);
    line-height: 0.96;
    letter-spacing: -0.055em;
    font-weight: 800;
    max-width: 850px;
    margin: 0;
}

.hero-title span {
    color: #7c7c7c;
}

.hero-copy {
    color: #a4a4a4;
    font-size: 17px;
    line-height: 1.7;
    max-width: 670px;
    margin-top: 24px;
}

.pixel-line {
    height: 5px;
    width: 76px;
    background: #f2f2f2;
    margin: 28px 0;
    box-shadow: 10px 0 #777, 20px 0 #444;
}

.section-title {
    font-size: 27px;
    font-weight: 700;
    letter-spacing: -0.025em;
    margin-bottom: 8px;
}

.section-copy {
    color: #9b9b9b;
    line-height: 1.65;
}

.metric-card, .info-card, .result-card, .input-card {
    background: linear-gradient(145deg, #111111, #0b0b0b);
    border: 1px solid #292929;
    padding: 22px;
    border-radius: 3px;
}

.metric-card {
    min-height: 132px;
}

.metric-label {
    color: #777;
    font-family: 'DM Mono', monospace;
    font-size: 10px;
    letter-spacing: .12em;
    text-transform: uppercase;
}

.metric-value {
    font-size: 30px;
    font-weight: 700;
    margin-top: 14px;
}

.metric-note {
    color: #777;
    font-size: 12px;
    margin-top: 5px;
}

.info-card {
    min-height: 175px;
}

.info-number {
    font-family: 'DM Mono', monospace;
    color: #666;
    font-size: 12px;
}

.info-title {
    font-size: 18px;
    font-weight: 700;
    margin: 14px 0 7px;
}

.info-text {
    color: #929292;
    line-height: 1.55;
    font-size: 14px;
}

.form-section {
    border-top: 1px solid #272727;
    padding-top: 28px;
    margin-top: 26px;
}

.form-title {
    font-size: 19px;
    font-weight: 700;
    margin-bottom: 5px;
}

.form-subtitle {
    color: #777;
    font-size: 13px;
    margin-bottom: 18px;
}

div[data-testid="stTextInput"] label,
div[data-testid="stNumberInput"] label,
div[data-testid="stSelectbox"] label,
div[data-testid="stSlider"] label {
    color: #d0d0d0 !important;
    font-size: 13px !important;
    font-weight: 600 !important;
}

div[data-testid="stTextInput"] small,
div[data-testid="stNumberInput"] small,
div[data-testid="stSelectbox"] small {
    color: #777 !important;
}

div[data-baseweb="select"] > div,
input {
    background-color: #0d0d0d !important;
    border-color: #303030 !important;
    color: #f4f4f4 !important;
}

div[data-testid="stSlider"] div[role="slider"] {
    background: #f2f2f2;
}

.stButton > button {
    background: #f2f2f2;
    color: #050505;
    border: 1px solid #f2f2f2;
    border-radius: 2px;
    font-family: 'DM Mono', monospace;
    font-weight: 600;
    letter-spacing: .05em;
    min-height: 46px;
    transition: all .16s ease;
}

.stButton > button:hover {
    background: #d4d4d4;
    border-color: #d4d4d4;
    transform: translateY(-1px);
}

.secondary-button > button {
    background: transparent !important;
    color: #ddd !important;
    border-color: #383838 !important;
}

.result-card {
    text-align: center;
    padding: 34px 26px;
    margin-top: 20px;
}

.result-kicker {
    font-family: 'DM Mono', monospace;
    color: #777;
    font-size: 11px;
    letter-spacing: .14em;
}

.result-main {
    font-size: clamp(30px, 4vw, 50px);
    line-height: 1;
    font-weight: 800;
    margin: 17px 0 10px;
}

.result-prob {
    font-family: 'DM Mono', monospace;
    font-size: 38px;
    font-weight: 500;
}

.result-caption {
    color: #777;
    font-size: 11px;
    letter-spacing: .1em;
    text-transform: uppercase;
    margin-top: 5px;
}

.status {
    margin-top: 20px;
    color: #8c8c8c;
    font-family: 'DM Mono', monospace;
    font-size: 10px;
    letter-spacing: .1em;
}

.disclaimer {
    border-left: 2px solid #555;
    padding: 12px 15px;
    color: #858585;
    font-size: 12px;
    line-height: 1.6;
    background: #0b0b0b;
}

.footer {
    border-top: 1px solid #222;
    margin-top: 70px;
    padding-top: 18px;
    color: #555;
    font-family: 'DM Mono', monospace;
    font-size: 10px;
    letter-spacing: .08em;
}

div[data-testid="stAlert"] {
    border-radius: 2px;
}
</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# HELPERS
# ============================================================

@st.cache_resource
def load_model():
    for path in MODEL_CANDIDATES:
        if path.exists():
            return joblib.load(path)
    raise FileNotFoundError(
        "No trained model found. Place shopper_revenue_model.pkl beside app.py."
    )


def logo_data_uri():
    for path in LOGO_CANDIDATES:
        if path.exists():
            raw = path.read_bytes()
            encoded = base64.b64encode(raw).decode("ascii")
            if path.suffix.lower() == ".svg":
                return f"data:image/svg+xml;base64,{encoded}"
            return f"data:image/png;base64,{encoded}"
    return None


def go(page):
    st.session_state["page"] = page
    st.rerun()


def friendly_number(value, suffix=""):
    if isinstance(value, float):
        return f"{value:.2f}{suffix}"
    return f"{value}{suffix}"


def render_nav():
    st.markdown(
        """
        <div class="revpro-nav">
            <div class="brand">REVPRO <small>SESSION INTELLIGENCE</small></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    nav = st.columns([1, 1, 1, 6])
    with nav[0]:
        if st.button("HOME", use_container_width=True):
            go("Home")
    with nav[1]:
        if st.button("ABOUT", use_container_width=True):
            go("About")
    with nav[2]:
        if st.button("PREDICT", use_container_width=True):
            go("Predict")


def section_header(kicker, title, copy):
    st.markdown(f'<div class="eyebrow">{html.escape(kicker)}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="section-title">{html.escape(title)}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="section-copy">{html.escape(copy)}</div>', unsafe_allow_html=True)


# ============================================================
# NAVIGATION STATE
# ============================================================

if "page" not in st.session_state:
    st.session_state["page"] = "Home"

render_nav()


# ============================================================
# HOME
# ============================================================

if st.session_state["page"] == "Home":
    logo = logo_data_uri()
    if logo:
        st.markdown(
            f'<img src="{logo}" style="width:72px;height:72px;object-fit:contain;margin-bottom:20px;">',
            unsafe_allow_html=True,
        )

    st.markdown('<div class="eyebrow">E-COMMERCE / MACHINE LEARNING</div>', unsafe_allow_html=True)
    st.markdown(
        '<h1 class="hero-title">Predict purchase intent<br><span>before the transaction.</span></h1>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="hero-copy">RevPro analyzes an online shopping session and uses a trained machine-learning model to estimate whether the visitor is likely to complete a purchase.</div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="pixel-line"></div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns([1.4, 1.4, 1.4])
    with c1:
        if st.button("START PREDICTION  →", use_container_width=True):
            go("Predict")
    with c2:
        st.markdown(
            '<div class="metric-card"><div class="metric-label">TASK</div><div class="metric-value">Binary</div><div class="metric-note">Purchase / no purchase</div></div>',
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            '<div class="metric-card"><div class="metric-label">MODEL</div><div class="metric-value">Decision Tree</div><div class="metric-note">Depth limited to 6</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    section_header(
        "THE IDEA",
        "From visitor behaviour to a prediction.",
        "The model receives session-level browsing information, applies the preprocessing learned during training, and returns a binary purchase prediction.",
    )

    st.markdown("<br>", unsafe_allow_html=True)
    cols = st.columns(3)
    cards = [
        ("01", "Visitor data", "Browsing activity, engagement signals, visitor type, traffic context and session details."),
        ("02", "ML pipeline", "The supplied pipeline handles categorical encoding and numerical scaling before classification."),
        ("03", "Purchase signal", "RevPro presents the model's predicted class and purchase probability when available."),
    ]
    for col, (num, title, text) in zip(cols, cards):
        with col:
            st.markdown(
                f'<div class="info-card"><div class="info-number">{num}</div><div class="info-title">{title}</div><div class="info-text">{text}</div></div>',
                unsafe_allow_html=True,
            )


# ============================================================
# ABOUT
# ============================================================

elif st.session_state["page"] == "About":
    section_header(
        "ABOUT REVPRO",
        "A readable interface for a real ML pipeline.",
        "RevPro is a Streamlit interface built around the supplied online-shopping purchase prediction model.",
    )

    st.markdown("<br>", unsafe_allow_html=True)
    a, b = st.columns(2)
    with a:
        st.markdown(
            """
            <div class="info-card">
                <div class="info-number">WHAT IT DOES</div>
                <div class="info-title">Purchase-intent prediction</div>
                <div class="info-text">
                A visitor enters information describing their current shopping session.
                RevPro sends those values through the trained preprocessing pipeline and
                Decision Tree classifier to produce a purchase/no-purchase prediction.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with b:
        st.markdown(
            """
            <div class="info-card">
                <div class="info-number">WHY THE UI IS DIFFERENT</div>
                <div class="info-title">Human labels, machine columns</div>
                <div class="info-text">
                The model keeps its original feature names and column order internally.
                The interface translates those technical names into plain-English labels
                so a normal user can understand what information they are entering.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br><br>", unsafe_allow_html=True)
    section_header(
        "MODEL PIPELINE",
        "What happens behind the screen?",
        "The saved model is a complete scikit-learn pipeline. RevPro does not retrain it during prediction.",
    )

    steps = st.columns(5)
    for col, num, title in zip(
        steps,
        ["01", "02", "03", "04", "05"],
        ["Session input", "Validation", "Encoding", "Scaling", "Decision"],
    ):
        with col:
            st.markdown(
                f'<div class="metric-card"><div class="metric-label">{num}</div><div class="metric-value" style="font-size:18px">{title}</div></div>',
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        '<div class="disclaimer"><strong>Interpretation note:</strong> A model probability is a model output, not a guarantee that a visitor will purchase. Real behaviour can be affected by factors that are not present in the session data.</div>',
        unsafe_allow_html=True,
    )


# ============================================================
# PREDICTION
# ============================================================

elif st.session_state["page"] == "Predict":
    section_header(
        "REVPRO / PREDICT",
        "Tell us about the shopping session.",
        "Use the visitor's current session behaviour. The labels below are written for people; the original model feature names remain unchanged behind the scenes.",
    )

    try:
        model = load_model()
    except Exception as exc:
        st.error("The trained model could not be loaded. Make sure the .pkl file is beside app.py.")
        st.stop()

    # --------------------------------------------------------
    # Friendly feature dictionary
    # --------------------------------------------------------
    # Keys MUST remain the exact model feature names.
    # Values are the user-facing names/help.
    feature_help = {
        "Administrative": ("Administrative pages visited", "Number of admin/account-related pages viewed."),
        "Administrative_Duration": ("Time spent on admin pages", "Total time spent on admin/account-related pages, in seconds."),
        "Informational": ("Informational pages visited", "Number of information/help-oriented pages viewed."),
        "Informational_Duration": ("Time spent on informational pages", "Total time spent on informational pages, in seconds."),
        "ProductRelated": ("Product pages visited", "Number of product-related pages viewed."),
        "ProductRelated_Duration": ("Time spent on product pages", "Total time spent on product-related pages, in seconds."),
        "BounceRates": ("Bounce rate", "Average bounce rate for the session/page interaction."),
        "ExitRates": ("Exit rate", "Average exit rate for the session/page interaction."),
        "PageValues": ("Page value", "Page-value signal associated with the session."),
        "SpecialDay": ("Special-event proximity", "Closeness to a special day such as a promotional/holiday event."),
        "Month": ("Shopping month", "Month in which the session occurred."),
        "OperatingSystems": ("Operating system", "Operating-system category used by the visitor."),
        "Browser": ("Browser", "Browser category used by the visitor."),
        "Region": ("Visitor region", "Region category associated with the visitor."),
        "TrafficType": ("Traffic source type", "Traffic-source category assigned to the session."),
        "VisitorType": ("Visitor type", "Whether the visitor is new, returning, or another recorded type."),
        "Weekend": ("Weekend visit", "Whether the session occurred on a weekend."),
    }

    # --------------------------------------------------------
    # Browsing behaviour
    # --------------------------------------------------------
    st.markdown('<div class="form-section">', unsafe_allow_html=True)
    st.markdown('<div class="form-title">Browsing activity</div>', unsafe_allow_html=True)
    st.markdown('<div class="form-subtitle">What did the visitor browse, and how much time did they spend there?</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        administrative = st.number_input(
            feature_help["Administrative"][0],
            min_value=0,
            value=0,
            step=1,
            help=feature_help["Administrative"][1],
        )
        informational = st.number_input(
            feature_help["Informational"][0],
            min_value=0,
            value=0,
            step=1,
            help=feature_help["Informational"][1],
        )
    with c2:
        administrative_duration = st.number_input(
            feature_help["Administrative_Duration"][0],
            min_value=0.0,
            value=0.0,
            step=10.0,
            help=feature_help["Administrative_Duration"][1],
        )
        informational_duration = st.number_input(
            feature_help["Informational_Duration"][0],
            min_value=0.0,
            value=0.0,
            step=10.0,
            help=feature_help["Informational_Duration"][1],
        )
    with c3:
        product_related = st.number_input(
            feature_help["ProductRelated"][0],
            min_value=0,
            value=1,
            step=1,
            help=feature_help["ProductRelated"][1],
        )
        product_related_duration = st.number_input(
            feature_help["ProductRelated_Duration"][0],
            min_value=0.0,
            value=0.0,
            step=10.0,
            help=feature_help["ProductRelated_Duration"][1],
        )

    # --------------------------------------------------------
    # Engagement
    # --------------------------------------------------------
    st.markdown('<div class="form-section">', unsafe_allow_html=True)
    st.markdown('<div class="form-title">Engagement signals</div>', unsafe_allow_html=True)
    st.markdown('<div class="form-subtitle">Session-level indicators describing how the visitor interacted with the site.</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        bounce_rates = st.number_input(
            feature_help["BounceRates"][0],
            min_value=0.0,
            max_value=1.0,
            value=0.02,
            step=0.01,
            format="%.4f",
            help=feature_help["BounceRates"][1] + " Enter as a decimal from 0 to 1.",
        )
        exit_rates = st.number_input(
            feature_help["ExitRates"][0],
            min_value=0.0,
            max_value=1.0,
            value=0.03,
            step=0.01,
            format="%.4f",
            help=feature_help["ExitRates"][1] + " Enter as a decimal from 0 to 1.",
        )
    with c2:
        page_values = st.number_input(
            feature_help["PageValues"][0],
            min_value=0.0,
            value=0.0,
            step=1.0,
            help=feature_help["PageValues"][1],
        )
        special_day = st.slider(
            feature_help["SpecialDay"][0],
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=0.1,
            help=feature_help["SpecialDay"][1] + " Use the dataset's 0–1 scale.",
        )
    with c3:
        month = st.selectbox(
            feature_help["Month"][0],
            ["Feb", "Mar", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
            help=feature_help["Month"][1],
        )
        weekend = st.selectbox(
            feature_help["Weekend"][0],
            [False, True],
            format_func=lambda x: "Yes" if x else "No",
            help=feature_help["Weekend"][1],
        )

    # --------------------------------------------------------
    # Visitor / session context
    # --------------------------------------------------------
    st.markdown('<div class="form-section">', unsafe_allow_html=True)
    st.markdown('<div class="form-title">Visitor & session context</div>', unsafe_allow_html=True)
    st.markdown('<div class="form-subtitle">Technical and traffic information captured for the session.</div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        operating_system = st.number_input(
            feature_help["OperatingSystems"][0],
            min_value=1,
            value=2,
            step=1,
            help=feature_help["OperatingSystems"][1],
        )
    with c2:
        browser = st.number_input(
            feature_help["Browser"][0],
            min_value=1,
            value=2,
            step=1,
            help=feature_help["Browser"][1],
        )
    with c3:
        region = st.number_input(
            feature_help["Region"][0],
            min_value=1,
            value=1,
            step=1,
            help=feature_help["Region"][1],
        )
    with c4:
        traffic_type = st.number_input(
            feature_help["TrafficType"][0],
            min_value=1,
            value=2,
            step=1,
            help=feature_help["TrafficType"][1],
        )

    c1, c2 = st.columns(2)
    with c1:
        visitor_type = st.selectbox(
            feature_help["VisitorType"][0],
            ["Returning_Visitor", "New_Visitor", "Other"],
            format_func=lambda x: x.replace("_", " "),
            help=feature_help["VisitorType"][1],
        )
    with c2:
        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        st.caption("Weekend status was selected above under Engagement signals.")

    st.markdown("<br>", unsafe_allow_html=True)
    analyze = st.button("ANALYZE SESSION  →", use_container_width=True)

    if analyze:
        input_data = pd.DataFrame(
            [{
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
                "OperatingSystems": operating_system,
                "Browser": browser,
                "Region": region,
                "TrafficType": traffic_type,
                "VisitorType": visitor_type,
                "Weekend": weekend,
            }]
        )

        # Verify the UI-to-model contract before prediction.
        expected = list(getattr(model, "feature_names_in_", []))
        if expected and list(input_data.columns) != expected:
            st.error("The prediction input does not match the trained model's expected feature order.")
            st.stop()

        try:
            prediction = model.predict(input_data)[0]
            probabilities = model.predict_proba(input_data)[0] if hasattr(model, "predict_proba") else None

            if probabilities is not None and hasattr(model, "classes_"):
                class_to_probability = dict(zip(model.classes_, probabilities))
                purchase_probability = float(class_to_probability.get(1, 0.0))
            else:
                purchase_probability = None

            if int(prediction) == 1:
                headline = "PURCHASE LIKELY"
                supporting = "The model classified this session as a revenue-generating visit."
            else:
                headline = "PURCHASE UNLIKELY"
                supporting = "The model classified this session as a non-revenue visit."

            st.markdown(
                f"""
                <div class="result-card">
                    <div class="result-kicker">REVPRO ANALYSIS / COMPLETE</div>
                    <div class="result-main">{headline}</div>
                    <div style="color:#929292;font-size:14px;max-width:620px;margin:auto;">{supporting}</div>
                    {
                        f'<div class="result-prob" style="margin-top:24px;">{purchase_probability * 100:.2f}%</div><div class="result-caption">Purchase probability</div>'
                        if purchase_probability is not None
                        else ''
                    }
                    <div class="status">MODEL OUTPUT: {int(prediction)} &nbsp;•&nbsp; PIPELINE: READY</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="form-title">Session summary</div>', unsafe_allow_html=True)
            st.markdown('<div class="form-subtitle">The model received the following visitor/session information.</div>', unsafe_allow_html=True)

            summary = [
                ("Pages", f"{administrative + informational + product_related} tracked pages"),
                ("Product browsing", f"{product_related} product pages"),
                ("Page value", f"{page_values:.2f}"),
                ("Visitor", visitor_type.replace("_", " ")),
                ("Month", month),
                ("Weekend", "Yes" if weekend else "No"),
            ]
            cols = st.columns(3)
            for col, (label, value) in zip(cols * 2, summary):
                with col:
                    st.markdown(
                        f'<div class="metric-card"><div class="metric-label">{html.escape(label)}</div><div class="metric-value" style="font-size:19px">{html.escape(str(value))}</div></div>',
                        unsafe_allow_html=True,
                    )

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(
                '<div class="disclaimer"><strong>Important:</strong> This result is a machine-learning prediction based only on the supplied session features. It should not be interpreted as a guarantee of what the visitor will do.</div>',
                unsafe_allow_html=True,
            )

        except Exception:
            st.error(
                "RevPro could not process this session. Check the entered values and make sure the supplied model file matches the project version."
            )

st.markdown(
    '<div class="footer">REVPRO • E-COMMERCE PURCHASE PREDICTION • STREAMLIT ML APPLICATION</div>',
    unsafe_allow_html=True,
)
