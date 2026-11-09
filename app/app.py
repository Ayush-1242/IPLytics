import json
import sys
from pathlib import Path
from datetime import date

import streamlit as st


# ============================================================
# 1. PROJECT IMPORTS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.predict import (
    predict_match,
    get_available_teams,
    get_available_venues,
    features_df,
    clean_venue_name
)


# ============================================================
# 2. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="IPLytics | Cricket Analytics",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# 3. LOAD MODEL INFORMATION
# ============================================================

selection_path = ROOT / "models" / "model_selection.json"

with open(selection_path, "r") as file:
    selection = json.load(file)


# ============================================================
# 4. STADIUM DISPLAY NAME STANDARDIZATION
# ============================================================

def display_venue_name(venue):

    short_names = {

        "Andhra Cricket Association-Visakhapatnam District Cricket Association Stadium, Visakhapatnam":
            "ACA-VDCA Stadium, Visakhapatnam",

        "Bharat Ratna Shri Atal Bihari Vajpayee Ekana Cricket Stadium, Lucknow":
            "Ekana Cricket Stadium, Lucknow",

        "Himachal Pradesh Cricket Association Stadium, Dharamsala":
            "HPCA Stadium, Dharamsala",

        "Maharaja Yadavindra Singh International Cricket Stadium, New Chandigarh":
            "Maharaja Yadavindra Singh Stadium",

        "Punjab Cricket Association IS Bindra Stadium, Mohali, Chandigarh":
            "IS Bindra Stadium, Mohali",

        "Shaheed Veer Narayan Singh International Cricket Stadium, Raipur":
            "Shaheed Veer Narayan Singh Stadium, Raipur",

        "Vidarbha Cricket Association Stadium, Jamtha, Nagpur":
            "Vidarbha Cricket Association Stadium, Nagpur",

        "M Chinnaswamy Stadium, Bengaluru":
            "M Chinnaswamy Stadium",

        "Narendra Modi Stadium, Ahmedabad":
            "Narendra Modi Stadium"
    }

    return short_names.get(venue, venue)


# ============================================================
# 5. PREMIUM DASHBOARD CSS
# ============================================================

st.markdown("""
<style>

@import url(
'https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap'
);

/* MAIN APPLICATION */

.stApp {
    background-color: #0b1120;
    color: #f1f5f9;
    font-family: 'Inter', sans-serif;
}

header[data-testid="stHeader"] {
    background: transparent;
}

.block-container {
    max-width: 1400px;
    padding-top: 2rem;
    padding-bottom: 2rem;
}

/* SIDEBAR WIDTH */

section[data-testid="stSidebar"] {
    min-width: 380px !important;
    max-width: 380px !important;
    background-color: #111b2e;
    border-right: 1px solid #293951;
}

section[data-testid="stSidebar"] > div {
    background-color: #111b2e;
}

/* SIDEBAR TEXT */

section[data-testid="stSidebar"] * {
    color: #e2e8f0;
}

/* BRAND */

.brand {
    font-size: 38px;
    font-weight: 800;
    letter-spacing: -1.5px;
    color: #ff793f;
}

.brand span {
    color: #f8fafc;
}

.tagline {
    color: #94a3b8;
    font-size: 14px;
    margin-top: -8px;
}

/* SECTION TITLES */

.section-title {
    color: #f8fafc;
    font-size: 21px;
    font-weight: 700;
    margin-top: 20px;
    margin-bottom: 15px;
}

/* MODEL STAT CARDS */

.stat-card {
    background: #142138;
    border: 1px solid #293951;
    border-radius: 14px;
    padding: 18px;
    min-height: 105px;
}

.stat-label {
    color: #94a3b8;
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 1px;
}

.stat-value {
    color: #ffffff;
    font-size: 23px;
    font-weight: 700;
    margin-top: 9px;
}

/* MATCH CARDS */

.panel {
    background: linear-gradient(145deg, #17243a, #111b2e);
    border: 1px solid #293951;
    border-radius: 16px;
    padding: 22px;
    margin-bottom: 16px;
}

/* MATCH WINNER CARD */

.winner-card {
    background: linear-gradient(120deg, #202f49, #17243a);
    border: 1px solid #ff793f;
    border-radius: 18px;
    text-align: center;
    padding: 35px 20px;
    margin-top: 18px;
}

.winner-label {
    color: #cbd5e1;
    font-size: 14px;
    text-transform: uppercase;
    letter-spacing: 2px;
}

.winner-name {
    color: #ff793f;
    font-size: 36px;
    font-weight: 800;
    margin: 12px 0;
}

.winner-model {
    color: #cbd5e1;
    font-size: 13px;
}

/* PROBABILITY BARS */

.prob-label {
    display: flex;
    justify-content: space-between;
    color: #e2e8f0;
    font-size: 14px;
    margin-bottom: 10px;
}

.prob-track {
    width: 100%;
    height: 12px;
    background: #293951;
    border-radius: 20px;
    overflow: hidden;
}

.prob-fill-orange {
    height: 100%;
    background: linear-gradient(90deg, #f97316, #ffae42);
    border-radius: 20px;
}

.prob-fill-blue {
    height: 100%;
    background: linear-gradient(90deg, #2586d9, #54b6ff);
    border-radius: 20px;
}

/* BUTTONS */

.stButton > button {
    background: linear-gradient(90deg, #f97316, #ff9347);
    color: white;
    border: none;
    border-radius: 10px;
    min-height: 48px;
    font-weight: 700;
    width: 100%;
}

.stButton > button:hover {
    background: linear-gradient(90deg, #ea580c, #f97316);
    color: white;
    border: none;
}

/* DROPDOWN CONTROL */

.stSelectbox [data-baseweb="select"] > div {
    background-color: #17243a !important;
    border: 1px solid #34445c !important;
    min-height: 48px;
}

.stSelectbox [data-baseweb="select"] {
    font-size: 17px !important;
}

.stSelectbox [data-baseweb="select"] * {
    color: #f8fafc !important;
    -webkit-text-fill-color: #f8fafc !important;
    font-size: 17px !important;
}

/* DROPDOWN ARROW */

.stSelectbox [data-baseweb="select"] svg {
    fill: #f8fafc !important;
}

/* DROPDOWN POPUP */

[data-baseweb="popover"] {
    background-color: #17243a !important;
}

[data-baseweb="popover"] [role="listbox"] {
    background-color: #17243a !important;
}

[data-baseweb="popover"] [role="option"] {
    background-color: #17243a !important;
    color: #f8fafc !important;
    font-size: 17px !important;
    line-height: 1.6 !important;
    min-height: 45px !important;
    padding: 12px 14px !important;
}

[data-baseweb="popover"] [role="option"]:hover {
    background-color: #293951 !important;
}

/* DATE PICKER */

.stDateInput input {
    background-color: #17243a !important;
    color: #f8fafc !important;
    -webkit-text-fill-color: #f8fafc !important;
    border: 1px solid #34445c !important;
    min-height: 44px;
    font-size: 16px !important;
}

.stDateInput button {
    color: #f8fafc !important;
}

/* INPUT LABELS */

.stSelectbox label,
.stDateInput label {
    color: #cbd5e1 !important;
    font-size: 15px !important;
    font-weight: 600 !important;
}

/* DIVIDERS */

hr {
    border-color: #293951;
}

/* FOOTER */

.footer {
    text-align: center;
    color: #64748b;
    font-size: 12px;
    padding: 25px 0;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# 6. LOAD AVAILABLE TEAMS AND VENUES
# ============================================================

teams = get_available_teams()
venues = get_available_venues()


# ============================================================
# 7. SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        '<div class="brand">🏏 IPLytics</div>',
        unsafe_allow_html=True
    )

    st.caption("MACHINE LEARNING • CRICKET ANALYTICS")

    st.divider()

    st.markdown("### 🏆 Match Configuration")

    # TEAM 1

    default_team1 = (
        teams.index("Chennai Super Kings")
        if "Chennai Super Kings" in teams else 0
    )

    team1 = st.selectbox(
        "Team 1",
        teams,
        index=default_team1,
        key="team1"
    )

    # TEAM 2

    team2_options = [
        team for team in teams if team != team1
    ]

    default_team2 = (
        team2_options.index("Mumbai Indians")
        if "Mumbai Indians" in team2_options else 0
    )

    team2 = st.selectbox(
        "Team 2",
        team2_options,
        index=default_team2,
        key="team2"
    )

    # VENUE
    # format_func changes only the displayed label.
    # The original venue value is preserved.

    venue = st.selectbox(
        "Match Venue",
        venues,
        format_func=display_venue_name,
        key="venue"
    )

    # MATCH DATE

    match_date = st.date_input(
        "Match Date",
        value=date.today(),
        min_value=date(2008, 1, 1),
        key="match_date"
    )

    st.divider()

    # PREDICTION BUTTON

    predict_button = st.button(
        "⚡ PREDICT MATCH WINNER",
        use_container_width=True
    )

    st.write("")

    st.caption("Prediction timing: Before toss")

    st.caption("Data available through: 31 May 2026")


# ============================================================
# 8. MAIN HEADER
# ============================================================

header_col1, header_col2 = st.columns([3, 1])

with header_col1:

    st.markdown(
        '<div class="brand">IPLytics <span>Analytics</span></div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="tagline">'
        'Intelligent IPL match predictions powered by machine learning'
        '</div>',
        unsafe_allow_html=True
    )

with header_col2:

    st.markdown(
        """
        <div style="
            text-align:right;
            padding-top:12px;
            color:#94a3b8;
            font-size:13px;
        ">
            <span style="color:#22c55e;">●</span>
            PREDICTION ENGINE READY
        </div>
        """,
        unsafe_allow_html=True
    )

st.divider()


# ============================================================
# 9. MODEL OVERVIEW
# ============================================================

st.markdown(
    '<div class="section-title">📊 Model Overview</div>',
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3)

with col1:

    st.markdown(
        f"""
        <div class="stat-card">
            <div class="stat-label">Selected Algorithm</div>
            <div class="stat-value">
                {selection['selected_model']}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col2:

    st.markdown(
        f"""
        <div class="stat-card">
            <div class="stat-label">Training Matches</div>
            <div class="stat-value">
                {selection['training_matches']:,}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col3:

    st.markdown(
        f"""
        <div class="stat-card">
            <div class="stat-label">Historical Data Through</div>
            <div class="stat-value">
                {selection['training_data_end']}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# 10. MATCH PREVIEW
# ============================================================

st.markdown(
    '<div class="section-title">🏏 Match Preview</div>',
    unsafe_allow_html=True
)

team_col1, vs_col, team_col2 = st.columns([5, 1, 5])

with team_col1:
    with st.container(border=True):
        st.caption("TEAM 1")
        st.markdown(
            f"<h2 style='color:#ff9347;text-align:center;'>{team1}</h2>",
            unsafe_allow_html=True
        )

with vs_col:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("## VS")

with team_col2:
    with st.container(border=True):
        st.caption("TEAM 2")
        st.markdown(
            f"<h2 style='color:#54b6ff;text-align:center;'>{team2}</h2>",
            unsafe_allow_html=True
        )
# ============================================================
# 11. VENUE CITY DETECTION
# ============================================================

venue_rows = features_df[
    features_df["venue"].apply(clean_venue_name) == venue
]

if not venue_rows.empty:

    city_values = venue_rows["city"].dropna()

    if not city_values.empty:
        city = city_values.mode().iloc[0]
    else:
        city = "Unknown"

else:
    city = "Unknown"


info1, info2 = st.columns(2)

with info1:

    st.caption(f"📍 {display_venue_name(venue)}")

with info2:

    st.caption(
        f"📅 {match_date.strftime('%d %B %Y')}"
    )


# ============================================================
# 12. PREDICTION STATE
# ============================================================

if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None

current_signature = (
    team1,
    team2,
    venue,
    match_date
)

if (
    st.session_state.get("prediction_signature")
    != current_signature
):
    st.session_state.prediction_result = None


# ============================================================
# 13. GENERATE PREDICTION
# ============================================================

if predict_button:

    if team1 == team2:

        st.error("Please select two different teams.")

    else:

        try:

            with st.spinner(
                "Analyzing historical IPL performance..."
            ):

                result = predict_match(
                    team1=team1,
                    team2=team2,
                    venue=venue,
                    city=city,
                    prediction_date=match_date
                )

            st.session_state.prediction_result = result

            st.session_state.prediction_signature = (
                current_signature
            )

        except Exception as error:

            st.session_state.prediction_result = None

            st.error(
                f"Prediction failed: {error}"
            )


# ============================================================
# 14. DISPLAY PREDICTION RESULTS
# ============================================================

result = st.session_state.prediction_result

if result:

    st.divider()

    st.markdown("## 🎯 Match Prediction")

    # WINNER CARD

    with st.container(border=True):

        st.markdown(
            "<div style='text-align:center;color:#94a3b8;'>"
            "PREDICTED WINNER"
            "</div>",
            unsafe_allow_html=True
        )

        st.markdown("")

        st.markdown(
            f"<h1 style='text-align:center;color:#ff793f;'>"
            f"{result['predicted_winner']}"
            f"</h1>",
            unsafe_allow_html=True
        )

        st.caption(
            f"Selected Algorithm: {result['selected_model']}"
        )

    # WIN PROBABILITIES

    st.markdown("## 📊 Estimated Win Probabilities")

    p1 = float(result["team1_probability"])
    p2 = float(result["team2_probability"])

    prob1, prob2 = st.columns(2)

    with prob1:

        st.markdown(f"### 🟠 {team1}")

        st.metric(
            label="Estimated Win Probability",
            value=f"{p1:.2f}%"
        )

        st.progress(
            min(max(p1 / 100, 0.0), 1.0)
        )

    with prob2:

        st.markdown(f"### 🔵 {team2}")

        st.metric(
            label="Estimated Win Probability",
            value=f"{p2:.2f}%"
        )

        st.progress(
            min(max(p2 / 100, 0.0), 1.0)
        )

    st.success("Prediction generated successfully!")

    st.caption(
        f"Prediction date: {result['prediction_date']}"
    )

    st.info(
        "These are experimental model estimates, not guaranteed "
        "match outcomes."
    )


# ============================================================
# 15. INITIAL DASHBOARD MESSAGE
# ============================================================

else:

    with st.container(border=True):
        st.markdown(
            "<h1 style='text-align:center;'>🏏</h1>",
            unsafe_allow_html=True
        )

        st.markdown(
            "<h3 style='text-align:center;color:#f8fafc;'>"
            "Ready for the next match?"
            "</h3>",
            unsafe_allow_html=True
        )

        st.markdown(
            "<p style='text-align:center;color:#94a3b8;'>"
            "Select the teams, venue and date from the sidebar, "
            "then click Predict Match Winner."
            "</p>",
            unsafe_allow_html=True
        )


# ============================================================
# 16. ABOUT IPLYTICS
# ============================================================

st.divider()

st.markdown("## ℹ️ About IPLytics")

with st.container(border=True):

    st.markdown(
        "IPLytics uses historical IPL match data and machine "
        "learning to estimate match outcomes before the toss."
    )

    st.markdown("")

    st.caption(
        "Predictions are experimental estimates, not guaranteed "
        "results. The model does not use confirmed playing XIs, "
        "live injuries, weather updates or toss information. "
        "Squad statistics are estimates based on historical data."
    )


# ============================================================
# 17. FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        IPLytics • IPL Machine Learning Project
        • Built with Python & Streamlit
    </div>
    """,
    unsafe_allow_html=True
)
