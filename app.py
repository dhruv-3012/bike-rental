import streamlit as st
import pickle
import numpy as np

# ---- Page Config ----
st.set_page_config(
    page_title="Bikecast",
    page_icon="🚲",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---- Custom CSS ----
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

    /* Global */
    html, body, [class*="css"] {
        font-family: 'Space Grotesk', sans-serif;
        background-color: #0a0e1a;
        color: #e2e8f0;
    }

    .stApp {
        background: linear-gradient(135deg, #0a0e1a 0%, #0d1526 50%, #0a1020 100%);
    }

    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1526 0%, #0a1020 100%);
        border-right: 1px solid rgba(20, 184, 166, 0.2);
    }
    [data-testid="stSidebar"] .stRadio label {
        color: #94a3b8 !important;
        font-size: 0.95rem;
        padding: 8px 0;
        cursor: pointer;
        transition: color 0.2s;
    }
    [data-testid="stSidebar"] .stRadio label:hover {
        color: #14b8a6 !important;
    }

    /* Stat Cards */
    .stat-card {
        background: linear-gradient(135deg, #111827 0%, #1a2236 100%);
        border: 1px solid rgba(20, 184, 166, 0.15);
        border-radius: 16px;
        padding: 24px;
        position: relative;
        overflow: hidden;
    }
    .stat-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 2px;
        background: linear-gradient(90deg, #14b8a6, #06b6d4);
    }
    .stat-label {
        font-size: 0.75rem;
        letter-spacing: 0.12em;
        color: #64748b;
        text-transform: uppercase;
        margin-bottom: 8px;
    }
    .stat-value {
        font-size: 2rem;
        font-weight: 700;
        color: #f1f5f9;
        line-height: 1.1;
        font-family: 'JetBrains Mono', monospace;
    }
    .stat-sub {
        font-size: 0.8rem;
        color: #475569;
        margin-top: 6px;
    }
    .stat-icon {
        font-size: 1.5rem;
        margin-bottom: 12px;
    }

    /* Section Headers */
    .section-title {
        font-size: 1.4rem;
        font-weight: 700;
        color: #f1f5f9;
        margin-bottom: 4px;
    }
    .section-sub {
        font-size: 0.85rem;
        color: #64748b;
        margin-bottom: 24px;
    }

    /* Logo */
    .logo-area {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 20px 0 30px 0;
    }
    .logo-text {
        font-size: 1.4rem;
        font-weight: 700;
        color: #f1f5f9;
    }
    .logo-badge {
        background: linear-gradient(90deg, #14b8a6, #06b6d4);
        color: #0a0e1a;
        font-size: 0.65rem;
        font-weight: 700;
        padding: 2px 7px;
        border-radius: 6px;
        letter-spacing: 0.05em;
    }

    /* Weather pill */
    .weather-pill {
        background: linear-gradient(90deg, #fef3c7, #fde68a);
        color: #92400e;
        border-radius: 30px;
        padding: 8px 20px;
        font-weight: 600;
        font-size: 0.9rem;
        display: inline-block;
    }

    /* Input styling */
    .stSelectbox > div > div, .stNumberInput > div > div > input, .stSlider {
        background-color: #111827 !important;
        border-color: rgba(20,184,166,0.2) !important;
        color: #e2e8f0 !important;
        border-radius: 10px !important;
    }

    /* Predict button */
    .stButton > button {
        background: linear-gradient(90deg, #14b8a6, #06b6d4);
        color: #0a0e1a;
        border: none;
        border-radius: 12px;
        font-weight: 700;
        font-size: 1rem;
        padding: 14px 40px;
        width: 100%;
        letter-spacing: 0.03em;
        transition: opacity 0.2s, transform 0.1s;
    }
    .stButton > button:hover {
        opacity: 0.9;
        transform: translateY(-1px);
    }

    /* Result box */
    .result-box {
        background: linear-gradient(135deg, #0d2818 0%, #0a1f12 100%);
        border: 1px solid rgba(20,184,166,0.3);
        border-radius: 16px;
        padding: 28px;
        text-align: center;
        margin-top: 20px;
    }
    .result-number {
        font-size: 3.5rem;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
        color: #14b8a6;
        line-height: 1;
    }
    .result-label {
        color: #64748b;
        margin-top: 8px;
        font-size: 0.9rem;
    }

    /* Divider */
    .custom-divider {
        border: none;
        border-top: 1px solid rgba(20,184,166,0.1);
        margin: 20px 0;
    }

    /* Input labels */
    .stSelectbox label, .stSlider label, .stNumberInput label {
        color: #94a3b8 !important;
        font-size: 0.85rem !important;
        font-weight: 500 !important;
        letter-spacing: 0.03em;
    }

    /* Nav item styling in sidebar */
    .nav-item {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 10px 14px;
        border-radius: 10px;
        margin-bottom: 4px;
        cursor: pointer;
        transition: background 0.2s;
        color: #64748b;
        font-size: 0.9rem;
    }
    .nav-item.active {
        background: rgba(20,184,166,0.1);
        color: #14b8a6;
        font-weight: 600;
    }
    .nav-item:hover {
        background: rgba(20,184,166,0.06);
        color: #94a3b8;
    }
</style>
""", unsafe_allow_html=True)

# ---- Sidebar ----
with st.sidebar:
    st.markdown("""
    <div class="logo-area">
        <span style="font-size:1.6rem">🚲</span>
        <span class="logo-text">Bikecast</span>
        <span class="logo-badge">AI</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    page = st.radio(
        "",
        ["📊  Dashboard", "🌤  Weather Forecast", "🔮  Predict Demand", "📈  Analytics"],
        label_visibility="collapsed"
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="position:absolute; bottom:30px; left:20px; right:20px;">
        <div style="background:rgba(20,184,166,0.08); border:1px solid rgba(20,184,166,0.15); border-radius:12px; padding:14px;">
            <div style="font-size:0.75rem; color:#64748b; margin-bottom:4px;">MODEL STATUS</div>
            <div style="display:flex; align-items:center; gap:8px;">
                <div style="width:8px;height:8px;border-radius:50%;background:#22c55e;box-shadow:0 0 6px #22c55e;"></div>
                <span style="color:#94a3b8; font-size:0.85rem;">Active · Ready</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ---- Dashboard Page ----
if "Dashboard" in page:
    col_title, col_weather = st.columns([3, 1])
    with col_title:
        st.markdown('<div class="section-title">Bike Rental Dashboard</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">Historical analysis · 365 days · 8,760 hourly records</div>', unsafe_allow_html=True)
    with col_weather:
        st.markdown('<div style="text-align:right; padding-top:10px;"><span class="weather-pill">☀️ Clear 22°C</span></div>', unsafe_allow_html=True)

    # Stat Cards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("""
        <div class="stat-card">
            <div class="stat-icon">🚲</div>
            <div class="stat-label">Total Rentals</div>
            <div class="stat-value">1,058,318</div>
            <div class="stat-sub">Past 12 months</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="stat-card">
            <div class="stat-icon">📈</div>
            <div class="stat-label">Avg Daily</div>
            <div class="stat-value">2,900</div>
            <div class="stat-sub">Rentals per day</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class="stat-card">
            <div class="stat-icon">⏰</div>
            <div class="stat-label">Peak Hour</div>
            <div class="stat-value">17:00</div>
            <div class="stat-sub">Avg 257 bikes/hr</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown("""
        <div class="stat-card">
            <div class="stat-icon">🌦️</div>
            <div class="stat-label">Weather Effect</div>
            <div class="stat-value">13% drop</div>
            <div class="stat-sub">Clear vs rainy days</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Chart section
    st.markdown("""
    <div class="stat-card">
        <div class="section-title" style="font-size:1.1rem;">Average Hourly Demand Pattern</div>
        <div class="section-sub">Registered vs Casual riders across 24 hours</div>
    </div>
    """, unsafe_allow_html=True)

    # Sample chart data
    hours = list(range(24))
    registered = [10,5,3,2,3,15,60,130,180,120,100,110,115,100,110,120,185,190,150,100,80,60,40,20]
    casual = [5,2,1,1,2,5,10,20,35,45,50,55,60,58,55,50,45,40,35,25,20,15,10,6]

    import pandas as pd
    chart_data = pd.DataFrame({'Hour': hours, 'Registered': registered, 'Casual': casual})
    st.line_chart(chart_data.set_index('Hour'), color=["#14b8a6", "#7c3aed"])

# ---- Predict Demand Page ----
elif "Predict" in page:
    st.markdown('<div class="section-title">Predict Demand</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Enter conditions to forecast bike rental demand</div>', unsafe_allow_html=True)

    col_form, col_result = st.columns([3, 2], gap="large")

    with col_form:
        st.markdown('<div class="stat-card">', unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            season = st.selectbox("🗓 Season", [1, 2, 3, 4], format_func=lambda x: {1:"Spring",2:"Summer",3:"Fall",4:"Winter"}[x])
            yr = st.selectbox("📅 Year", [0, 1], format_func=lambda x: "2011" if x==0 else "2012")
            month = st.slider("📆 Month", 1, 12)
            holiday = st.selectbox("🏖 Holiday", [0, 1], format_func=lambda x: "No" if x==0 else "Yes")
            workingday = st.selectbox("💼 Working Day", [0, 1], format_func=lambda x: "No" if x==0 else "Yes")
        with c2:
            weather = st.selectbox("🌤 Weather", [1, 2, 3, 4], format_func=lambda x: {1:"Clear",2:"Cloudy",3:"Light Rain",4:"Heavy Rain"}[x])
            temp = st.number_input("🌡 Temperature (norm.)", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
            atemp = st.number_input("🤔 Feels Like (norm.)", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
            humidity = st.number_input("💧 Humidity", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
            windspeed = st.number_input("💨 Wind Speed (norm.)", min_value=0.0, max_value=1.0, value=0.2, step=0.01)

        hour = st.slider("🕐 Hour of Day", 0, 23, 17)
        st.markdown('</div>', unsafe_allow_html=True)

        predict_btn = st.button("🔮  Predict Rental Demand")

    with col_result:
        st.markdown("""
        <div class="stat-card" style="height:100%">
            <div class="stat-label">Prediction Output</div>
        """, unsafe_allow_html=True)

        if predict_btn:
            features = np.array([[season, yr, month, holiday, workingday, weather, temp, atemp, humidity, windspeed, hour]])
            try:
                model = pickle.load(open("bike_model.pkl", "rb"))
                prediction_log = model.predict(features)
                prediction = int(np.exp(prediction_log)[0])

                st.markdown(f"""
                <div class="result-box">
                    <div style="color:#64748b; font-size:0.85rem; margin-bottom:12px;">ESTIMATED RENTALS</div>
                    <div class="result-number">{prediction:,}</div>
                    <div class="result-label">bikes / hour</div>
                </div>

                <div style="margin-top:20px; display:flex; gap:12px; flex-wrap:wrap;">
                    <div style="flex:1; background:rgba(20,184,166,0.08); border-radius:10px; padding:14px; text-align:center;">
                        <div style="color:#64748b; font-size:0.75rem;">SEASON</div>
                        <div style="color:#f1f5f9; font-weight:600;">{["","Spring","Summer","Fall","Winter"][season]}</div>
                    </div>
                    <div style="flex:1; background:rgba(20,184,166,0.08); border-radius:10px; padding:14px; text-align:center;">
                        <div style="color:#64748b; font-size:0.75rem;">HOUR</div>
                        <div style="color:#f1f5f9; font-weight:600;">{hour:02d}:00</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            except FileNotFoundError:
                st.markdown("""
                <div class="result-box">
                    <div style="color:#f59e0b; font-size:1rem;">⚠️ Model file not found</div>
                    <div class="result-label">Place bike_model.pkl in the same directory</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="result-box" style="opacity:0.5;">
                <div style="color:#64748b; font-size:2rem; margin-bottom:10px;">🔮</div>
                <div style="color:#475569;">Fill in the conditions and<br>click Predict to see results</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

# ---- Weather Forecast Page ----
elif "Weather" in page:
    st.markdown('<div class="section-title">Weather Forecast</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Upcoming weather conditions and their impact on rental demand</div>', unsafe_allow_html=True)

    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    icons = ["☀️", "⛅", "🌧️", "☀️", "☀️", "⛅", "🌤️"]
    temps = [22, 19, 15, 24, 26, 21, 23]
    impact = ["+12%", "-5%", "-28%", "+18%", "+22%", "+8%", "+14%"]
    colors = ["#22c55e","#f59e0b","#ef4444","#22c55e","#22c55e","#22c55e","#22c55e"]

    cols = st.columns(7)
    for i, col in enumerate(cols):
        with col:
            st.markdown(f"""
            <div class="stat-card" style="text-align:center; padding:16px 10px;">
                <div style="font-size:0.75rem; color:#64748b; margin-bottom:8px;">{days[i]}</div>
                <div style="font-size:1.8rem; margin-bottom:8px;">{icons[i]}</div>
                <div style="font-weight:600; color:#f1f5f9;">{temps[i]}°C</div>
                <div style="font-size:0.8rem; color:{colors[i]}; margin-top:6px; font-weight:600;">{impact[i]}</div>
            </div>
            """, unsafe_allow_html=True)

# ---- Analytics Page ----
elif "Analytics" in page:
    st.markdown('<div class="section-title">Analytics</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Deep dive into rental patterns and model performance</div>', unsafe_allow_html=True)

    import pandas as pd
    import numpy as np

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="stat-card">', unsafe_allow_html=True)
        st.markdown('<div style="color:#94a3b8; font-size:0.9rem; font-weight:600; margin-bottom:12px;">📅 Monthly Rentals</div>', unsafe_allow_html=True)
        monthly = pd.DataFrame({
            'Month': ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'],
            'Rentals': [40000, 45000, 72000, 88000, 110000, 115000, 120000, 118000, 105000, 95000, 68000, 42000]
        })
        st.bar_chart(monthly.set_index('Month'), color="#14b8a6")
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="stat-card">', unsafe_allow_html=True)
        st.markdown('<div style="color:#94a3b8; font-size:0.9rem; font-weight:600; margin-bottom:12px;">🌤 Weather Impact</div>', unsafe_allow_html=True)
        weather_df = pd.DataFrame({
            'Condition': ['Clear','Cloudy','Light Rain','Heavy Rain'],
            'Avg Rentals': [320, 280, 220, 90]
        })
        st.bar_chart(weather_df.set_index('Condition'), color="#06b6d4")
        st.markdown('</div>', unsafe_allow_html=True)