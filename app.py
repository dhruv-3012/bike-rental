import streamlit as st
import pickle
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Bike Rental AI",
    page_icon="🚲",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
#  DATA
# ─────────────────────────────────────────────
hours = list(range(24))
registered = [12,6,4,3,4,18,70,155,185,130,108,112,118,105,112,122,178,192,155,108,84,62,42,22]
casual      = [4, 2,1,1,2, 6,12, 25, 42, 52, 58, 62, 65, 63, 60, 58, 48, 38, 28,22,16,10, 5]
months_label = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
total_rentals= [40200,22000,52000,72000,96000,115000,138000,135000,138000,104000,72000,25000]
avg_temp     = [4,5,9,15,19,24,27,26,21,15,9,5]

C = {
    'teal':'#14b8a6','cyan':'#38bdf8','purple':'#a78bfa',
    'orange':'#fb923c','green':'#4ade80',
    'grid':'rgba(255,255,255,0.04)','muted':'#3a5472','text':'#c8d6e8',
}

# ─────────────────────────────────────────────
#  HELPER FUNCTIONS
# ─────────────────────────────────────────────
def normalize_temperature(temp_celsius):
    """Convert Celsius temperature to normalized value (0-1 range)"""
    min_temp = -5
    max_temp = 40
    clipped_temp = np.clip(temp_celsius, min_temp, max_temp)
    normalized = (clipped_temp - min_temp) / (max_temp - min_temp)
    return normalized

def normalize_humidity(humidity_percent):
    """Convert humidity percentage (0-100) to normalized value (0-1 range)"""
    clipped_humidity = np.clip(humidity_percent, 0, 100)
    normalized = clipped_humidity / 100.0
    return normalized

def normalize_windspeed(windspeed_kmh):
    """Convert wind speed in km/h to normalized value (0-1 range)"""
    min_wind = 0
    max_wind = 50
    clipped_wind = np.clip(windspeed_kmh, min_wind, max_wind)
    normalized = (clipped_wind - min_wind) / (max_wind - min_wind)
    return normalized

def predict_for_hours(model, base_features, hours_range):
    """Generate predictions for a range of hours while keeping other features constant"""
    predictions = []
    for hour in hours_range:
        features = base_features.copy()
        features[8] = hour  # hour is at index 8
        pred_log = model.predict(features.reshape(1, -1))
        pred = int(max(0, np.expm1(pred_log)[0]))
        predictions.append(pred)
    return predictions

# ─────────────────────────────────────────────
#  CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    background-color: #080c18 !important;
    color: #c8d6e8 !important;
}
.stApp { background: #080c18 !important; }

/* Custom navigation area */
.main-header {
    font-size: 1.65rem;
    font-weight: 800;
    color: #e2ecfb;
}

/* Top nav bar */
.top-nav-bar {
    display: flex;
    align-items: center;
    gap: 4px;
    padding: 8px 12px;
    background: linear-gradient(145deg, #0f1826, #0c1522);
    border-bottom: 1px solid rgba(56,189,248,0.10);
    margin-bottom: 24px;
}

/* Radio nav styling - tab style with large icons */
div[data-testid="stRadio"] > label { display: none !important; }
div[data-testid="stRadio"] > div {
    display: flex !important;
    flex-direction: row !important;
    gap: 0 !important;
    background: transparent !important;
}
div[data-testid="stRadio"] label {
    display: flex !important;
    align-items: center !important;
    gap: 10px !important;
    padding: 10px 28px !important;
    border-bottom: 2px solid transparent !important;
    color: #5a7595 !important;
    font-size: 1.35rem !important;
    font-weight: 600 !important;
    cursor: pointer !important;
    transition: all 0.2s !important;
    border-radius: 0 !important;
    background: transparent !important;
    white-space: nowrap !important;
}
div[data-testid="stRadio"] label:hover {
    color: #c8d6e8 !important;
    border-bottom: 2px solid rgba(56,189,248,0.4) !important;
}
/* Hide the radio circle dot */
div[data-testid="stRadio"] label > div:first-child {
    display: none !important;
}
/* Active/checked tab */
div[data-testid="stRadio"] label:has(input:checked) {
    color: #38bdf8 !important;
    border-bottom: 2px solid #38bdf8 !important;
}

/* KPI cards */
.kpi-card {
    background: linear-gradient(145deg, #111827, #0f1c2e);
    border: 1px solid rgba(56,189,248,0.12);
    border-radius: 14px;
    padding: 20px 22px;
    position: relative;
    overflow: hidden;
}
.kpi-card::after {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: var(--accent, linear-gradient(90deg,#14b8a6,#38bdf8));
}
.kpi-label {
    font-size: 0.68rem;
    letter-spacing: 0.13em;
    text-transform: uppercase;
    color: #3a5472;
    font-weight: 600;
    margin-bottom: 10px;
}
.kpi-value {
    font-size: 1.9rem;
    font-weight: 800;
    color: #e8f0fb;
    font-family: 'JetBrains Mono', monospace;
    line-height: 1;
    margin-bottom: 6px;
}
.kpi-sub { font-size: 0.74rem; color: #2e4461; }
.kpi-icon { position: absolute; top: 16px; right: 18px; font-size: 1.35rem; opacity: 0.5; }

.sec-title { font-size: 1rem; font-weight: 700; color: #e2ecfb; margin-bottom: 2px; }
.sec-sub { font-size: 0.76rem; color: #3a5472; margin-bottom: 12px; }

.chart-card {
    background: linear-gradient(145deg, #0f1826, #0c1522);
    border: 1px solid rgba(56,189,248,0.09);
    border-radius: 14px;
    padding: 20px 18px 10px;
}

/* Form styling */
.stSelectbox > div > div, 
.stNumberInput > div > div > input,
.stSlider > div {
    background: transparent !important;
    border: none !important;
    border-bottom: 1px solid rgba(56, 189, 248, 0.2) !important;
    border-radius: 0 !important;
    box-shadow: none !important;
    padding-left: 0 !important;
}

.stSelectbox > div > div:focus, 
.stNumberInput > div > div > input:focus {
    border-bottom: 1px solid #38bdf8 !important;
}

/* Widget styling */
label, .stSelectbox label, .stNumberInput label, .stSlider label {
    color: #5a7595 !important;
    font-size: 0.75rem !important;
    font-weight: 400 !important;
    margin-bottom: 2px !important;
}

/* Predict button */
.stButton > button {
    background: linear-gradient(90deg, #14b8a6, #38bdf8) !important;
    color: #040810 !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    padding: 13px 0 !important;
    width: 100% !important;
    letter-spacing: 0.04em !important;
    margin-top: 20px;
}
.stButton > button:hover { opacity: 0.88 !important; transform: translateY(-1px) !important; }

/* Result card */
.result-card {
    background: linear-gradient(145deg, #0a1f14, #071510);
    border: 1px solid rgba(20,184,166,0.22);
    border-radius: 14px;
    padding: 30px 22px;
    text-align: center;
}
.result-num {
    font-size: 3.4rem;
    font-weight: 800;
    color: #2dd4bf;
    font-family: 'JetBrains Mono', monospace;
    line-height: 1;
}

/* Full width chart container */
.full-width-chart {
    width: 100%;
    margin-top: 30px;
    margin-bottom: 20px;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  SESSION STATE FOR NAVIGATION
# ─────────────────────────────────────────────
if 'page' not in st.session_state:
    st.session_state.page = "Dashboard"

# ─────────────────────────────────────────────
#  TOP NAVIGATION BAR
# ─────────────────────────────────────────────
header_row, status_col = st.columns([6, 1])
with header_row:
    st.markdown('<div class="main-header">🚲 Bike Rental AI</div>', unsafe_allow_html=True)
with status_col:
    st.markdown("""
    <div style="padding:6px 12px;background:rgba(20,184,166,0.05);border:1px solid rgba(20,184,166,0.13);border-radius:10px;display:flex;align-items:center;gap:8px;margin-top:10px;">
        <span style="width:8px;height:8px;border-radius:50%;background:#22c55e;box-shadow:0 0 7px #22c55e;display:inline-block;flex-shrink:0;"></span>
        <span style="color:#7a91b0;font-size:0.78rem;white-space:nowrap;">Active · Ready</span>
    </div>
    """, unsafe_allow_html=True)

# Navigation using st.radio styled as tabs
nav_pages = ["Dashboard", "Weather Forecast", "Predict Demand", "Analytics"]
nav_labels = ["📊  Dashboard", "🌤  Weather", "🔮  Predict", "📈  Analytics"]

selected = st.radio(
    label="nav",
    options=nav_pages,
    format_func=lambda x: nav_labels[nav_pages.index(x)],
    horizontal=True,
    label_visibility="collapsed",
    index=nav_pages.index(st.session_state.page) if st.session_state.page in nav_pages else 0,
    key="nav_radio"
)
if selected != st.session_state.page:
    st.session_state.page = selected
    st.rerun()

st.markdown('<hr style="border:none;border-top:1px solid rgba(56,189,248,0.10);margin:4px 0 20px 0;">', unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  PAGE CONTENT
# ─────────────────────────────────────────────
with st.container():
        # ─────────────────────────────────────────────
        #  DASHBOARD PAGE
        # ─────────────────────────────────────────────
        if st.session_state.page == "Dashboard":
            col_h, col_w = st.columns([5,1])
            with col_h:
                st.markdown('<div class="main-header">Bike Rental Dashboard</div>', unsafe_allow_html=True)
                st.markdown('<div style="color:#3a5472;margin-bottom:18px;">Historical analysis · 365 days · 8,760 hourly records</div>', unsafe_allow_html=True)
            with col_w:
                st.markdown('<div style="text-align:right;padding-top:6px;"><span style="background:linear-gradient(90deg,#fef3c7,#fde68a);color:#92400e;border-radius:30px;padding:7px 16px;font-weight:700;font-size:0.82rem;">☀️ Clear 22°C</span></div>', unsafe_allow_html=True)

            # KPI Cards
            c1, c2, c3, c4 = st.columns(4, gap="small")
            with c1:
                st.markdown("""
                <div class="kpi-card" style="--accent:linear-gradient(90deg,#14b8a6,#38bdf8);">
                    <span class="kpi-icon">🚲</span>
                    <div class="kpi-label">TOTAL RENTALS</div>
                    <div class="kpi-value">1,058,318</div>
                    <div class="kpi-sub">Past 12 months</div>
                </div>
                """, unsafe_allow_html=True)
            with c2:
                st.markdown("""
                <div class="kpi-card" style="--accent:linear-gradient(90deg,#a78bfa,#818cf8);">
                    <span class="kpi-icon">📈</span>
                    <div class="kpi-label">AVG DAILY</div>
                    <div class="kpi-value">2,900</div>
                    <div class="kpi-sub">Rentals per day</div>
                </div>
                """, unsafe_allow_html=True)
            with c3:
                st.markdown("""
                <div class="kpi-card" style="--accent:linear-gradient(90deg,#fb923c,#f59e0b);">
                    <span class="kpi-icon">⏰</span>
                    <div class="kpi-label">PEAK HOUR</div>
                    <div class="kpi-value">17:00</div>
                    <div class="kpi-sub">Avg 257 bikes / hr</div>
                </div>
                """, unsafe_allow_html=True)
            with c4:
                st.markdown("""
                <div class="kpi-card" style="--accent:linear-gradient(90deg,#4ade80,#22d3ee);">
                    <span class="kpi-icon">🌦</span>
                    <div class="kpi-label">WEATHER EFFECT</div>
                    <div class="kpi-value">13% drop</div>
                    <div class="kpi-sub">Clear vs rainy days</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Hourly chart
            st.markdown('<div class="chart-card">', unsafe_allow_html=True)
            st.markdown('<div class="sec-title">Average Hourly Demand Pattern</div><div class="sec-sub">Registered vs Casual riders across 24 hours</div>', unsafe_allow_html=True)

            fig1 = go.Figure()
            fig1.add_trace(go.Scatter(
                x=hours, y=registered, name="Registered",
                mode='lines', fill='tozeroy',
                line=dict(color=C['cyan'], width=2.5, shape='spline'),
                fillcolor='rgba(56,189,248,0.13)',
                hovertemplate='<b>Registered</b>: %{y}'
            ))
            fig1.add_trace(go.Scatter(
                x=hours, y=casual, name="Casual",
                mode='lines', fill='tozeroy',
                line=dict(color=C['purple'], width=2.5, shape='spline'),
                fillcolor='rgba(167,139,250,0.13)',
                hovertemplate='<b>Casual</b>: %{y}'
            ))
            fig1.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                font=dict(family='Plus Jakarta Sans', color=C['text'], size=12),
                height=295, margin=dict(l=10,r=10,t=6,b=40),
                xaxis=dict(tickvals=hours, ticktext=[f"{h:02d}:00" for h in hours], gridcolor=C['grid'], color=C['muted']),
                yaxis=dict(range=[0,215], gridcolor=C['grid'], color=C['muted']),
                hovermode='x unified',
                legend=dict(bgcolor='rgba(0,0,0,0)', orientation='h', yanchor='bottom', y=-0.26, xanchor='center', x=0.5)
            )
            st.plotly_chart(fig1, use_container_width=True, config={'displayModeBar':False})
            st.markdown('</div>', unsafe_allow_html=True)

        # ─────────────────────────────────────────────
        #  WEATHER FORECAST PAGE
        # ─────────────────────────────────────────────
        elif st.session_state.page == "Weather Forecast":
            st.markdown('<div class="main-header">Weather Forecast</div>', unsafe_allow_html=True)
            st.markdown('<div style="color:#3a5472;margin-bottom:22px;">7-day outlook and estimated impact on bike demand</div>', unsafe_allow_html=True)

            days    = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
            icons   = ["☀️","⛅","🌧️","☀️","☀️","⛅","🌤️"]
            highs   = [22,19,15,24,26,21,23]
            lows    = [14,13,10,16,17,14,15]
            impacts = ["+12%","-5%","-28%","+18%","+22%","+8%","+14%"]
            icolors = ["#4ade80","#fb923c","#f87171","#4ade80","#4ade80","#4ade80","#4ade80"]

            cols = st.columns(7, gap="small")
            for i, col in enumerate(cols):
                with col:
                    st.markdown(f"""
                    <div class="kpi-card" style="--accent:{icolors[i]};padding:16px 10px;text-align:center;">
                        <div style="font-size:0.65rem;color:#3a5472;letter-spacing:.1em;text-transform:uppercase;">{days[i]}</div>
                        <div style="font-size:1.8rem;margin:10px 0;">{icons[i]}</div>
                        <div style="font-weight:700;color:#e2ecfb;font-size:0.95rem;">{highs[i]}°C</div>
                        <div style="font-size:0.72rem;color:#3a5472;">{lows[i]}°C low</div>
                        <div style="margin-top:10px;font-size:0.82rem;font-weight:700;color:{icolors[i]};">{impacts[i]}</div>
                    </div>
                    """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="chart-card">', unsafe_allow_html=True)
            st.markdown('<div class="sec-title">Forecasted Demand</div><div class="sec-sub">Estimated daily rentals for next 7 days</div>', unsafe_allow_html=True)
            
            fig_w = go.Figure(go.Bar(
                x=days, y=[3360,2760,1920,3720,4080,3180,3540],
                marker_color=C['cyan'], marker_line_width=0,
                hovertemplate='%{x}: <b>%{y:,}</b> est. rentals<extra></extra>'
            ))
            fig_w.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                font=dict(family='Plus Jakarta Sans', color=C['text'], size=12),
                height=240, margin=dict(l=10,r=10,t=6,b=10),
                showlegend=False, bargap=0.3,
                xaxis=dict(gridcolor=C['grid'], color=C['muted']),
                yaxis=dict(gridcolor=C['grid'], color=C['muted'])
            )
            st.plotly_chart(fig_w, use_container_width=True, config={'displayModeBar':False})
            st.markdown('</div>', unsafe_allow_html=True)

        # ─────────────────────────────────────────────
        #  PREDICT DEMAND PAGE
        # ─────────────────────────────────────────────
        elif st.session_state.page == "Predict Demand":
            st.markdown('<div class="main-header">Predict Demand</div>', unsafe_allow_html=True)
            st.markdown('<div style="color:#3a5472;margin-bottom:22px;">Enter conditions to forecast hourly bike rentals</div>', unsafe_allow_html=True)

            col_form, col_out = st.columns([3,2], gap="large")

            with col_form:
                r1, r2 = st.columns(2)
                with r1:
                    season = st.selectbox("Season", [1,2,3,4],
                        format_func=lambda x:{1:"🌸 Spring",2:"☀️ Summer",3:"🍂 Fall",4:"❄️ Winter"}[x],
                        key="season_select")
                    yr = st.selectbox("Year", [0,1], format_func=lambda x:"2011" if x==0 else "2012", key="year_select")
                    month = st.slider("Month", 1, 12, 6, key="month_slider")
                    holiday = st.selectbox("Holiday", [0,1], format_func=lambda x:"No" if x==0 else "Yes", key="holiday_select")
                    workingday = st.selectbox("Working Day", [0,1], format_func=lambda x:"No" if x==0 else "Yes", key="working_select")
                
                with r2:
                    weather = st.selectbox("Weather", [1,2,3,4],
                        format_func=lambda x:{1:"☀️ Clear",2:"⛅ Cloudy",3:"🌧 Light Rain",4:"⛈ Heavy Rain"}[x],
                        key="weather_select")
                    temp_celsius = st.number_input("Temperature (°C)", -10.0, 45.0, 20.0, 0.5, key="temp_input")
                    atemp_celsius = st.number_input("Feels Like (°C)", -10.0, 45.0, 20.0, 0.5, key="atemp_input")
                    humidity_percent = st.number_input("Humidity (%)", 0.0, 100.0, 60.0, 1.0, key="humidity_input")
                    windspeed_kmh = st.number_input("Wind Speed (km/h)", 0.0, 60.0, 10.0, 0.5, key="wind_input")
                
                hour = st.slider("Hour of Day (0–23)", 0, 23, 17, key="hour_slider")
                day = st.slider("Day of Month", 1, 31, 15, key="day_slider")
                dayofweek = st.selectbox("Day of Week", [0,1,2,3,4,5,6], 
                                        format_func=lambda x: ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"][x],
                                        key="dow_select")
                
                predict_btn = st.button("🔮  Predict Rental Demand", key="predict_btn")

            with col_out:
                # Normalize the values
                temp_norm = normalize_temperature(temp_celsius)
                atemp_norm = normalize_temperature(atemp_celsius)
                humidity_norm = normalize_humidity(humidity_percent)
                windspeed_norm = normalize_windspeed(windspeed_kmh)
                
                # Create base feature array
                base_features = np.array([
                    season, holiday, workingday, weather,
                    temp_norm, atemp_norm, humidity_norm, windspeed_norm,
                    hour, day, month, yr, dayofweek
                ])
                
                season_names = {1:"Spring",2:"Summer",3:"Fall",4:"Winter"}
                weather_names = {1:"Clear",2:"Cloudy",3:"Light Rain",4:"Heavy Rain"}

                if predict_btn:
                    try:
                        model = pickle.load(open("bike_model.pkl", "rb"))
                        
                        # Predict for the selected hour
                        features = base_features.reshape(1, -1)
                        pred_log = model.predict(features)
                        prediction = int(max(0, np.expm1(pred_log)[0]))
                        
                        # Generate predictions for all hours
                        all_hours_predictions = predict_for_hours(model, base_features, hours)
                        
                        # Display result
                        st.markdown(f"""
                        <div class="result-card">
                            <div style="font-size:0.65rem;letter-spacing:.14em;text-transform:uppercase;color:#1a4036;margin-bottom:14px;">Estimated Rentals / Hour</div>
                            <div class="result-num">{prediction:,}</div>
                            <div style="color:#1a4036;margin-top:8px;font-size:0.8rem;">bikes per hour</div>
                            <div style="display:flex;gap:10px;margin-top:22px;">
                                <div style="flex:1;background:rgba(20,184,166,0.07);border-radius:10px;padding:12px;text-align:center;">
                                    <div style="color:#1a4036;font-size:0.65rem;letter-spacing:.1em;text-transform:uppercase;">Season</div>
                                    <div style="color:#e2ecfb;font-weight:600;margin-top:4px;font-size:0.85rem;">{season_names[season]}</div>
                                </div>
                                <div style="flex:1;background:rgba(20,184,166,0.07);border-radius:10px;padding:12px;text-align:center;">
                                    <div style="color:#1a4036;font-size:0.65rem;letter-spacing:.1em;text-transform:uppercase;">Hour</div>
                                    <div style="color:#e2ecfb;font-weight:600;margin-top:4px;font-size:0.85rem;">{hour:02d}:00</div>
                                </div>
                                <div style="flex:1;background:rgba(20,184,166,0.07);border-radius:10px;padding:12px;text-align:center;">
                                    <div style="color:#1a4036;font-size:0.65rem;letter-spacing:.1em;text-transform:uppercase;">Weather</div>
                                    <div style="color:#e2ecfb;font-weight:600;margin-top:4px;font-size:0.85rem;">{weather_names[weather]}</div>
                                </div>
                            </div>
                            <div style="display:flex;gap:10px;margin-top:10px;justify-content:center;font-size:0.7rem;color:#3a5472;">
                                <span>{temp_celsius:.1f}°C</span> • <span>{humidity_percent:.0f}%</span> • <span>{windspeed_kmh:.0f} km/h</span>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                    except FileNotFoundError:
                        st.markdown("""
                        <div class="result-card" style="opacity:0.7;">
                            <div style="font-size:2rem;margin-bottom:12px;">⚠️</div>
                            <div style="color:#fb923c;font-weight:600;">bike_model.pkl not found</div>
                            <div style="color:#3a5472;font-size:0.8rem;margin-top:8px;">Please train and save the model first</div>
                        </div>
                        """, unsafe_allow_html=True)
                    except Exception as e:
                        st.markdown(f"""
                        <div class="result-card" style="opacity:0.7;">
                            <div style="font-size:2rem;margin-bottom:12px;">⚠️</div>
                            <div style="color:#fb923c;font-weight:600;">Prediction Error</div>
                            <div style="color:#3a5472;font-size:0.8rem;margin-top:8px;">{str(e)}</div>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="result-card" style="opacity:0.4;">
                        <div style="font-size:2.4rem;margin-bottom:14px;">🔮</div>
                        <div style="color:#3a5472;font-size:0.88rem;">Fill in the conditions<br>and click Predict</div>
                    </div>
                    """, unsafe_allow_html=True)
            
            # ─────────────────────────────────────────────
            #  HOURLY PREDICTION GRAPH - LARGE AND AT BOTTOM
            # ─────────────────────────────────────────────
            if predict_btn and 'all_hours_predictions' in locals() and 'prediction' in locals():
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown('<div class="full-width-chart">', unsafe_allow_html=True)
                st.markdown('<div class="chart-card">', unsafe_allow_html=True)
                st.markdown(f'<div class="sec-title">Hourly Demand Forecast</div><div class="sec-sub">Predicted bike rentals throughout the day ({season_names[season]}, {weather_names[weather]})</div>', unsafe_allow_html=True)
                
                fig_hourly = go.Figure()
                
                # Add the prediction line
                fig_hourly.add_trace(go.Scatter(
                    x=hours, 
                    y=all_hours_predictions,
                    name="Predicted Demand",
                    mode='lines+markers',
                    line=dict(color=C['cyan'], width=4, shape='spline'),
                    marker=dict(size=10, color=C['cyan']),
                    hovertemplate='Hour %{x:02d}:00<br><b>%{y:,}</b> bikes<extra></extra>'
                ))
                
                # Add a marker for the selected hour
                fig_hourly.add_trace(go.Scatter(
                    x=[hour],
                    y=[prediction],
                    name=f"Selected Hour ({hour:02d}:00)",
                    mode='markers',
                    marker=dict(size=20, color=C['orange'], line=dict(width=3, color='white')),
                    hovertemplate='Selected: %{x:02d}:00<br><b>%{y:,}</b> bikes<extra></extra>'
                ))
                
                # Update layout for very large graph
                fig_hourly.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)', 
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(family='Plus Jakarta Sans', color=C['text'], size=14),
                    height=700,  # Much larger height
                    margin=dict(l=100, r=60, t=60, b=120),  # Increased margins
                    xaxis=dict(
                        tickvals=hours, 
                        ticktext=[f"{h:02d}:00" for h in hours], 
                        tickangle=-45,
                        gridcolor=C['grid'], 
                        color=C['muted'], 
                        title="Hour of Day",
                        title_font=dict(size=16),
                        tickfont=dict(size=12)
                    ),
                    yaxis=dict(
                        gridcolor=C['grid'], 
                        color=C['muted'], 
                        title="Predicted Rentals (bikes per hour)",
                        title_font=dict(size=16),
                        tickfont=dict(size=12)
                    ),
                    showlegend=True,
                    legend=dict(
                        bgcolor='rgba(0,0,0,0)', 
                        orientation='h', 
                        yanchor='bottom', 
                        y=1.02, 
                        xanchor='center', 
                        x=0.5,
                        font=dict(size=14)
                    ),
                    hovermode='x unified',
                    hoverlabel=dict(
                        bgcolor='#1a2840', 
                        bordercolor='rgba(56,189,248,0.3)',
                        font=dict(family='Plus Jakarta Sans', color='#e2ecfb', size=14)
                    )
                )
                
                st.plotly_chart(fig_hourly, use_container_width=True, config={'displayModeBar':False})
                st.markdown('</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

        # ─────────────────────────────────────────────
        #  ANALYTICS PAGE
        # ─────────────────────────────────────────────
        elif st.session_state.page == "Analytics":
            st.markdown('<div class="main-header">Analytics</div>', unsafe_allow_html=True)
            st.markdown('<div style="color:#3a5472;margin-bottom:22px;">Deep dive into rental patterns and trends</div>', unsafe_allow_html=True)

            # Monthly dual-axis chart
            st.markdown('<div class="chart-card">', unsafe_allow_html=True)
            st.markdown('<div class="sec-title">Monthly Rental Volume</div><div class="sec-sub">Total rentals per month with average temperature</div>', unsafe_allow_html=True)

            fig_m = make_subplots(specs=[[{"secondary_y": True}]])
            fig_m.add_trace(go.Bar(
                x=months_label, y=total_rentals, name="Total Rentals",
                marker_color=C['cyan'], marker_line_width=0,
                hovertemplate='%{x}: <b>%{y:,}</b><extra></extra>'
            ), secondary_y=False)
            fig_m.add_trace(go.Bar(
                x=months_label, y=avg_temp, name="Avg Temp (°C)",
                marker_color=C['orange'], marker_line_width=0,
                hovertemplate='%{x}: <b>%{y}°C</b><extra></extra>'
            ), secondary_y=True)

            fig_m.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                font=dict(family='Plus Jakarta Sans', color=C['text'], size=12),
                margin=dict(l=10,r=10,t=6,b=40), height=310,
                bargap=0.15, bargroupgap=0.05,
                hovermode='x unified',
                legend=dict(bgcolor='rgba(0,0,0,0)', orientation='h', yanchor='bottom', y=-0.28, xanchor='center', x=0.5)
            )
            fig_m.update_xaxes(gridcolor=C['grid'], color=C['muted'])
            fig_m.update_yaxes(gridcolor=C['grid'], color=C['muted'], secondary_y=False)
            fig_m.update_yaxes(gridcolor='rgba(0,0,0,0)', color=C['orange'], secondary_y=True)
            st.plotly_chart(fig_m, use_container_width=True, config={'displayModeBar':False})
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Insight cards
            insights = [
                ("🌡️","Temperature Sweet Spot",
                 "Demand peaks at 20–25°C. Below 5°C or above 35°C reduces rentals by up to 60%."),
                ("💧","Humidity Effect",
                 "Humidity above 80% correlates with 30–40% fewer rentals compared to dry conditions."),
                ("👥","Commuter Pattern",
                 "Registered users dominate weekday peaks at 8am & 5pm. Casual riders peak on weekends."),
                ("🌦️","Weather Sensitivity",
                 "Rainy conditions reduce demand by ~45%. Clear sky days see up to 3× more casual riders."),
            ]
            cols = st.columns(4, gap="small")
            for i, (icon, title, body) in enumerate(insights):
                with cols[i]:
                    st.markdown(f"""
                    <div class="kpi-card" style="--accent:linear-gradient(90deg,#38bdf8,#14b8a6);">
                        <div style="font-size:1.3rem;margin-bottom:10px;">{icon}</div>
                        <div style="font-weight:700;color:#e2ecfb;font-size:0.86rem;margin-bottom:8px;">{title}</div>
                        <div style="font-size:0.76rem;color:#3a5472;line-height:1.5;">{body}</div>
                    </div>
                    """, unsafe_allow_html=True)
