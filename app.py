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
#  GLOBAL CSS
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
#MainMenu, footer, header { visibility: hidden; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: #0c1220 !important;
    border-right: 1px solid rgba(56,189,248,0.1) !important;
    min-width: 240px !important;
    max-width: 240px !important;
}
[data-testid="stSidebar"] > div:first-child { padding: 0 !important; }
[data-testid="stSidebar"] .stRadio > div { gap: 0 !important; }
[data-testid="stSidebar"] .stRadio label {
    display: flex !important;
    align-items: center !important;
    width: 100% !important;
    padding: 11px 20px !important;
    margin: 2px 0 !important;
    border-radius: 10px !important;
    color: #5a7595 !important;
    font-size: 0.88rem !important;
    font-weight: 500 !important;
    cursor: pointer !important;
    transition: all 0.18s !important;
    background: transparent !important;
}
[data-testid="stSidebar"] .stRadio label:hover {
    background: rgba(56,189,248,0.07) !important;
    color: #c8d6e8 !important;
}
[data-testid="stSidebar"] .stRadio [data-testid="stMarkdownContainer"] p { margin: 0 !important; }
[data-testid="stSidebar"] .stRadio [data-baseweb="radio"] > div:first-child { display: none !important; }

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

/* Form & result */
.form-card {
    background: #0f1826;
    border: 1px solid rgba(56,189,248,0.1);
    border-radius: 14px;
    padding: 24px 20px;
}
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

/* Widget styling */
label, .stSelectbox label, .stNumberInput label, .stSlider label {
    color: #5a7595 !important;
    font-size: 0.8rem !important;
    font-weight: 500 !important;
}
.stSelectbox > div > div {
    background: #111d2e !important;
    border: 1px solid rgba(56,189,248,0.13) !important;
    border-radius: 9px !important;
    color: #c8d6e8 !important;
}
.stNumberInput > div > div > input {
    background: #111d2e !important;
    border: 1px solid rgba(56,189,248,0.13) !important;
    border-radius: 9px !important;
    color: #c8d6e8 !important;
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
}
.stButton > button:hover { opacity: 0.88 !important; transform: translateY(-1px) !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  DATA
# ─────────────────────────────────────────────
hours = list(range(24))
registered = [12,6,4,3,4,18,70,155,185,130,108,112,118,105,112,122,178,192,155,108,84,62,42,22]
casual      = [4, 2,1,1,2, 6,12, 25, 42, 52, 58, 62, 65, 63, 60, 58, 52, 48, 38, 28,22,16,10, 5]
months_label = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
total_rentals= [40200,22000,52000,72000,96000,115000,138000,135000,138000,104000,72000,25000]
avg_temp     = [4,5,9,15,19,24,27,26,21,15,9,5]

C = {
    'teal':'#14b8a6','cyan':'#38bdf8','purple':'#a78bfa',
    'orange':'#fb923c','green':'#4ade80',
    'grid':'rgba(255,255,255,0.04)','muted':'#3a5472','text':'#c8d6e8',
}

BASE_LAYOUT = dict(
    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
    font=dict(family='Plus Jakarta Sans', color=C['text'], size=12),
    xaxis=dict(gridcolor=C['grid'], color=C['muted'], showline=False),
    yaxis=dict(gridcolor=C['grid'], color=C['muted'], showline=False),
    legend=dict(bgcolor='rgba(0,0,0,0)', orientation='h',
                yanchor='bottom', y=-0.26, xanchor='center', x=0.5,
                font=dict(size=12, color=C['text'])),
    hovermode='x unified',
    hoverlabel=dict(bgcolor='#1a2840', bordercolor='rgba(56,189,248,0.3)',
                    font=dict(family='Plus Jakarta Sans', color='#e2ecfb', size=13))
)


def kpi(icon, label, value, sub, accent):
    st.markdown(f"""
    <div class="kpi-card" style="--accent:{accent}">
        <span class="kpi-icon">{icon}</span>
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-sub">{sub}</div>
    </div>""", unsafe_allow_html=True)

def sec(title, sub):
    st.markdown(f'<div class="sec-title">{title}</div><div class="sec-sub">{sub}</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:28px 20px 16px;">
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:30px;">
            <span style="font-size:1.6rem;">🚲</span>
            <span style="font-size:1.15rem;font-weight:800;color:#e2ecfb;letter-spacing:-0.01em;">Bike Rental</span>
            <span style="background:linear-gradient(90deg,#14b8a6,#38bdf8);color:#040810;font-size:0.58rem;font-weight:800;padding:2px 7px;border-radius:6px;letter-spacing:.06em;">AI</span>
        </div>
        <div style="font-size:0.62rem;letter-spacing:.13em;text-transform:uppercase;color:#1e3050;padding:0 4px;margin-bottom:8px;">Navigation</div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio("", [
        "📊  Dashboard",
        "🌤  Weather Forecast",
        "🔮  Predict Demand",
        "📈  Analytics"
    ], label_visibility="collapsed")

    st.markdown("""
    <div style="padding:16px 20px;margin-top:20px;">
        <div style="background:rgba(20,184,166,0.05);border:1px solid rgba(20,184,166,0.13);border-radius:12px;padding:14px 16px;">
            <div style="font-size:0.62rem;letter-spacing:.12em;text-transform:uppercase;color:#1e3050;margin-bottom:8px;">Model Status</div>
            <div style="display:flex;align-items:center;gap:8px;">
                <span style="display:inline-block;width:7px;height:7px;border-radius:50%;background:#22c55e;box-shadow:0 0 7px #22c55e;flex-shrink:0;"></span>
                <span style="color:#7a91b0;font-size:0.81rem;font-weight:500;">Active · Ready</span>
            </div>
            <div style="margin-top:8px;font-size:0.72rem;color:#1e3050;">Random Forest · v2.1</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  DASHBOARD
# ─────────────────────────────────────────────
if "Dashboard" in page:

    col_h, col_w = st.columns([5,1])
    with col_h:
        st.markdown('<div style="font-size:1.65rem;font-weight:800;color:#e2ecfb;line-height:1.1;">Bike Rental Dashboard</div>', unsafe_allow_html=True)
        st.markdown('<div style="font-size:0.78rem;color:#3a5472;margin-bottom:18px;">Historical analysis · 365 days · 8,760 hourly records</div>', unsafe_allow_html=True)
    with col_w:
        st.markdown('<div style="text-align:right;padding-top:6px;"><span style="background:linear-gradient(90deg,#fef3c7,#fde68a);color:#92400e;border-radius:30px;padding:7px 16px;font-weight:700;font-size:0.82rem;">☀️ Clear 22°C</span></div>', unsafe_allow_html=True)

    c1,c2,c3,c4 = st.columns(4, gap="small")
    with c1: kpi("🚲","Total Rentals","1,058,318","Past 12 months","linear-gradient(90deg,#14b8a6,#38bdf8)")
    with c2: kpi("📈","Avg Daily","2,900","Rentals per day","linear-gradient(90deg,#a78bfa,#818cf8)")
    with c3: kpi("⏰","Peak Hour","17:00","Avg 257 bikes / hr","linear-gradient(90deg,#fb923c,#f59e0b)")
    with c4: kpi("🌦","Weather Effect","13% drop","Clear vs rainy days","linear-gradient(90deg,#4ade80,#22d3ee)")

    st.markdown("<br>", unsafe_allow_html=True)

    # Hourly chart
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    sec("Average Hourly Demand Pattern", "Registered vs Casual riders across 24 hours")

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
    lay1 = {**BASE_LAYOUT}
    lay1['height'] = 295
    lay1['margin'] = dict(l=10,r=10,t=6,b=40)
    lay1['xaxis'] = dict(
        tickvals=hours, ticktext=[f"{h}h" for h in hours],
        gridcolor=C['grid'], color=C['muted'], showline=False
    )
    lay1['yaxis'] = dict(range=[0,215], gridcolor=C['grid'], color=C['muted'])
    fig1.update_layout(**lay1)
    st.plotly_chart(fig1, use_container_width=True, config={'displayModeBar':False})
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_l, col_r = st.columns(2, gap="medium")

    with col_l:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        sec("Weather Impact on Demand", "Average hourly rentals by weather condition")
        fig2 = go.Figure(go.Bar(
            x=['Clear','Cloudy','Light Rain','Heavy Rain'],
            y=[118, 112, 106, 14],
            marker_color=[C['orange'], '#94a3b8', '#60a5fa', C['purple']],
            marker_line_width=0,
            hovertemplate='%{x}: <b>%{y}</b> avg rentals<extra></extra>'
        ))
        lay2 = {**BASE_LAYOUT}
        lay2['height'] = 280
        lay2['margin'] = dict(l=10,r=10,t=6,b=10)
        lay2['yaxis'] = dict(gridcolor=C['grid'], color=C['muted'], range=[0,145])
        lay2['bargap'] = 0.35
        lay2['showlegend'] = False
        fig2.update_layout(**lay2)
        st.plotly_chart(fig2, use_container_width=True, config={'displayModeBar':False})
        st.markdown('</div>', unsafe_allow_html=True)

    with col_r:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        sec("Seasonal Split", "Avg daily rentals by season")
        fig3 = go.Figure(go.Pie(
            labels=['Spring','Summer','Fall','Winter'],
            values=[27,34,28,12],
            hole=0.0,
            marker=dict(
                colors=[C['green'], C['orange'], '#fb923c', '#60a5fa'],
                line=dict(color='#0c1522', width=2.5)
            ),
            textinfo='label+percent',
            textfont=dict(size=13, color='#e2ecfb'),
            hovertemplate='<b>%{label}</b>: %{value}%<extra></extra>'
        ))
        lay3 = {**BASE_LAYOUT}
        lay3['height'] = 280
        lay3['margin'] = dict(l=10,r=10,t=6,b=10)
        lay3['showlegend'] = False
        fig3.update_layout(**lay3)
        st.plotly_chart(fig3, use_container_width=True, config={'displayModeBar':False})
        st.markdown('</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  WEATHER FORECAST
# ─────────────────────────────────────────────
elif "Weather" in page:
    st.markdown('<div style="font-size:1.65rem;font-weight:800;color:#e2ecfb;margin-bottom:4px;">Weather Forecast</div>', unsafe_allow_html=True)
    st.markdown('<div style="font-size:0.78rem;color:#3a5472;margin-bottom:22px;">7-day outlook and estimated impact on bike demand</div>', unsafe_allow_html=True)

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
    sec("Forecasted Demand", "Estimated daily rentals for next 7 days")
    fig_w = go.Figure(go.Bar(
        x=days, y=[3360,2760,1920,3720,4080,3180,3540],
        marker_color=C['cyan'], marker_line_width=0,
        hovertemplate='%{x}: <b>%{y:,}</b> est. rentals<extra></extra>'
    ))
    lay_w = {**BASE_LAYOUT}
    lay_w['height'] = 240
    lay_w['margin'] = dict(l=10,r=10,t=6,b=10)
    lay_w['showlegend'] = False
    lay_w['bargap'] = 0.3
    fig_w.update_layout(**lay_w)
    st.plotly_chart(fig_w, use_container_width=True, config={'displayModeBar':False})
    st.markdown('</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  PREDICT DEMAND
# ─────────────────────────────────────────────
elif "Predict" in page:
    st.markdown('<div style="font-size:1.65rem;font-weight:800;color:#e2ecfb;margin-bottom:4px;">Predict Demand</div>', unsafe_allow_html=True)
    st.markdown('<div style="font-size:0.78rem;color:#3a5472;margin-bottom:22px;">Enter conditions to forecast hourly bike rentals</div>', unsafe_allow_html=True)

    col_form, col_out = st.columns([3,2], gap="large")

    with col_form:
        st.markdown('<div class="form-card">', unsafe_allow_html=True)
        r1, r2 = st.columns(2)
        with r1:
            season = st.selectbox("Season", [1,2,3,4],
                format_func=lambda x:{1:"🌸 Spring",2:"☀️ Summer",3:"🍂 Fall",4:"❄️ Winter"}[x])
            yr = st.selectbox("Year", [0,1], format_func=lambda x:"2011" if x==0 else "2012")
            month = st.slider("Month", 1, 12, 6)
            holiday = st.selectbox("Holiday", [0,1], format_func=lambda x:"No" if x==0 else "Yes")
            workingday = st.selectbox("Working Day", [0,1], format_func=lambda x:"No" if x==0 else "Yes")
        with r2:
            weather = st.selectbox("Weather", [1,2,3,4],
                format_func=lambda x:{1:"☀️ Clear",2:"⛅ Cloudy",3:"🌧 Light Rain",4:"⛈ Heavy Rain"}[x])
            temp = st.number_input("Temperature (norm.)", 0.0, 1.0, 0.5, 0.01)
            atemp = st.number_input("Feels Like (norm.)", 0.0, 1.0, 0.5, 0.01)
            humidity = st.number_input("Humidity (norm.)", 0.0, 1.0, 0.5, 0.01)
            windspeed = st.number_input("Wind Speed (norm.)", 0.0, 1.0, 0.2, 0.01)

        hour = st.slider("Hour of Day (0–23)", 0, 23, 17)
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        predict_btn = st.button("🔮  Predict Rental Demand")

    with col_out:
        features = np.array([[season, yr, month, holiday, workingday,
                               weather, temp, atemp, humidity, windspeed, hour]])
        season_names = {1:"Spring",2:"Summer",3:"Fall",4:"Winter"}
        weather_names = {1:"Clear",2:"Cloudy",3:"Light Rain",4:"Heavy Rain"}

        if predict_btn:
            try:
                model = pickle.load(open("bike_model.pkl", "rb"))
                pred_log = model.predict(features)
                prediction = int(np.exp(pred_log)[0])
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
                </div>
                """, unsafe_allow_html=True)
            except FileNotFoundError:
                st.markdown("""
                <div class="result-card" style="opacity:0.7;">
                    <div style="font-size:2rem;margin-bottom:12px;">⚠️</div>
                    <div style="color:#fb923c;font-weight:600;">bike_model.pkl not found</div>
                    <div style="color:#3a5472;font-size:0.8rem;margin-top:8px;">Place your model in the same folder as app.py</div>
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
#  ANALYTICS
# ─────────────────────────────────────────────
elif "Analytics" in page:
    st.markdown('<div style="font-size:1.65rem;font-weight:800;color:#e2ecfb;margin-bottom:4px;">Analytics</div>', unsafe_allow_html=True)
    st.markdown('<div style="font-size:0.78rem;color:#3a5472;margin-bottom:22px;">Deep dive into rental patterns and trends</div>', unsafe_allow_html=True)

    # Monthly dual-axis chart
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    sec("Monthly Rental Volume", "Total rentals per month with average temperature")

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
        legend=dict(bgcolor='rgba(0,0,0,0)', orientation='h',
                    yanchor='bottom', y=-0.28, xanchor='center', x=0.5,
                    font=dict(size=12, color=C['text'])),
        hovermode='x unified',
        hoverlabel=dict(bgcolor='#1a2840', bordercolor='rgba(56,189,248,0.3)',
                        font=dict(family='Plus Jakarta Sans', color='#e2ecfb', size=13))
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
