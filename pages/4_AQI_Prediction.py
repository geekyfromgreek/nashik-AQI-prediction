import plotly.graph_objects as go
import streamlit as st
from utils import (
    AQI_CATEGORIES,
    inject_css,
    render_page_header,
    render_prediction_sidebar,
    plotly_layout,
)

st.set_page_config(
    page_title="AQI Prediction — AQI Analysis",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_css()
render_prediction_sidebar()

render_page_header(
    "AQI Prediction",
    "Predict Air Quality Index using the trained Artificial Neural Network.",
)

if "predicted_aqi" not in st.session_state:
    st.markdown(
        '<div class="section-card" style="text-align:center;padding:48px;">'
        '<p style="color:#64748B;font-size:15px;margin:0;">'
        "Enter prediction parameters in the sidebar and click "
        "<strong>Predict AQI</strong> to see results here."
        "</p></div>",
        unsafe_allow_html=True,
    )
    st.stop()

aqi = st.session_state["predicted_aqi"]
cat = st.session_state["predicted_category"]
clr = st.session_state["predicted_color"]
inputs = st.session_state.get("prediction_inputs", {})

col_left, col_right = st.columns([1, 1.3])

with col_left:
    st.markdown(
        f'<div class="prediction-result">'
        f'<div class="prediction-label">Predicted Air Quality Index</div>'
        f'<div class="prediction-aqi" style="color:{clr};">{aqi:.1f}</div>'
        f'<div class="prediction-category" '
        f'style="background:{clr}18;color:{clr};border:1px solid {clr}40;">'
        f"{cat}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    details_html = '<div class="section-card">'
    details_html += '<div class="section-card-header">Prediction Details</div>'
    details_html += '<table style="width:100%;font-size:13px;color:#334155;">'
    detail_rows = [
        ("Pollutant", inputs.get("pollutant", "—")),
        ("Date", f"{inputs.get('year','—')}-{inputs.get('month','—'):02d}-{inputs.get('day','—'):02d}" if inputs else "—"),
        ("AQI Lag 1 (Yesterday)", f"{inputs.get('aqi_lag_1', '—')}"),
        ("AQI Lag 7 (Week Ago)", f"{inputs.get('aqi_lag_7', '—')}"),
    ]
    for label, val in detail_rows:
        details_html += (
            f'<tr style="border-bottom:1px solid #F1F5F9;">'
            f'<td style="padding:8px 4px;color:#64748B;font-weight:500;">{label}</td>'
            f'<td style="padding:8px 4px;text-align:right;font-weight:600;">{val}</td>'
            f"</tr>"
        )
    details_html += "</table></div>"
    st.markdown(details_html, unsafe_allow_html=True)

with col_right:
    fig_gauge = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=aqi,
            number=dict(font=dict(size=40, color=clr, family="Inter"), suffix=""),
            gauge=dict(
                axis=dict(range=[0, 500], tickwidth=1, tickcolor="#E2E8F0"),
                bar=dict(color=clr, thickness=0.25),
                bgcolor="#F7F9F7",
                borderwidth=1,
                bordercolor="#E2E8F0",
                steps=[
                    dict(
                        range=[c["min"], c["max"]],
                        color=f"rgba({int(c['color'][1:3], 16)},{int(c['color'][3:5], 16)},{int(c['color'][5:7], 16)},0.12)",
                    )
                    for c in AQI_CATEGORIES
                ],
                threshold=dict(line=dict(color=clr, width=3), thickness=0.8, value=aqi),
            ),
            title=dict(text="AQI Gauge", font=dict(size=14, color="#1E293B", family="Inter")),
        )
    )
    fig_gauge.update_layout(
        height=340,
        margin=dict(l=30, r=30, t=50, b=20),
        paper_bgcolor="#FFFFFF",
        font=dict(family="Inter, sans-serif"),
    )
    st.plotly_chart(fig_gauge, use_container_width=True)

    st.markdown("## AQI Category Scale")
    legend_html = '<div style="display:flex;flex-wrap:wrap;gap:8px;margin-top:4px;">'
    for c in AQI_CATEGORIES:
        active = " font-weight:700;border-width:2px;" if c["name"] == cat else ""
        legend_html += (
            f'<div style="display:inline-flex;align-items:center;gap:6px;'
            f"padding:6px 12px;border-radius:6px;"
            f"background:{c['color']}10;border:1px solid {c['color']}30;"
            f'font-size:12px;color:{c["color"]};{active}">'
            f'{c["name"]} ({c["min"]}–{c["max"]})</div>'
        )
    legend_html += "</div>"
    st.markdown(legend_html, unsafe_allow_html=True)
