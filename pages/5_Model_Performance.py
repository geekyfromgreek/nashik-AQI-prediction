import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from utils import (
    MODEL_RESULTS,
    PLOTLY_COLORS,
    inject_css,
    plotly_layout,
    render_page_header,
    render_prediction_sidebar,
)

st.set_page_config(
    page_title="Model Performance — AQI Analysis",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_css()
render_prediction_sidebar()

render_page_header(
    "Model Performance",
    "Evaluation metrics across 9 regression models. The ANN was selected as the final model.",
)

ann = MODEL_RESULTS[MODEL_RESULTS["Model"] == "ANN"].iloc[0]
st.markdown(
    '<div class="model-highlight">'
    "<h4>Selected Model — Artificial Neural Network (ANN)</h4>"
    f"<p>MAE: <strong>{ann['MAE']}</strong> &nbsp;|&nbsp; "
    f"RMSE: <strong>{ann['RMSE']}</strong> &nbsp;|&nbsp; "
    f"R² Score: <strong>{ann['R² Score']}</strong> &nbsp;|&nbsp; "
    "Best overall performance among all evaluated models.</p>"
    "</div>",
    unsafe_allow_html=True,
)

st.markdown("## Evaluation Metrics — All Models")

def highlight_ann(row):
    if row["Model"] == "ANN":
        return ["background-color: #E8F5E9; font-weight: 600"] * len(row)
    return [""] * len(row)

styled_df = MODEL_RESULTS.style.apply(highlight_ann, axis=1).format(
    {"MAE": "{:.4f}", "RMSE": "{:.4f}", "R² Score": "{:.4f}"}
)
st.dataframe(styled_df, use_container_width=True, hide_index=True, height=380)

st.markdown("## Visual Comparison")

tab1, tab2, tab3, tab4 = st.tabs(["MAE", "RMSE", "R² Score", "Combined"])

results_sorted = MODEL_RESULTS.sort_values("MAE")

with tab1:
    colors = ["#2E7D32" if m == "ANN" else "#81C784" for m in results_sorted["Model"]]
    fig_mae = go.Figure()
    fig_mae.add_trace(
        go.Bar(
            x=results_sorted["Model"],
            y=results_sorted["MAE"],
            marker_color=colors,
            text=results_sorted["MAE"].round(2),
            textposition="outside",
            hovertemplate="%{x}<br>MAE: %{y:.4f}<extra></extra>",
        )
    )
    fig_mae.update_layout(
        **plotly_layout("Mean Absolute Error (MAE) — Lower is Better", height=420),
        yaxis_title="MAE",
    )
    st.plotly_chart(fig_mae, use_container_width=True)

with tab2:
    results_rmse = MODEL_RESULTS.sort_values("RMSE")
    colors_rmse = ["#2E7D32" if m == "ANN" else "#81C784" for m in results_rmse["Model"]]
    fig_rmse = go.Figure()
    fig_rmse.add_trace(
        go.Bar(
            x=results_rmse["Model"],
            y=results_rmse["RMSE"],
            marker_color=colors_rmse,
            text=results_rmse["RMSE"].round(2),
            textposition="outside",
            hovertemplate="%{x}<br>RMSE: %{y:.4f}<extra></extra>",
        )
    )
    fig_rmse.update_layout(
        **plotly_layout("Root Mean Squared Error (RMSE) — Lower is Better", height=420),
        yaxis_title="RMSE",
    )
    st.plotly_chart(fig_rmse, use_container_width=True)

with tab3:
    results_r2 = MODEL_RESULTS.sort_values("R² Score", ascending=False)
    colors_r2 = ["#2E7D32" if m == "ANN" else "#81C784" for m in results_r2["Model"]]
    fig_r2 = go.Figure()
    fig_r2.add_trace(
        go.Bar(
            x=results_r2["Model"],
            y=results_r2["R² Score"],
            marker_color=colors_r2,
            text=results_r2["R² Score"].round(4),
            textposition="outside",
            hovertemplate="%{x}<br>R²: %{y:.4f}<extra></extra>",
        )
    )
    fig_r2.update_layout(
        **plotly_layout("R² Score — Higher is Better", height=420),
        yaxis_title="R² Score",
        yaxis_range=[0, 1],
    )
    st.plotly_chart(fig_r2, use_container_width=True)

with tab4:
    fig_comb = go.Figure()
    fig_comb.add_trace(
        go.Bar(
            name="MAE",
            x=results_sorted["Model"],
            y=results_sorted["MAE"],
            marker_color="#2E7D32",
        )
    )
    fig_comb.add_trace(
        go.Bar(
            name="RMSE",
            x=results_sorted["Model"],
            y=results_sorted["RMSE"],
            marker_color="#66BB6A",
        )
    )
    fig_comb.add_trace(
        go.Scatter(
            name="R² Score (×30)",
            x=results_sorted["Model"],
            y=results_sorted["R² Score"] * 30,
            mode="lines+markers",
            line=dict(color="#EF6C00", width=2.5),
            marker=dict(size=8),
            yaxis="y2",
        )
    )
    fig_comb.update_layout(
        **plotly_layout(
            "Combined Model Comparison",
            height=460,
            legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.9)"),
        ),
        barmode="group",
        yaxis_title="Error Metric",
        yaxis2=dict(
            title="R² Score (scaled ×30)",
            overlaying="y",
            side="right",
            gridcolor="#F1F5F9",
            range=[0, 30],
        ),
    )
    st.plotly_chart(fig_comb, use_container_width=True)

st.markdown("## ANN Architecture")

c1, c2 = st.columns([1.2, 1])

with c1:
    st.markdown(
        '<div class="section-card">'
        '<div class="section-card-header">Network Architecture</div>'
        '<table style="width:100%;font-size:13px;color:#334155;">'
        '<tr style="border-bottom:1px solid #F1F5F9;">'
        '<td style="padding:8px 4px;color:#64748B;font-weight:500;">Input Layer</td>'
        '<td style="padding:8px 4px;text-align:right;font-weight:600;">19 features</td></tr>'
        '<tr style="border-bottom:1px solid #F1F5F9;">'
        '<td style="padding:8px 4px;color:#64748B;font-weight:500;">Hidden Layer 1</td>'
        '<td style="padding:8px 4px;text-align:right;font-weight:600;">Dense(128, ReLU) + Dropout(0.2)</td></tr>'
        '<tr style="border-bottom:1px solid #F1F5F9;">'
        '<td style="padding:8px 4px;color:#64748B;font-weight:500;">Hidden Layer 2</td>'
        '<td style="padding:8px 4px;text-align:right;font-weight:600;">Dense(64, ReLU) + Dropout(0.2)</td></tr>'
        '<tr style="border-bottom:1px solid #F1F5F9;">'
        '<td style="padding:8px 4px;color:#64748B;font-weight:500;">Output Layer</td>'
        '<td style="padding:8px 4px;text-align:right;font-weight:600;">Dense(1) — Linear</td></tr>'
        '<tr style="border-bottom:1px solid #F1F5F9;">'
        '<td style="padding:8px 4px;color:#64748B;font-weight:500;">Optimizer</td>'
        '<td style="padding:8px 4px;text-align:right;font-weight:600;">Adam</td></tr>'
        '<tr style="border-bottom:1px solid #F1F5F9;">'
        '<td style="padding:8px 4px;color:#64748B;font-weight:500;">Loss Function</td>'
        '<td style="padding:8px 4px;text-align:right;font-weight:600;">Mean Squared Error (MSE)</td></tr>'
        '<tr>'
        '<td style="padding:8px 4px;color:#64748B;font-weight:500;">Training</td>'
        '<td style="padding:8px 4px;text-align:right;font-weight:600;">100 epochs, batch 32, 20% val split</td></tr>'
        "</table></div>",
        unsafe_allow_html=True,
    )

with c2:
    st.markdown(
        '<div class="section-card">'
        '<div class="section-card-header">Why ANN?</div>'
        "<ul style='color:#64748B;font-size:13px;line-height:2;padding-left:18px;'>"
        "<li>Lowest MAE (18.82) among all models</li>"
        "<li>Lowest RMSE (28.39) — smallest prediction error</li>"
        "<li>Highest R² Score (0.7662) — explains 76.6% variance</li>"
        "<li>Outperforms tuned ensemble methods (RF, XGBoost)</li>"
        "<li>Dropout regularisation prevents overfitting</li>"
        "<li>Captures non-linear pollutant-AQI relationships</li>"
        "</ul></div>",
        unsafe_allow_html=True,
    )
