import streamlit as st
import option_pricing
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="Option Pricer", page_icon="📈", layout="centered")

st.title("📊 European Option Pricer")

st.markdown("This app calculates the price of a European option using one of three models: "
            "**Black-Scholes**, **Monte Carlo**, or **Binomial Tree**.")


col1, col2 = st.columns(2)
with col1:
    S = st.number_input("Spot Price ($S$)", value=100.0)
    K = st.number_input("Strike Price ($K$)", value=100.0)
    T = st.number_input("Time to Maturity ($T$, years)", value=1.0, format="%.2f")

with col2:
    r = st.number_input("Risk-Free Rate ($r$)", value=0.05, format="%.4f")
    sigma = st.number_input("Volatility ($\\sigma$)", value=0.2, format="%.4f")


model_type = st.selectbox("Pricing Model", ["Black-Scholes", "Monte Carlo", "Binomial Tree"])


with st.expander("⚙️ Advanced Settings"):
    if model_type == "Monte Carlo":
        N = st.slider("Number of Simulations", 1000, 100000, 10000, step=1000)
    elif model_type == "Binomial Tree":
        N = st.slider("Number of Time Steps", 10, 1000, 500, step=10)

call_option = option_pricing.EuropeanOption(S, K, T, r, sigma, option_pricing.OptionType.Call)
put_option = option_pricing.EuropeanOption(S, K, T, r, sigma, option_pricing.OptionType.Put)


if model_type == "Black-Scholes":
    model = option_pricing.BlackScholesModel()
elif model_type == "Monte Carlo":
    model = option_pricing.MonteCarloModel(N)
elif model_type == "Binomial Tree":
    model = option_pricing.BinomialModel(N)

call_price = model.price(call_option)
put_price = model.price(put_option)


st.markdown("### 💰 Option Prices")

col1, col2 = st.columns(2)

with col1:
    st.markdown(
        f"""
        <div style="background-color:#dbeeff;padding:20px;border-radius:10px;
                    border-left:6px solid #1f77b4;color:#000000;">
            <h4 style="margin:0;font-weight:bold;">Call Option</h4>
            <h2 style="margin:0;font-weight:bold;">${call_price:.4f}</h2>
        </div>
        """,
        unsafe_allow_html=True
    )

with col2:
    st.markdown(
        f"""
        <div style="background-color:#ffe5e5;padding:20px;border-radius:10px;
                    border-left:6px solid #d62728;color:#000000;">
            <h4 style="margin:0;font-weight:bold;">Put Option</h4>
            <h2 style="margin:0;font-weight:bold;">${put_price:.4f}</h2>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")
st.markdown("### Option Price Surface")

plot_type = st.selectbox("Select Option Type for Surface", ["Call", "Put"], key="surface_option_type")


st.markdown("### Surface Axes: Strike vs Volatility")

strike_min = st.number_input("Lower Strike Bound", value=K * 0.5)
strike_max = st.number_input("Upper Strike Bound", value=K * 1.5)
vol_min = st.number_input("Lower Volatility Bound", value=0.05, min_value=0.001)
vol_max = st.number_input("Upper Volatility Bound", value=0.8)
T_fixed = st.number_input("Fixed Maturity (years)", value=T)

steps = st.slider("Grid Resolution", 10, 100, 30, step=5)

K_vals = np.linspace(strike_min, strike_max, steps)
V_vals = np.linspace(vol_min, vol_max, steps)
Z = np.zeros((len(V_vals), len(K_vals)))

option_type_enum = option_pricing.OptionType.Call if plot_type == "Call" else option_pricing.OptionType.Put
purchase_price = st.number_input(
    "Option Purchase Price (What You Paid)", value=2.50, format="%.4f"
)
for i, sigma_val in enumerate(V_vals):
    for j, K_val in enumerate(K_vals):
        option = option_pricing.EuropeanOption(S, K_val, T_fixed, r, sigma_val, option_type_enum)

        if model_type == "Black-Scholes":
            model = option_pricing.BlackScholesModel()
        elif model_type == "Monte Carlo":
            model = option_pricing.MonteCarloModel(N)
        else:
            model = option_pricing.BinomialModel(N)

        try:
            Z[i, j] = model.price(option) - purchase_price
        except:
            Z[i, j] = np.nan



Z = np.nan_to_num(Z, nan=0.0)

cmin = np.min(Z)
cmax = np.max(Z)
if np.isclose(cmin, cmax):
    cmin -= 1e-4
    cmax += 1e-4

surface = go.Surface(
    z=Z,
    x=K_vals,
    y=V_vals,
    colorscale="RdYlGn",
    cmin=cmin,
    cmax=cmax,
    hovertemplate=
        "Strike: %{x:.2f}<br>" +
        "Volatility: %{y:.2%}<br>" +
        "P&L: %{z:.2f}<extra></extra>"
)

fig = go.Figure(data=[surface])
fig.update_layout(
    title=f"{plot_type} P&L Surface vs Strike & Volatility ({model_type})",
    scene=dict(
        xaxis_title="Strike Price (K)",
        yaxis_title="Volatility (σ)",
        zaxis_title="Profit / Loss ($)"
    ),
    height=600
)

st.plotly_chart(fig, use_container_width=True)
