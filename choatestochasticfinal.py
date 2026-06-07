#./.venv/bin/python3 -m streamlit run choatestochasticfinal.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.stats import t

st.set_page_config(page_title="SOLOMON: Endowment Risk Suite", layout="wide")
st.markdown("""
    <style>
    .main { background-color: #0f172a; color: white; }
    div[data-testid="stMetricValue"] { color: #38bdf8; }
    .stSlider [data-baseweb="slider"] { background-color: #38bdf8; }
    </style>
    """, unsafe_allow_html=True)

st.title("Institutional Risk & Quant Analyses Dashboard")
st.subheader("Unified Stochastic Engine & Strategic Planning Algorithm")

def run_endowment_sim(v0, years, spend_rate, r, vol, inf, fee, sims, use_tails, rule, fund_percentile, cap_shock_amt=0, cap_shock_yr=0):
    dt = 1/252
    steps = int(years * 252)
    paths = np.zeros((sims, steps))
    paths[:, 0] = v0
    
    prev_spend = np.full(sims, (v0 * spend_rate) / 252) 
    
    for i in range(1, steps):
        if use_tails:
            shocks = t.rvs(df=3, size=sims) * (vol / np.sqrt(252))
        else:
            shocks = np.random.normal(0, vol * np.sqrt(dt), sims)
        
        growth = np.exp((r - fee) * dt + shocks)
        
        if rule == "Yale Hybrid":
            current_spend = (0.8 * prev_spend * (1 + inf*dt)) + (0.2 * (paths[:, i-1] * spend_rate / 252))
        else:
            current_year = i // 252
            inf_adj = (1 + inf) ** current_year
            current_spend = (v0 * spend_rate * inf_adj) / 252
            
        shock_term = 0
        if cap_shock_yr > 0 and i == int(cap_shock_yr * 252):
            shock_term = cap_shock_amt

        draw_this_step = current_spend

        fundraise_this_step = (draw_this_step * (fund_percentile/100))
            
        paths[:, i] = (paths[:, i-1] * growth) - draw_this_step + fundraise_this_step - shock_term
        prev_spend = current_spend 
        
    return paths

with st.sidebar:
    st.image("Choate Rosemary Hall Logo.png", width=200)
    
    st.header("1. Endowment Parameters")
    initial_val = st.number_input("Current Endowment Value ($)", value=300000000, step=10000000)
    years_proj = st.slider("Projection Horizon (Years)", 5, 30, 15)
    rule_type = st.selectbox("Spending Rule", ["Fixed Percentage", "Yale Hybrid"])
    fund_percentile = st.slider("Fundraising Percentage (In Terms of Draw)")
    initial_percentage = st.slider("Initial Spending in Percentage of Endowment")
    
    st.header("2. Market Variables")
    exp_ret = st.slider("Expected Market Return (%)", 0.0, 15.0, 7.5) / 100
    volatility = st.slider("Annual Volatility (60-40 Ratio is 11.4%)", 5.0, 35.0, 15.0) / 100
    inflation = st.slider("Annual Inflation (%)", 0.0, 10.0, 2.5) / 100
    mgt_fee = st.slider("Management/Advisor Fees (%)", 0.0, 2.0, 0.5) / 100
    
    st.header("3. Risk & Shock Settings")
    n_sims = st.selectbox("Simulation Iterations", [1000, 2000, 5000], index=1)
    tail_risk = st.checkbox("Enable Fat-Tail Shocks (Student-t)", value=True)
    
    st.header("4. Project Planning")
    shock_amt = st.number_input("One-time Capital Withdrawal ($)", value=0, step=1000000)
    shock_yr = st.number_input("Year of Withdrawal (0 = None)", value=0, min_value=0, max_value=years_proj)

    st.header("5. Custom Data Import")
    uploaded_file = st.file_uploader("Upload Historical Returns (CSV)", type="csv")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        if 'returns' in df.columns:
            hist_vol = df['returns'].std() * np.sqrt(252)
            st.sidebar.success(f"Calibrated Vol: {hist_vol:.2%}")
            volatility = hist_vol 

with st.spinner('Simulating Market Paths...'):
    data = run_endowment_sim(
    initial_val, years_proj, initial_percentage/100, exp_ret, volatility, 
    inflation, mgt_fee, n_sims, tail_risk, rule_type, fund_percentile, shock_amt, shock_yr
)

final_vals = data[:, -1]
prob_erosion = np.sum(final_vals < initial_val) / n_sims * 100
var_95 = np.percentile(final_vals, 5)
cvar_95 = final_vals[final_vals <= var_95].mean()

col1, col2, col3 = st.columns(3)
col1.metric("Prob. of Principal Erosion", f"{prob_erosion:.1f}%")
col2.metric("95% VaR (Worst Case)", f"${var_95/1e6:.1f}M")
col3.metric("95% CVaR (Expected Tail Loss)", f"${cvar_95/1e6:.1f}M")

st.markdown("### Stochastic Projection Cone")
fig, ax = plt.subplots(figsize=(12, 5), facecolor='#0f172a')
ax.set_facecolor('#0f172a')
time = np.linspace(0, years_proj, data.shape[1])

p95 = np.percentile(data, 95, axis=0)
p75 = np.percentile(data, 75, axis=0)
p50 = np.percentile(data, 50, axis=0)
p25 = np.percentile(data, 25, axis=0)
p5 = np.percentile(data, 5, axis=0)

ax.fill_between(time, p5, p95, color='#38bdf8', alpha=0.1, label='90% Confidence')
ax.fill_between(time, p25, p75, color='#38bdf8', alpha=0.3, label='50% Confidence')
ax.plot(time, p50, color='#38bdf8', linewidth=2, label='Median Projection')
ax.axhline(initial_val, color='#ef4444', linestyle='--', label='Initial Principal')

ax.set_title(f"Endowment Forecast: {rule_type} Rule", color='white', fontsize=14)
ax.set_ylabel("Portfolio Value ($)", color='#94a3b8')
ax.tick_params(colors='#94a3b8')
ax.legend()
plt.grid(alpha=0.1)
st.pyplot(fig)

st.markdown("---")
st.header("Strategic Sensitivity Analysis")

if st.button("Generate Heatmap Analysis"):
    with st.spinner("Analyzing 25,000 potential market regimes..."):

        rates = np.linspace(0.03, 0.06, 5) 
        vols = np.linspace(0.10, 0.25, 5)  
        heatmap_data = np.zeros((5, 5))

        for r_idx, r in enumerate(rates):
            for v_idx, v in enumerate(vols):

                sim_res = run_endowment_sim(
                    initial_val, 10, r, exp_ret, v, 
                    inflation, mgt_fee, 500, tail_risk, rule_type, fund_percentile
                )
                heatmap_data[r_idx, v_idx] = np.sum(sim_res[:, -1] < initial_val) / 500
        
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        sns.heatmap(heatmap_data, annot=True, fmt=".1%", cmap="RdYlGn_r", 
                    xticklabels=[f"{int(x*100)}%" for x in vols], 
                    yticklabels=[f"{x*100:.1f}%" for x in rates], ax=ax2)
        ax2.set_title("Probability of Principal Erosion (Spending vs. Volatility)")
        ax2.set_xlabel("Market Volatility")
        ax2.set_ylabel("Annual Spending Rate")
        st.pyplot(fig2)

st.info("**Disclaimer:** This dashboard is a mathematical simulation for strategic planning. It uses Stochastic Differential Equations and does not guarantee future financial results.")