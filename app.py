# app.py
# Incremental Intelligence — Prototype (production-polished prototype)
# Dependencies: streamlit, pandas, numpy, plotly
# Author: Shubham Jain (prototype)
# Date: 2026-03-04

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import math
import json
from datetime import date, timedelta

st.set_page_config(page_title="Incremental Intelligence — Prototype", layout="wide")

# ------------------------
# Helpers & Mock Data
# ------------------------

def pretty_currency(v):
    try:
        return f"${int(v):,}"
    except:
        return str(v)

def revenue_from_spend(a, b, spend):
    # power-law response: Revenue ≈ a * spend^b
    try:
        return a * (spend ** b)
    except Exception:
        return 0.0

def marginal_revenue(a, b, spend):
    # derivative: a * b * spend^(b-1)
    try:
        return a * b * (spend ** (b - 1)) if spend > 0 else 0.0
    except Exception:
        return 0.0

def avg_roas_from_attrib(attributed_revenue, spend):
    try:
        return attributed_revenue / spend if spend > 0 else 0.0
    except Exception:
        return 0.0

def marginal_roas_from_model(a, b, spend):
    # marginal revenue per $1 spent
    return marginal_revenue(a, b, spend)

def saturation_pct_from_ratio(marginal_roas, avg_roas, kappa=4.0):
    # create a realistic saturation percent from ratio marginal/avg using a sigmoid-like transform
    # raw_ratio close to 1 -> saturation ~ 50
    # raw_ratio >>1 -> saturation low
    # raw_ratio <<1 -> saturation high
    eps = 1e-9
    raw = marginal_roas / (avg_roas + eps)
    # convert to scale where 1 => 0.5 in sigmoid input
    x = raw
    # apply logistic transform, then invert to get saturation
    # sigmoid = 1 / (1 + exp(-k*(x - 1))) -- higher x -> sigmoid -> closer to 1
    try:
        sigmoid = 1.0 / (1.0 + math.exp(-kappa * (x - 1.0)))
    except OverflowError:
        sigmoid = 0.0 if x < 1 else 1.0
    sat = (1.0 - sigmoid) * 100.0
    # clamp
    if sat < 0:
        sat = 0.0
    if sat > 100:
        sat = 100.0
    # Round to sensible value
    return round(sat, 1)

def confidence_score(experiment_present, mmm_present, attribution_variance):
    exp_w = 0.55 if experiment_present else 0.0
    mmm_w = 0.25 if mmm_present else 0.0
    att_w = max(0, 1 - min(attribution_variance / 2.0, 1)) * 0.20
    score = int((exp_w + mmm_w + att_w) * 100)
    return min(max(score, 0), 100)

# Simple MDE approximation for proportion tests (two-sided, alpha=0.05, power=0.8)
def detectable_lift_proportion(p0, n):
    if n <= 1:
        return 1.0
    z_alpha = 1.96  # two-sided 5%
    z_beta = 0.84   # power 80%
    mde = (z_alpha + z_beta) * math.sqrt(2 * p0 * (1 - p0) / n)
    return mde

def estimate_n_from_spend(spend, avg_order_value, conv_rate):
    # estimate number of conversions during period from spend
    if avg_order_value <= 0 or conv_rate <= 0:
        return 1
    est_orders = (spend / avg_order_value) * conv_rate
    return max(1, int(est_orders))

def weighted_portfolio_metrics(portfolio_df):
    total_spend = float(portfolio_df["spend"].sum()) if not portfolio_df.empty else 0.0
    if total_spend == 0:
        return {"total_spend":0,"w_avg_roas":0,"w_marginal":0,"w_saturation":0,"w_conf":0}
    w_avg_roas = (portfolio_df["avg_roas"] * portfolio_df["spend"]).sum() / total_spend
    w_marginal = (portfolio_df["marginal_roas"] * portfolio_df["spend"]).sum() / total_spend
    w_saturation = (portfolio_df["saturation"] * portfolio_df["spend"]).sum() / total_spend
    w_conf = (portfolio_df["confidence"] * portfolio_df["spend"]).sum() / total_spend
    return {"total_spend": int(total_spend), "w_avg_roas": round(w_avg_roas,2), "w_marginal": round(w_marginal,2), "w_saturation": round(w_saturation,1), "w_conf": round(w_conf,1)}

def recommend_test_designs(dmas_df, campaign_spend, avg_order_value, conv_rate):
    df = dmas_df.copy()
    df["est_conv"] = ((campaign_spend * df["spend_share"]) / avg_order_value) * conv_rate
    df = df.sort_values("est_conv", ascending=False).reset_index(drop=True)
    candidates = []
    for size in [3,5,8]:
        chosen = df.head(size)
        n = max(1, int(chosen["est_conv"].sum()))
        mde = detectable_lift_proportion(conv_rate, n)
        candidates.append({
            "name": f"Top {size} DMAs",
            "size": size,
            "dmas": chosen["dma"].tolist(),
            "n": n,
            "mde_pct": round(mde*100,2),
            "est_conv_total": int(chosen["est_conv"].sum())
        })
    return candidates

# ------------------------
# Mock portfolio & DMAs (starting dataset)
# ------------------------
default_campaigns = [
    {"campaign":"Brand US", "spend":2000000, "a":50, "b":0.30, "avg_roas":8.5, "has_experiment":False, "mmm_share":0.12, "att_variance":1.2, "type":"brand"},
    {"campaign":"NonBrand US", "spend":3000000, "a":20, "b":0.60, "avg_roas":4.2, "has_experiment":True, "mmm_share":0.20, "att_variance":0.6, "type":"nonbrand"},
    {"campaign":"Geo Expansion", "spend":800000, "a":18, "b":0.55, "avg_roas":3.5, "has_experiment":False, "mmm_share":0.08, "att_variance":0.9, "type":"nonbrand"},
    {"campaign":"Branded Retail", "spend":500000, "a":40, "b":0.35, "avg_roas":7.0, "has_experiment":False, "mmm_share":0.10, "att_variance":1.4, "type":"brand"}
]

sample_dmas = pd.DataFrame([
    {"dma":"NY","spend_share":0.18,"pop":8000000},
    {"dma":"LA","spend_share":0.14,"pop":4000000},
    {"dma":"Chicago","spend_share":0.10,"pop":2700000},
    {"dma":"Dallas","spend_share":0.06,"pop":1300000},
    {"dma":"Atlanta","spend_share":0.05,"pop":1100000},
    {"dma":"Seattle","spend_share":0.04,"pop":750000},
    {"dma":"Denver","spend_share":0.03,"pop":700000},
    {"dma":"Phoenix","spend_share":0.05,"pop":1600000},
    {"dma":"Miami","spend_share":0.03,"pop":1500000},
    {"dma":"Minneapolis","spend_share":0.02,"pop":1100000},
])

@st.cache_data
def build_portfolio_df(df):
    rows = []
    for r in df.to_dict(orient="records"):
        spend = r["spend"]
        a = r.get("a", 1.0)
        b = r.get("b", 0.5)
        # compute marginal roas from model and avg_roas (if any)
        marg_rev = marginal_revenue(a, b, spend)
        marginal_roas = marg_rev  # marginal revenue per $1
        avg_roas = r.get("avg_roas", 0.0)
        sat = saturation_pct_from_ratio(marginal_roas, avg_roas)
        conf = confidence_score(r.get("has_experiment", False), r.get("mmm_share", 0.0)>0.0, r.get("att_variance", 1.0))
        rows.append({
            "campaign": r["campaign"],
            "spend": int(spend),
            "avg_roas": round(avg_roas,2),
            "marginal_roas": round(marginal_roas,2),
            "elasticity_b": r.get("b", 0.5),
            "saturation": sat,
            "confidence": conf,
            "has_experiment": r.get("has_experiment", False),
            "mmm_share": r.get("mmm_share", 0.0),
            "att_variance": r.get("att_variance", 1.0),
            "type": r.get("type", "nonbrand")
        })
    return pd.DataFrame(rows)

# ------------------------
# AI Summary & Ask-AI helpers
# ------------------------
def generate_portfolio_summary(portfolio_df, combined_df, rec_df, brand_cap):
    lines = []
    if portfolio_df.empty:
        return ["No campaigns selected."]
    top_spends = portfolio_df.sort_values("spend", ascending=False).head(3)
    tops = ", ".join(top_spends["campaign"].tolist())
    lines.append(f"Top spend campaigns: {tops} (range {pretty_currency(top_spends['spend'].min())} – {pretty_currency(top_spends['spend'].max())}).")
    high_sat = portfolio_df[portfolio_df["saturation"] >= 75]
    if not high_sat.empty:
        lines.append("High saturation: " + ", ".join(high_sat["campaign"].tolist()) + ". Consider reducing spend or running validation.")
    low_mroas = portfolio_df[portfolio_df["marginal_roas"] < (portfolio_df["avg_roas"] * 0.6)]
    if not low_mroas.empty:
        lines.append("Campaigns with low marginal ROAS vs avg: " + ", ".join(low_mroas["campaign"].tolist()) + ".")
    low_conf = combined_df[combined_df["final_confidence"] < 60] if not combined_df.empty else pd.DataFrame()
    if not low_conf.empty:
        lines.append("Low confidence in measurement for: " + ", ".join(low_conf["campaign"].tolist()) + ". Synthetic or controlled tests recommended.")
    inc = rec_df[rec_df["action"]=="Increase spend"] if not rec_df.empty else pd.DataFrame()
    dec = rec_df[rec_df["action"]=="Reduce spend"] if not rec_df.empty else pd.DataFrame()
    if not inc.empty:
        lines.append("Suggested increases: " + ", ".join(inc["campaign"].tolist()) + ".")
    if not dec.empty:
        lines.append("Suggested reductions: " + ", ".join(dec["campaign"].tolist()) + ".")
    total_spend = float(portfolio_df["spend"].sum())
    brand_spend = float(portfolio_df[portfolio_df["type"]=="brand"]["spend"].sum())
    brand_pct = round(100 * brand_spend / total_spend, 1) if total_spend > 0 else 0.0
    if brand_pct > brand_cap:
        lines.append(f"Governance alert: Brand spend is {brand_pct}% (> cap {brand_cap}%).")
    lines.append(f"Portfolio weighted confidence: {round(portfolio_df['confidence'].mean(),1)}%.")
    return lines

def ask_ai_question(q, portfolio_df, combined_df, rec_df, dmas_df=None):
    ql = q.lower().strip()
    if len(ql) == 0:
        return "Please enter a short question like 'best campaign', 'low confidence', 'mde', or 'governance'."
    if "best campaign" in ql or "which campaign" in ql or "top campaign" in ql:
        if portfolio_df.empty:
            return "No campaigns selected."
        best = portfolio_df.sort_values("marginal_roas", ascending=False).iloc[0]
        return f"By marginal ROAS, **{best['campaign']}** looks strongest (marginal ROAS ≈ {best['marginal_roas']}). Consider increasing spend there if confidence is sufficient."
    if "low confidence" in ql:
        low_conf = combined_df[combined_df["final_confidence"] < 60] if not combined_df.empty else pd.DataFrame()
        return "Low confidence campaigns: " + (", ".join(low_conf["campaign"].tolist()) if not low_conf.empty else "None detected.")
    if "governance" in ql or "brand cap" in ql:
        total_spend = float(portfolio_df["spend"].sum()) if not portfolio_df.empty else 0
        brand_spend = float(portfolio_df[portfolio_df["type"]=="brand"]["spend"].sum()) if not portfolio_df.empty else 0
        brand_pct = round(100 * brand_spend / total_spend,1) if total_spend > 0 else 0.0
        return f"Current brand spend is {brand_pct}% of portfolio spend."
    if "mde" in ql or "detectable" in ql:
        out = []
        for _, r in portfolio_df.iterrows():
            n = max(1, estimate_n_from_spend(r["spend"], 50.0, 0.01))
            mde = detectable_lift_proportion(0.01, n)
            out.append(f"{r['campaign']}: approx MDE ±{round(mde*100,2)}%")
        return "Approx MDE per campaign (assumptions AOV=50, conv=1%):\n" + "\n".join(out)
    # optional LLM path if OPENAI_API_KEY is set in Streamlit secrets
    try:
        key = None
        if "OPENAI_API_KEY" in st.secrets:
            key = st.secrets["OPENAI_API_KEY"]
        else:
            import os
            key = os.environ.get("OPENAI_API_KEY")
    except Exception:
        key = None
    if key:
        try:
            import openai
            openai.api_key = key
            prompt = f"You are a helpful analytics assistant. Question: {q}\nContext: portfolio summary.\n{portfolio_df.to_csv(index=False)}"
            resp = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[{"role":"system","content":"You are an analytics assistant."},
                          {"role":"user","content":prompt}],
                max_tokens=300,
                temperature=0.2
            )
            text = resp.choices[0].message.content.strip()
            return text
        except Exception as e:
            return f"LLM call failed (check OPENAI_API_KEY): {e}"
    return ("I don't have a full answer from rules. Try: 'best campaign', 'low confidence', 'mde', or add an OpenAI key to Streamlit secrets for full LLM responses.")

# ------------------------
# Session state init
# ------------------------
if "df_mod" not in st.session_state:
    st.session_state["df_mod"] = pd.DataFrame(default_campaigns)
if "data_refresh_ts" not in st.session_state:
    st.session_state["data_refresh_ts"] = None
# default chosen_dmas safe key
if "chosen_dmas" not in st.session_state:
    st.session_state["chosen_dmas"] = None

# ------------------------
# Sidebar: Governance, Scope, Ask-AI, First-party toggle
# ------------------------
st.sidebar.header("Governance & Settings")
brand_cap = st.sidebar.slider("Max Brand Spend % of total (governance)", 0, 100, 40)
cpa_cap = st.sidebar.number_input("CPA cap (mock)", min_value=0.0, value=15.0)
confidence_threshold = st.sidebar.slider("Confidence threshold (%) for auto-action", 0, 100, 60)

st.sidebar.markdown("---")
st.sidebar.header("Scope & Data")
default_start = pd.to_datetime(date.today() - timedelta(days=90))
default_end = pd.to_datetime(date.today())
date_range = st.sidebar.date_input("Date range", value=(default_start.date(), default_end.date()))
all_campaigns = st.session_state["df_mod"]["campaign"].tolist()
selected_campaigns = st.sidebar.multiselect("Campaigns (scope)", options=all_campaigns, default=all_campaigns)
auto_pop = st.sidebar.checkbox("Auto-populate measurement fields if integrations available (simulated)", value=True)
if st.sidebar.button("Refresh / Simulate Data"):
    st.session_state["data_refresh_ts"] = pd.Timestamp.now()
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.header("First-party data")
use_cdp = st.sidebar.checkbox("Use Adobe CDP signals (simulated)", value=True)
st.sidebar.caption("When enabled, displays first-party match rates and attributed share (demo).")

st.sidebar.markdown("---")
st.sidebar.header("Ask the Assistant")
ask_q = st.sidebar.text_input("Ask (e.g., 'best campaign', 'low confidence', 'mde')", key="ask_q")
if st.sidebar.button("Ask AI"):
    ai_ans = ask_ai_question(ask_q, build_portfolio_df(st.session_state["df_mod"]), pd.DataFrame(), pd.DataFrame(), sample_dmas)
    st.sidebar.markdown("**Answer:**")
    st.sidebar.write(ai_ans)
    st.sidebar.caption("Add OPENAI_API_KEY to Streamlit secrets for fuller LLM responses.")

st.sidebar.markdown("---")
st.sidebar.caption("Prototype by Shubham Jain — shubhamjain.1142@gmail.com")

# ------------------------
# Build & scale portfolio by date range (simulation)
# ------------------------
start_date = pd.to_datetime(date_range[0])
end_date = pd.to_datetime(date_range[1])
period_days = max(1, (end_date - start_date).days)
scale = period_days / 28.0

df_base = st.session_state["df_mod"].copy()
df_scaled = df_base.copy()
df_scaled["spend"] = (df_scaled["spend"] * scale).astype(int)

portfolio_all = build_portfolio_df(df_scaled)
portfolio = portfolio_all[portfolio_all["campaign"].isin(selected_campaigns)].reset_index(drop=True)
agg = weighted_portfolio_metrics(portfolio)

# ------------------------
# Main UI
# ------------------------
st.title("Incremental Intelligence — Prototype")
st.markdown("A prototype for elasticity, triangulation, recommendations, and experimentation. Data is simulated unless integrations are connected. Spend inputs are simulation-only and do not retrain models.")

tab1, tab2, tab3, tab4 = st.tabs(["Portfolio overview", "Measurement triangulation", "Recommendations", "Experimentation Studio"])

# ------------------------
# Tab 1: Portfolio overview & quick health check
# ------------------------
with tab1:
    st.header("Portfolio overview & quick health check")
    st.markdown(f"Date window: **{start_date.date()}** → **{end_date.date()}** — values shown are for the selected date window (simulation).")

    cols = st.columns(3)
    cols[0].metric("Portfolio spend (selected range)", pretty_currency(agg["total_spend"]))
    cols[1].metric("Weighted Avg ROAS", agg["w_avg_roas"], help="Avg ROAS = attributed revenue / spend")
    cols[2].metric("Weighted Marginal ROAS", agg["w_marginal"], help="Marginal ROAS ≈ expected revenue for an extra $1 spent (model-based)")

    st.markdown("**Portfolio health**")
    left, right = st.columns([2,1])
    with left:
        with st.expander("View portfolio table (detailed)"):
            st.dataframe(portfolio[["campaign","type","spend","avg_roas","marginal_roas","elasticity_b","saturation","confidence"]].sort_values("spend", ascending=False), height=360)
    with right:
        fig = px.bar(portfolio, x="campaign", y=["avg_roas","marginal_roas"], barmode="group", title="Avg ROAS vs Marginal ROAS")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("**Saturation (per campaign)**")
        sat_fig = px.imshow([portfolio["saturation"].tolist()], labels=dict(x="Campaign", color="Saturation %"), x=portfolio["campaign"])
        st.plotly_chart(sat_fig, use_container_width=True)

    st.markdown("**Edit baseline (monthly) spend — simulation only**")
    # allow editing baseline spends
    for i, row in enumerate(df_base.to_dict(orient="records")):
        s = st.number_input(f"{row['campaign']} baseline monthly spend (simulation)", value=int(row["spend"]), step=50000, key=f"base_spend_{i}")
    if st.button("Apply baseline spend changes"):
        df_mod = st.session_state["df_mod"].copy()
        for i in range(len(df_mod)):
            df_mod.at[i, "spend"] = int(st.session_state.get(f"base_spend_{i}", df_mod.at[i,"spend"]))
        st.session_state["df_mod"] = df_mod
        st.rerun()

    # AI summary
    st.subheader("Automated summary")
    with st.expander("View AI-generated summary"):
        # build combined_df & rec_df placeholders for better summary (triangulation & rec will populate later)
        combined_df = pd.DataFrame()  # will be replaced in triangulation section after user inputs
        rec_df = pd.DataFrame()
        summary_lines = generate_portfolio_summary(portfolio, combined_df, rec_df, brand_cap)
        for s in summary_lines:
            st.write("- " + s)
        st.caption("This summary is heuristics-based from the prototype data. Optionally enable an OpenAI key in Streamlit secrets for full LLM answers.")

# ------------------------
# Tab 2: Measurement triangulation
# ------------------------
with tab2:
    st.header("Measurement triangulation — reconcile attributed, experiments, and MMM")
    st.markdown("Provide or override values from different measurement sources.")

    methods = []
    # We'll populate inputs per campaign; use safe session_state defaults
    for i, r in enumerate(portfolio.to_dict(orient="records")):
        st.markdown(f"**{r['campaign']}**")
        col1, col2, col3, col4 = st.columns([2,2,2,1])
        default_att = float(round(r["spend"] * (r["avg_roas"] * 0.05), 0)) if r["avg_roas"]>0 else 0.0
        default_exp = float(round(r["spend"] * (0.01 if r["has_experiment"] else 0.02), 0))
        default_mmm = float(round(r["spend"] * (r["mmm_share"] * 0.04 + 0.01), 0))

        st.session_state.setdefault(f"att_{i}", default_att if auto_pop else 0.0)
        st.session_state.setdefault(f"exp_{i}", default_exp if auto_pop else 0.0)
        st.session_state.setdefault(f"mmm_{i}", default_mmm if auto_pop else 0.0)
        st.session_state.setdefault(f"conf_{i}", 0)

        with col1:
            att_val = st.number_input(f"Attributed revenue ($) - {r['campaign']}", value=float(st.session_state[f"att_{i}"]), key=f"att_{i}")
        with col2:
            exp_val = st.number_input(f"Experiment lift ($) - {r['campaign']}", value=float(st.session_state[f"exp_{i}"]), key=f"exp_{i}")
        with col3:
            mmm_val = st.number_input(f"MMM-modeled incremental ($) - {r['campaign']}", value=float(st.session_state[f"mmm_{i}"]), key=f"mmm_{i}")
        with col4:
            conf_val = st.slider(f"Confidence override % (0 = auto) - {r['campaign']}", 0, 100, int(st.session_state[f"conf_{i}"]), key=f"conf_{i}")

        methods.append({
            "campaign": r["campaign"],
            "attributed": float(att_val),
            "experiment_lift": float(exp_val),
            "mmm": float(mmm_val),
            "conf_override": int(conf_val),
            "auto_conf": r["confidence"],
            "spend": r["spend"],
            "avg_roas": r["avg_roas"],
            "marginal_roas": r["marginal_roas"]
        })

    methods_df = pd.DataFrame(methods)
    st.markdown("**Triangulation inputs (current)**")
    with st.expander("View triangulation inputs table"):
        st.dataframe(methods_df, height=260)

    # Combine and compute triangulated iROAS
    combined = []
    for _, row in methods_df.iterrows():
        final_conf = int(row["conf_override"]) if row["conf_override"] > 0 else int(row["auto_conf"])
        # compute source-level iROAS estimates; use spend as denominator if available, else small epsilon
        eps = 1e-9
        att_iroas = (row["attributed"] / (row["spend"] + eps)) if row["spend"] > 0 else 0.0
        exp_iroas = (row["experiment_lift"] / (row["spend"] * 0.1 + eps)) if row["experiment_lift"]>0 else 0.0
        mmm_iroas = (row["mmm"] / (row["spend"] * 0.2 + eps)) if row["mmm"]>0 else 0.0
        # weigh by confidence heuristics
        w_exp = 0.6 if row["experiment_lift"] > 0 and final_conf >= 60 else 0.25 if row["experiment_lift"]>0 else 0.0
        w_att = 0.3 if row["attributed"] > 0 else 0.15
        w_mmm = 0.4 if row["mmm"] > 0 and row["auto_conf"]>40 else 0.15 if row["mmm"]>0 else 0.0
        # normalize weights
        weights = np.array([w_att, w_exp, w_mmm], dtype=float)
        if weights.sum() == 0:
            norm = np.array([1/3,1/3,1/3])
        else:
            norm = weights / weights.sum()
        tri_iroas = norm[0]*att_iroas + norm[1]*exp_iroas + norm[2]*mmm_iroas
        combined.append({
            "campaign": row["campaign"],
            "attributed": row["attributed"],
            "experiment_lift": row["experiment_lift"],
            "mmm": row["mmm"],
            "final_confidence": final_conf,
            "att_iroas": round(att_iroas,4),
            "exp_iroas": round(exp_iroas,4),
            "mmm_iroas": round(mmm_iroas,4),
            "triangulated_iROAS": round(tri_iroas,4),
            "weights": {"att": round(norm[0],2), "exp": round(norm[1],2), "mmm": round(norm[2],2)}
        })

    combined_df = pd.DataFrame(combined)
    st.markdown("**Triangulated results** (triangulated iROAS blends data sources based on confidence)")
    with st.expander("View triangulated table and breakdown"):
        if not combined_df.empty:
            st.dataframe(combined_df[["campaign","attributed","experiment_lift","mmm","final_confidence","triangulated_iROAS","att_iroas","exp_iroas","mmm_iroas","weights"]], height=300)
        else:
            st.write("No triangulation data yet.")

    # CTA: if any campaign low confidence, offer synthetic causal quick run
    low_conf_campaigns = combined_df[combined_df["final_confidence"] < confidence_threshold]["campaign"].tolist() if not combined_df.empty else []
    if low_conf_campaigns:
        st.info(f"Low confidence detected for: {', '.join(low_conf_campaigns)} (threshold {confidence_threshold}%).")
        if st.button("Run Synthetic Causal Analysis for low-confidence campaigns"):
            st.markdown("**Synthetic results (quick simulation)**")
            for c in low_conf_campaigns:
                port_row = portfolio[portfolio["campaign"]==c].iloc[0]
                base_effect = port_row["marginal_roas"] * 0.02
                noise = np.random.normal(0, 0.4)
                synthetic_lift = round(base_effect + noise, 2)
                synthetic_conf = int(max(30, min(85, 60 + np.random.randint(-10,20))))
                st.write(f"Campaign: **{c}** — Synthetic lift: **{synthetic_lift}%**, Confidence: **{synthetic_conf}%**")
            st.warning("Synthetic results are directional. Consider controlled validation.")

# ------------------------
# Tab 3: Recommendation Engine (reallocate + portfolio scale)
# ------------------------
with tab3:
    st.header("Recommendation Engine")
    st.markdown("Run reallocation or portfolio-scale simulations. Choose allocation strategy and review projected revenue & contribution impact.")

    # recompute portfolio (use full df_mod for up-to-date numbers)
    portfolio = build_portfolio_df(st.session_state["df_mod"])
    portfolio = portfolio[portfolio["campaign"].isin(selected_campaigns)].reset_index(drop=True)

    # build rec_df (simple rules)
    recs = []
    for r in portfolio.to_dict(orient="records"):
        tri_row = combined_df[combined_df["campaign"]==r["campaign"]]
        final_conf = int(tri_row["final_confidence"].values[0]) if not tri_row.empty else r["confidence"]
        action = "Hold"
        reason = ""
        projected = 0
        if r["saturation"] > 75 and r["marginal_roas"] < r["avg_roas"]*0.6 and final_conf > 50:
            action = "Reduce spend"
            reason = "High saturation & low marginal ROAS"
            projected = -round(r["spend"]*0.02,0)
        elif r["elasticity_b"] > 0.5 and final_conf > 40:
            action = "Increase spend"
            reason = "Higher elasticity"
            projected = round(r["spend"]*0.03,0)
        recs.append({"campaign":r["campaign"], "action":action, "reason":reason, "projected_incremental":int(projected), "confidence":final_conf})
    rec_df = pd.DataFrame(recs)
    st.subheader("Recommendations snapshot")
    with st.expander("View recommendation table"):
        st.dataframe(rec_df, height=240)

    st.markdown("### Simulation type")
    sim_type = st.radio("Choose simulation", options=["Reallocate between campaigns","Change total portfolio spend"], index=0)

    if sim_type == "Reallocate between campaigns":
        if len(portfolio) < 2:
            st.info("Select at least 2 campaigns in the sidebar to reallocate.")
        else:
            from_campaign = st.selectbox("From campaign (reduce)", options=portfolio["campaign"].tolist(), index=0, key="sim_from")
            to_campaign = st.selectbox("To campaign (increase)", options=portfolio["campaign"].tolist(), index=1, key="sim_to")
            shift_pct = st.slider("Shift % of 'From' spend to 'To' (simulate)", 0, 50, 10, key="sim_shift")
            if st.button("Run reallocation simulation"):
                from_row = portfolio[portfolio["campaign"]==from_campaign].iloc[0]
                to_row = portfolio[portfolio["campaign"]==to_campaign].iloc[0]
                moved_amount = int(from_row["spend"] * shift_pct/100)
                projected_inc = to_row["marginal_roas"] * moved_amount
                from_loss = from_row["marginal_roas"] * moved_amount
                net_incremental = projected_inc - from_loss
                st.write("Moved amount:", f"${moved_amount:,}")
                st.write(f"Projected incremental revenue (gain): ${int(projected_inc):,}")
                st.write(f"Projected revenue loss from reduced campaign: ${int(from_loss):,}")
                st.success(f"Net projected incremental revenue (approx): ${int(net_incremental):,}")

    else:
        st.markdown("### Portfolio scale simulation")
        scale_mode = st.radio("Scale mode", options=["Increase total budget", "Decrease total budget"], index=0)
        if scale_mode == "Increase total budget":
            add_amount = st.number_input("Add absolute $ to portfolio (e.g., 50000)", min_value=0, value=100000, step=10000)
            strategy = st.selectbox("Allocate new dollars by", options=["triangulated_iROAS","marginal_roas"], index=0)
            if st.button("Run portfolio increase simulation"):
                remaining = int(add_amount)
                # rank campaigns by chosen strategy
                if strategy == "triangulated_iROAS" and not combined_df.empty:
                    rank = combined_df.set_index("campaign")["triangulated_iROAS"].to_dict()
                else:
                    rank = portfolio.set_index("campaign")["marginal_roas"].to_dict()
                # allocate proportionally to rank (positive values)
                total_rank = sum([v for v in rank.values() if v>0]) or 1.0
                projection = []
                net_inc_total = 0.0
                for c in portfolio["campaign"].tolist():
                    alloc = int(add_amount * (max(rank.get(c,0),0) / total_rank))
                    # incremental revenue approx = marginal_roas * alloc
                    marg = portfolio[portfolio["campaign"]==c]["marginal_roas"].iloc[0]
                    inc_rev = marg * alloc
                    net_inc_total += inc_rev
                    projection.append((c, alloc, int(inc_rev)))
                st.markdown("**Allocation & estimated incremental revenue**")
                for p in projection:
                    st.write(f"{p[0]} → +{pretty_currency(p[1])} → estimated incremental rev ${p[2]:,}")
                st.success(f"Net incremental revenue (approx): ${int(net_inc_total):,}")
        else:
            reduce_amount = st.number_input("Reduce absolute $ from portfolio (e.g., 50000)", min_value=0, value=100000, step=10000)
            strategy = st.selectbox("Reduce dollars from", options=["High saturation", "Low triangulated_iROAS"], index=0)
            if st.button("Run portfolio reduction simulation"):
                if strategy == "High saturation":
                    rank = portfolio.set_index("campaign")["saturation"].to_dict()
                    # remove proportional to saturation
                    total_rank = sum(rank.values()) or 1.0
                    impact = []
                    for c,v in rank.items():
                        cut = int(reduce_amount * (v / total_rank))
                        # revenue loss approx = marginal_roas * cut
                        marg = portfolio[portfolio["campaign"]==c]["marginal_roas"].iloc[0]
                        loss = marg * cut
                        impact.append((c, cut, int(loss)))
                    st.markdown("**Cuts & estimated revenue loss**")
                    for it in impact:
                        st.write(f"{it[0]} → -{pretty_currency(it[1])} → estimated revenue loss ${it[2]:,}")
                    st.warning(f"Net revenue loss (approx): ${int(sum([x[2] for x in impact])):,}")

    # Contribution margin inputs & display
    st.markdown("---")
    st.subheader("Contribution margin & first-party signals")
    st.markdown("Enter cost inputs to compute contribution margin. This helps prioritize by profit, not just revenue.")
    c1, c2, c3 = st.columns(3)
    with c1:
        aov = st.number_input("Assumed AOV ($)", value=50.0, key="sim_aov")
    with c2:
        cogs = st.number_input("COGS per order ($)", value=20.0, key="sim_cogs")
    with c3:
        promo = st.number_input("Avg promo per order ($)", value=2.0, key="sim_promo")
    ship = st.number_input("Avg shipping per order ($)", value=3.0, key="sim_ship")
    contribution_per_order = aov - cogs - promo - ship
    cm_pct = round(100 * (contribution_per_order / aov),1) if aov>0 else 0.0
    st.write(f"Contribution per order: ${contribution_per_order:.2f} — Contribution margin: {cm_pct}%")
    if use_cdp:
        st.info("First-party signals (CDP) enabled: attributed revenue shown as % of total revenue where simulated data exists.")

# ------------------------
# Tab 4: Experimentation Studio (Controlled + Synthetic)
# ------------------------
# ------------------------
# Tab 4: Experimentation Studio
# ------------------------
with tab4:
    st.header("Experimentation Studio")
    st.markdown(
        "Design holdout or scale tests, or run synthetic causal analysis when RCT is not feasible."
    )

    if portfolio.empty:
        st.info("No campaigns selected in scope. Pick campaigns in sidebar.")
    else:
        selected_campaign = st.selectbox(
            "Select campaign to validate",
            options=portfolio["campaign"].tolist(),
            key="exp_campaign"
        )

        c_row = portfolio[portfolio["campaign"] == selected_campaign].iloc[0]

        mode = st.radio(
            "Mode",
            options=[
                "Controlled Experiment (Holdout / Scale)",
                "Synthetic Causal Analysis"
            ],
            index=0
        )

        # ======================================================
        # CONTROLLED EXPERIMENT MODE
        # ======================================================
        if mode == "Controlled Experiment (Holdout / Scale)":

            test_type = st.radio(
                "Test Type",
                options=["Holdout (Geo Holdout)", "Scale (Spend Ramp)"],
                index=0
            )

            st.markdown("### DMA Pool (Simulated)")
            dmas_df = sample_dmas.copy()

            # Ensure DMA values are clean strings
            dmas_df["dma"] = dmas_df["dma"].astype(str)

            st.dataframe(dmas_df, height=200)

            st.markdown("### Recommended Candidate Designs")

            avg_order_value = st.number_input(
                "Assumed Avg Order Value ($)",
                value=50.0,
                key="exp_aov"
            )

            conv_rate = st.number_input(
                "Baseline Conversion Rate (decimal)",
                value=0.01,
                key="exp_conv"
            )

            candidates = recommend_test_designs(
                dmas_df,
                campaign_spend=c_row["spend"],
                avg_order_value=avg_order_value,
                conv_rate=conv_rate
            )

            cand_df = pd.DataFrame(candidates)[
                ["name", "size", "dmas", "n", "mde_pct", "est_conv_total"]
            ]
            st.table(cand_df)

            # -------------------------
            # Candidate selection first
            # -------------------------
            chosen_design = st.radio(
                "Pick candidate design or Custom",
                options=[c["name"] for c in candidates] + ["Custom"],
                index=1,
                key="candidate_radio"
            )

            pick = None
            if chosen_design != "Custom":
                pick = next(
                    (c for c in candidates if c["name"] == chosen_design),
                    None
                )
                if pick:
                    st.write("Candidate details:")
                    st.write(pick)

            # -------------------------
            # SAFE DMA MULTISELECT
            # -------------------------
            st.markdown("### Customize DMAs")

            dmas = dmas_df["dma"].astype(str).tolist()
            preselected = dmas[:5]

            if pick:
                raw_default = pick.get("dmas", [])
            else:
                raw_default = st.session_state.get("chosen_dmas", preselected)

            # Coerce default to valid list of strings
            try:
                current_default = [str(x) for x in raw_default]
            except Exception:
                current_default = preselected

            # Filter invalid values
            current_default = [x for x in current_default if x in dmas]

            if not current_default:
                current_default = preselected

            chosen_dmas = st.multiselect(
                "Choose treatment DMAs",
                options=dmas,
                default=current_default,
                key="chosen_dmas"
            )

            excluded_dmas = st.multiselect(
                "Exclude DMAs (optional)",
                options=dmas,
                default=[],
                key="excluded_dmas"
            )

            chosen_effective = [
                d for d in chosen_dmas if d not in excluded_dmas
            ]

            # -------------------------
            # Test configuration
            # -------------------------
            if test_type == "Holdout (Geo Holdout)":
                treatment_pct = st.slider(
                    "Treatment % of Normal Exposure",
                    50, 100, 90
                )
            else:
                treatment_pct = st.slider(
                    "Spend Increase % in Treatment DMAs",
                    5, 200, 25
                )

            start_dt = st.date_input(
                "Start Date",
                value=date.today() + timedelta(days=7)
            )

            duration = st.number_input(
                "Duration (days)",
                min_value=7,
                max_value=90,
                value=28
            )

            end_dt = start_dt + timedelta(days=int(duration))

            if len(chosen_effective) > 0:
                chosen_df = dmas_df[
                    dmas_df["dma"].isin(chosen_effective)
                ].copy()

                chosen_df["est_conv"] = (
                    (c_row["spend"] * chosen_df["spend_share"])
                    / avg_order_value
                ) * conv_rate * (duration / 28.0)

                total_n = max(1, int(chosen_df["est_conv"].sum()))

                mde = detectable_lift_proportion(
                    conv_rate, total_n
                )

                st.markdown("### Design Summary")
                st.write(f"Treatment DMAs: {chosen_effective}")
                st.write(f"Estimated Conversions: {total_n}")
                st.write(
                    f"Estimated MDE: ±{round(mde * 100, 2)}%"
                )
            else:
                st.warning("Select at least one DMA for treatment.")

        # ======================================================
        # SYNTHETIC CAUSAL MODE
        # ======================================================
        else:
            st.markdown("### Synthetic Causal Analysis")

            method = st.selectbox(
                "Method",
                [
                    "Historical Bid/Spend Variation",
                    "Geo-Level Cross Section",
                    "Time-Series Structural Model"
                ]
            )

            hist_window = st.selectbox(
                "Historical Window",
                ["90 days", "180 days", "365 days"]
            )

            avg_order_value = st.number_input(
                "Assumed Avg Order Value ($)",
                value=50.0,
                key="syn_aov"
            )

            conv_rate = st.number_input(
                "Baseline Conversion Rate (decimal)",
                value=0.01,
                key="syn_conv"
            )

            if st.button("Run Synthetic Analysis"):
                base_effect = c_row["marginal_roas"] * 0.02
                noise = np.random.normal(0, 0.5)
                synthetic_lift = round(base_effect + noise, 2)

                synthetic_conf = int(
                    max(30, min(90, 60 + np.random.randint(-15, 30)))
                )

                st.write(
                    f"Estimated Incremental Lift: {synthetic_lift}%"
                )

                st.write(
                    f"Model Confidence: {synthetic_conf}%"
                )

                if synthetic_conf < confidence_threshold:
                    st.warning(
                        "Confidence moderate. Consider running controlled holdout."
                    )
                else:
                    st.success(
                        "Confidence acceptable for directional decision."
                    )
# ------------------------
# End of app
# ------------------------