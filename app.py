# app.py
# Adobe Incremental Intelligence Engine - prototype (with Synthetic Causal + Holdout + Scale)
# Dependencies: streamlit, pandas, numpy, plotly
# Author: Shubham Jain (prototype)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import math
import json
from datetime import date, timedelta

st.set_page_config(page_title="Incremental Intelligence Engine", layout="wide")

# ------------------------
# Helpers & Mock Data
# ------------------------
def revenue_from_spend(a, b, spend):
    return a * (spend ** b)

def marginal_revenue(a, b, spend):
    return a * b * (spend ** (b - 1)) if spend > 0 else 0.0

def saturation_score(avg_roas, marginal_roas):
    if avg_roas <= 0:
        return 0
    ratio = marginal_roas / (avg_roas + 1e-9)
    score = (1 - min(max(ratio, 0), 1)) * 100
    return round(score, 0)

def confidence_score(experiment_present, mmm_present, attribution_variance):
    exp_w = 0.5 if experiment_present else 0.0
    mmm_w = 0.3 if mmm_present else 0.0
    att_w = max(0, 1 - min(attribution_variance / 2.0, 1)) * 0.2
    score = int((exp_w + mmm_w + att_w) * 100)
    return min(score, 100)

# Simple MDE approximation for proportion tests (two-sided, alpha=0.05, power=0.8)
def detectable_lift_proportion(p0, n):
    if n <= 1:
        return 1.0
    z_alpha = 1.96  # two-sided 5%
    z_beta = 0.84   # power 80%
    mde = (z_alpha + z_beta) * math.sqrt(2 * p0 * (1 - p0) / n)
    return mde

def estimate_n_from_spend(spend, avg_order_value, conv_rate):
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
# Default portfolio and DMAs (mock)
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
        a = r["a"]
        b = r["b"]
        marg_rev = marginal_revenue(a, b, spend)
        marginal_roas = marg_rev
        sat = saturation_score(r["avg_roas"], marginal_roas)
        conf = confidence_score(r["has_experiment"], r["mmm_share"]>0.0, r["att_variance"])
        rows.append({
            "campaign": r["campaign"],
            "spend": spend,
            "avg_roas": r["avg_roas"],
            "marginal_roas": round(marginal_roas,2),
            "elasticity_b": r["b"],
            "saturation": sat,
            "confidence": conf,
            "has_experiment": r["has_experiment"],
            "mmm_share": r["mmm_share"],
            "att_variance": r["att_variance"],
            "type": r["type"]
        })
    return pd.DataFrame(rows)

# ------------------------
# Session state init
# ------------------------
if "df_mod" not in st.session_state:
    st.session_state["df_mod"] = pd.DataFrame(default_campaigns)
if "data_refresh_ts" not in st.session_state:
    st.session_state["data_refresh_ts"] = None

# ------------------------
# Sidebar: Governance & global settings + scope
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
st.sidebar.markdown("**Experimentation defaults**")
min_duration_days = st.sidebar.number_input("Min experiment duration (days)", min_value=7, max_value=180, value=28)
st.sidebar.markdown("---")
st.sidebar.caption("Prototype by Shubham Jain — Elasticity + Triangulation + Experimentation CTA + Governance")

# ------------------------
# Build portfolio filtered by selected campaigns and scaled by date range
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
# Main app UI
# ------------------------
st.title("Adobe Incremental Intelligence Engine (Prototype)")
st.markdown("Elasticity + Triangulation + Recommendation + Experimentation CTA + Governance")

tab1, tab2, tab3, tab4 = st.tabs(["📊 Portfolio Overview", "🔍 Triangulation", "💡 Recommendation", "🧪 Experimentation Studio"])

# ------------------------
# Tab 1: Portfolio Overview (clean UX: show selected-date spend)
# ------------------------
with tab1:
    st.header("Portfolio Overview")
    st.markdown(f"Date window: **{start_date.date()}** → **{end_date.date()}**")
    st.metric("Portfolio total spend (selected range)", f"${agg['total_spend']:,}")
    cols = st.columns(3)
    cols[0].metric("Weighted Avg ROAS", agg["w_avg_roas"])
    cols[1].metric("Weighted Marginal ROAS", agg["w_marginal"])
    cols[2].metric("Weighted Confidence", f"{agg['w_conf']}%")

    left, right = st.columns([2,1])
    with left:
        st.dataframe(portfolio[["campaign","type","spend","avg_roas","marginal_roas","elasticity_b","saturation","confidence"]].sort_values("saturation", ascending=False), height=360)
    with right:
        fig = px.bar(portfolio, x="campaign", y=["avg_roas","marginal_roas"], barmode="group", title="Avg ROAS vs Marginal ROAS")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("**Heatmap (Saturation)**")
        sat_fig = px.imshow([portfolio["saturation"]], labels=dict(x="Campaign", color="Saturation"), x=portfolio["campaign"])
        st.plotly_chart(sat_fig, use_container_width=True)

    st.markdown("**Edit baseline (monthly) spend for simulation only**")
    new_spends = []
    for i, row in enumerate(df_base.to_dict(orient="records")):
        s = st.number_input(f"{row['campaign']} base spend (monthly)", value=int(row["spend"]), step=50000, key=f"base_spend_{i}")
        new_spends.append(s)

    if st.button("Apply spend changes (update baseline)"):
        df_mod = st.session_state["df_mod"].copy()
        for i in range(len(df_mod)):
            df_mod.at[i, "spend"] = int(st.session_state.get(f"base_spend_{i}", df_mod.at[i,"spend"]))
        st.session_state["df_mod"] = df_mod
        st.rerun()

# ------------------------
# Tab 2: Triangulation (auto-populate or manual override) + Synthetic CTA
# ------------------------
with tab2:
    st.header("Measurement Triangulation")
    st.markdown("Auto-populate fields when integrations available (simulated). You can still override values manually.")

    methods = []
    for i, r in enumerate(portfolio.to_dict(orient="records")):
        st.markdown(f"**{r['campaign']}**")
        col1, col2, col3, col4 = st.columns(4)

        default_att = float(round(r["spend"] * (r["avg_roas"] * 0.05), 0))
        default_exp = float(round(r["spend"] * (0.01 if r["has_experiment"] else 0.02), 0))
        default_mmm = float(round(r["spend"] * (r["mmm_share"] * 0.04 + 0.01), 0))

        st.session_state.setdefault(f"att_{i}", default_att if auto_pop else 0.0)
        st.session_state.setdefault(f"exp_{i}", default_exp if auto_pop else 0.0)
        st.session_state.setdefault(f"mmm_{i}", default_mmm if auto_pop else 0.0)
        st.session_state.setdefault(f"conf_{i}", 0)

        with col1:
            att_val = st.number_input(f"Attribution inc ($) - {r['campaign']}", value=float(st.session_state[f"att_{i}"]), key=f"att_{i}")
        with col2:
            exp_val = st.number_input(f"Experiment inc ($) - {r['campaign']}", value=float(st.session_state[f"exp_{i}"]), key=f"exp_{i}")
        with col3:
            mmm_val = st.number_input(f"MMM inc ($) - {r['campaign']}", value=float(st.session_state[f"mmm_{i}"]), key=f"mmm_{i}")
        with col4:
            conf_val = st.slider(f"Confidence override % (0 = auto) - {r['campaign']}", 0, 100, int(st.session_state[f"conf_{i}"]), key=f"conf_{i}")

        methods.append({
            "campaign": r["campaign"],
            "attribution_inc": float(att_val),
            "experiment_inc": float(exp_val),
            "mmm_inc": float(mmm_val),
            "conf_override": int(conf_val),
            "auto_conf": r["confidence"]
        })

    methods_df = pd.DataFrame(methods)
    st.markdown("**Triangulation table (current inputs)**")
    st.dataframe(methods_df, height=260)

    st.markdown("**Conflict detection rules (prototype)**")
    st.write("- If Attribution ≥ 2× Experiment → possible over-attribution")
    st.write("- If MMM > Attribution → possible MMM-driven signal (under-attribution or cross-channel driver)")

    combined = []
    low_conf_campaigns = []
    for idx, row in methods_df.iterrows():
        flag = ""
        if row["attribution_inc"] >= 2 * max(row["experiment_inc"], 1):
            flag = "Over-attribution risk"
        elif row["mmm_inc"] > row["attribution_inc"]:
            flag = "MMM > Attribution"
        final_conf = int(row["conf_override"]) if row["conf_override"] > 0 else int(row["auto_conf"])
        combined.append({
            "campaign": row["campaign"],
            "attribution_inc": row["attribution_inc"],
            "experiment_inc": row["experiment_inc"],
            "mmm_inc": row["mmm_inc"],
            "final_confidence": final_conf,
            "flag": flag
        })
        if final_conf < confidence_threshold:
            low_conf_campaigns.append(row["campaign"])

    combined_df = pd.DataFrame(combined)
    st.table(combined_df)

    # CTA: run synthetic causal for low-confidence campaigns if any
    if len(low_conf_campaigns) > 0:
        st.info(f"Low confidence detected for: {', '.join(low_conf_campaigns)} (threshold {confidence_threshold}%).")
        if st.button("Run Synthetic Causal Analysis for low-confidence campaigns"):
            st.markdown("**Synthetic results (quick simulation)**")
            for c in low_conf_campaigns:
                row = next((m for m in methods if m["campaign"]==c), None)
                # mock synthetic: base on marginal_roas scaled; add noise
                portfolio_row = portfolio[portfolio["campaign"]==c].iloc[0]
                base_effect = portfolio_row["marginal_roas"] * 0.02
                noise = np.random.normal(0, 0.5)
                synthetic_lift = round(base_effect + noise, 2)
                synthetic_conf = int(max(30, min(85, 60 + np.random.randint(-10,20))))
                st.write(f"Campaign: **{c}** — Synthetic lift estimate: **{synthetic_lift}%**, Confidence: **{synthetic_conf}%**")
            st.warning("Synthetic results are directional. Consider controlled validation.")

# ------------------------
# Tab 3: Recommendation Engine
# ------------------------
with tab3:
    st.header("Recommendation Engine")
    st.markdown("Recommendations combine marginal economics + triangulation + governance. Simulate moving dollars to see projected impact.")

    portfolio = build_portfolio_df(st.session_state["df_mod"])
    portfolio = portfolio[portfolio["campaign"].isin(selected_campaigns)].reset_index(drop=True)

    recs = []
    for r in portfolio.to_dict(orient="records"):
        tri_row = combined_df[combined_df["campaign"]==r["campaign"]]
        final_conf = int(tri_row["final_confidence"].values[0]) if not tri_row.empty else r["confidence"]
        action = "Hold"
        reason = ""
        projected = 0
        if r["saturation"] > 70 and r["marginal_roas"] < r["avg_roas"]*0.6 and final_conf > 50:
            action = "Reduce spend"
            reason = "High saturation & low marginal ROAS"
            projected = -round(r["spend"]*0.02,0)
        elif r["elasticity_b"] > 0.5 and final_conf > 40:
            action = "Increase spend"
            reason = "High elasticity"
            projected = round(r["spend"]*0.03,0)
        recs.append({"campaign":r["campaign"], "action":action, "reason":reason, "projected_incremental":int(projected), "confidence":final_conf})
    rec_df = pd.DataFrame(recs)
    st.dataframe(rec_df, height=260)

    st.markdown("**Simulate reallocation**")
    if len(portfolio) >= 2:
        from_campaign = st.selectbox("From campaign (reduce)", options=portfolio["campaign"].tolist(), index=0, key="sim_from")
        to_campaign = st.selectbox("To campaign (increase)", options=portfolio["campaign"].tolist(), index=1, key="sim_to")
        shift_pct = st.slider("Shift % of 'From' spend to 'To' (simulate)", 0, 50, 10, key="sim_shift")
        if st.button("Show simulation impact"):
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
        st.info("Select at least 2 campaigns in sidebar to simulate reallocation.")

    total_spend = portfolio["spend"].sum() if not portfolio.empty else 0
    brand_spend = portfolio[portfolio["type"]=="brand"]["spend"].sum() if not portfolio.empty else 0
    current_brand_pct = round(100 * brand_spend / total_spend, 1) if total_spend > 0 else 0.0
    st.markdown("**Governance check**")
    if current_brand_pct > brand_cap:
        st.error(f"Governance violation: Brand % ({current_brand_pct}%) exceeds cap ({brand_cap}%). Approval required.")
    else:
        st.success("No governance violation (brand % ok).")

# ------------------------
# Tab 4: Experimentation Studio (with Synthetic option)
# ------------------------
with tab4:
    st.header("Experimentation Studio")
    st.markdown("Design holdouts or scale tests. Get DMA recommendations, run synthetic causal analyses, review candidate designs, and export a payload to activate (prototype).")

    if portfolio.empty:
        st.info("No campaigns selected in scope. Pick campaigns in sidebar.")
    else:
        selected_campaign = st.selectbox("Select campaign to validate", options=portfolio["campaign"].tolist(), key="exp_campaign")
        c_row = portfolio[portfolio["campaign"]==selected_campaign].iloc[0]

        # Mode selector: Controlled vs Synthetic
        mode = st.radio("Mode", options=["Controlled Experiment (Holdout/Scale)","Synthetic Causal Analysis"], index=0)

        if mode.startswith("Controlled"):
            test_type = st.radio("Test type", options=["Holdout (geo holdout)","Scale (spend ramp)"], index=0)

            st.markdown("**DMA pool (sample simulated DMAs)**")
            dmas_df = sample_dmas.copy()
            st.dataframe(dmas_df, width=700, height=200)

            st.markdown("**Recommended candidate designs**")
            avg_order_value = st.number_input("Assumed avg order value ($)", value=50.0, key="exp_aov2")
            conv_rate = st.number_input("Baseline conv rate (decimal)", value=0.01, key="exp_conv2")

            candidates = recommend_test_designs(dmas_df, campaign_spend=c_row["spend"], avg_order_value=avg_order_value, conv_rate=conv_rate)
            cand_df = pd.DataFrame(candidates)[["name","size","dmas","n","mde_pct","est_conv_total"]]
            st.table(cand_df)

            # DMA selection safe pattern
            st.markdown("**Customize DMAs**")
            dmas = dmas_df["dma"].tolist()
            preselected = dmas[:5]
            current_default = st.session_state.get("chosen_dmas", preselected)

            chosen_dmas = st.multiselect(
                "Choose treatment DMAs (pick some or use candidate)",
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

            chosen_effective = [d for d in chosen_dmas if d not in excluded_dmas]

            chosen_design = st.radio(
                "Pick candidate design or custom",
                options=[c["name"] for c in candidates] + ["Custom"],
                index=1,
                key="candidate_radio"
            )

            if chosen_design != "Custom":
                pick = next((c for c in candidates if c["name"]==chosen_design), None)
                if pick:
                    st.write("Candidate details:")
                    st.write(pick)
                    if st.button(f"Apply '{chosen_design}' candidate to DMAs"):
                        st.session_state["chosen_dmas"] = pick["dmas"]
                        st.rerun()

            if test_type.startswith("Holdout"):
                treatment_pct = st.slider("Treatment percent of normal exposure (e.g., 90 means 90% of normal)", 50, 100, 90)
            else:
                treatment_pct = st.slider("Spend increase percent in treatment DMAs (e.g., 20 means +20%)", 5, 200, 25)

            start_dt = st.date_input("Start date", value=date.today() + timedelta(days=7))
            duration = st.number_input("Duration (days)", min_value=7, max_value=90, value=28)
            end_dt = start_dt + timedelta(days=int(duration))

            if len(chosen_effective) == 0:
                st.warning("No DMAs selected for treatment — choose at least one.")
            else:
                chosen_df = dmas_df[dmas_df["dma"].isin(chosen_effective)].copy()
                chosen_df["est_conv"] = ((c_row["spend"] * chosen_df["spend_share"]) / avg_order_value) * conv_rate * (duration/28.0)
                total_n = max(1, int(chosen_df["est_conv"].sum()))
                mde = detectable_lift_proportion(conv_rate, total_n)
                st.markdown("**Selected design summary**")
                st.write(f"Treatment DMAs: {chosen_effective}")
                st.write(f"Estimated conversions in duration (treatment): {total_n}")
                st.write(f"Estimated MDE (approx): ±{round(mde*100,2)}%")

            st.markdown("---")
            st.markdown("**Export payload / Simulate activation**")
            notes = st.text_area("Notes (optional)", value="Prototype activation payload")
            payload = {
                "campaign": selected_campaign,
                "test_type": "holdout" if test_type.startswith("Holdout") else "scale",
                "start_date": start_dt.isoformat(),
                "end_date": end_dt.isoformat(),
                "treatment_dmas": chosen_effective,
                "excluded_dmas": excluded_dmas,
                "treatment_pct": int(treatment_pct),
                "duration_days": int(duration),
                "assumptions": {"avg_order_value": avg_order_value, "baseline_conv_rate": conv_rate},
                "notes": notes
            }
            st.code(payload, language="json")
            if st.button("Export JSON / Simulate Activate"):
                filename = f"experiment_payload_{selected_campaign}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json"
                with open(filename,"w") as f:
                    json.dump(payload, f, indent=2)
                st.success(f"Payload written to {filename} (this simulates activation).")
                st.info("In production this payload would be sent to orchestration services or used to call Google Ads APIs.")

        else:
            # Synthetic causal mode (explicit)
            st.markdown("### Synthetic Causal Analysis (explicit mode)")
            causal_method = st.selectbox(
                "Method",
                options=[
                    "Historical Bid/Spend Variation",
                    "Geo-Level Cross Section (Synthetic Control)",
                    "Time-Series Structural Model"
                ]
            )
            hist_window = st.selectbox("Historical window", options=["90 days","180 days","365 days"])
            avg_order_value = st.number_input("Assumed avg order value ($)", value=50.0, key="syn_aov")
            conv_rate = st.number_input("Baseline conv rate (decimal)", value=0.01, key="syn_conv")
            if st.button("Run Synthetic Causal Analysis (explicit)"):
                base_effect = c_row["marginal_roas"] * 0.02
                noise = np.random.normal(0, 0.5)
                synthetic_lift = round(base_effect + noise, 2)
                synthetic_conf = int(max(30, min(90, 60 + np.random.randint(-15,30))))
                st.write(f"Estimated Incremental Lift (synthetic) for {selected_campaign}: **{synthetic_lift}%**")
                st.write(f"Model Confidence: **{synthetic_conf}%**")
                if synthetic_conf < confidence_threshold:
                    st.warning("Confidence moderate/low. Recommend controlled validation (holdout) for higher certainty.")
                else:
                    st.success("Synthetic result has acceptable confidence for directional decisions.")
                # show recommended next steps and allow export
                st.markdown("**Next steps**")
                st.write("- Consider running a controlled geo holdout for validation.")
                st.write("- If budget constrained, consider partial holdout or longer duration.")
                payload = {
                    "campaign": selected_campaign,
                    "method": causal_method,
                    "hist_window": hist_window,
                    "synthetic_lift_pct": synthetic_lift,
                    "synthetic_confidence_pct": synthetic_conf,
                    "assumptions": {"avg_order_value": avg_order_value, "baseline_conv_rate": conv_rate}
                }
                st.code(payload, language="json")
                if st.button("Export synthetic result JSON"):
                    fname = f"synthetic_result_{selected_campaign}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json"
                    with open(fname,"w") as f:
                        json.dump(payload, f, indent=2)
                    st.success(f"Synthetic payload written to {fname}.")

# ------------------------
# End of app
# ------------------------