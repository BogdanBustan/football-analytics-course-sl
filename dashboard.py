import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(layout="wide", page_title="Player comparison dashboard")

@st.cache_data
def load_data(path):
    df = pd.read_csv(path, parse_dates=["Date"], dayfirst=False)
    # standardize column names (strip whitespace)
    df.columns = [c.strip() for c in df.columns]
    # numeric columns to coerce
    num_cols = ["Min","Gls","Ast","PK","PKatt","Sh","SoT","CrdY","CrdR",
                "Fls","Fld","Off","Crs","TklW","Int","OG","PKwon","PKcon"]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    # keep Date sorted
    df = df.sort_values("Date")
    return df


try:
    df = load_data("all_players.csv")
except Exception as e:
    st.error("No CSV loaded and `all_players.csv` not found.")
    st.stop()

# player selector
players = sorted(df["Player"].dropna().unique())
default = players[:5] if len(players) >= 5 else players
sel_players = st.sidebar.multiselect("Select players to compare", players, default)

# choose visualizations
st.sidebar.markdown("### Charts")
show_time = st.sidebar.checkbox("Per-match goals (time series)", value=True)
show_cum = st.sidebar.checkbox("Cumulative goals", value=True)
show_per90 = st.sidebar.checkbox("Per-90 comparison (Gls/Ast/Sh)", value=True)
show_shots = st.sidebar.checkbox("Shots vs SoT", value=True)
show_def = st.sidebar.checkbox("Defensive (TklW / Int)", value=False)

if not sel_players:
    st.info("Pick one or more players from the sidebar to see comparisons.")
    st.stop()

# helper aggregated stats
def aggregate_stats(df):
    agg = df.groupby("Player").agg(
        matches=("Date","count"),
        minutes=("Min","sum"),
        goals=("Gls","sum"),
        assists=("Ast","sum"),
        shots=("Sh","sum"),
        sot=("SoT","sum"),
        tklw=("TklW","sum"),
        interceptions=("Int","sum")
    ).reset_index()
    # per90 safe
    agg["goals_per90"] = np.where(agg["minutes"]>0, agg["goals"]/agg["minutes"]*90, 0)
    agg["assists_per90"] = np.where(agg["minutes"]>0, agg["assists"]/agg["minutes"]*90, 0)
    agg["shots_per90"] = np.where(agg["minutes"]>0, agg["shots"]/agg["minutes"]*90, 0)
    agg["sot_pct"] = np.where(agg["shots"]>0, agg["sot"]/agg["shots"]*100, 0)
    return agg

agg = aggregate_stats(df)

# filter selected players data
df_sel = df[df["Player"].isin(sel_players)].copy()
agg_sel = agg[agg["Player"].isin(sel_players)].copy()

st.title("Player comparison")
st.subheader(", ".join(sel_players))

# layout: two columns for match-level charts
col1, col2 = st.columns(2)

if show_time:
    # per-match goals time series
    with col1:
        ts = df_sel.sort_values("Date")
        fig = px.line(ts, x="Date", y="Gls", color="Player", markers=True,
                      title="Goals per match", labels={"Gls":"Goals"})
        st.plotly_chart(fig, use_container_width=True)

if show_cum:
    with col2:
        # cumulative goals by date per player
        cum = df_sel.sort_values(["Player","Date"]).groupby(["Player","Date"], as_index=False)["Gls"].sum()
        cum = cum.copy()
        cum["cum_goals"] = cum.groupby("Player")["Gls"].cumsum()
        fig2 = px.line(cum, x="Date", y="cum_goals", color="Player",
                       title="Cumulative goals", labels={"cum_goals":"Cumulative goals"})
        st.plotly_chart(fig2, use_container_width=True)

# full width per-90 comparison
if show_per90:
    metrics = ["goals_per90","assists_per90","shots_per90"]
    per90 = agg_sel[["Player"] + metrics].melt(id_vars="Player", var_name="metric", value_name="value")
    per90["metric"] = per90["metric"].replace({
        "goals_per90":"Goals / 90",
        "assists_per90":"Assists / 90",
        "shots_per90":"Shots / 90"
    })
    fig3 = px.bar(per90, x="Player", y="value", color="metric", barmode="group",
                  title="Per-90 comparisons", labels={"value":"Per 90"})
    st.plotly_chart(fig3, use_container_width=True)

if show_shots:
    st.markdown("### Shots and Shots on Target")
    shots_df = agg_sel[["Player","shots","sot"]].melt(id_vars="Player", var_name="type", value_name="count")
    fig4 = px.bar(shots_df, x="Player", y="count", color="type", barmode="group",
                  title="Shots and Shots on Target")
    st.plotly_chart(fig4, use_container_width=True)
    # SOT%
    sot_pct = agg_sel[["Player","sot_pct"]].sort_values("sot_pct", ascending=False)
    fig4b = px.bar(sot_pct, x="Player", y="sot_pct", title="Shots on Target %", labels={"sot_pct":"SOT %"})
    st.plotly_chart(fig4b, use_container_width=True)

if show_def:
    st.markdown("### Defensive summary")
    def_df = agg_sel[["Player","tklw","interceptions","minutes"]].copy()
    # per90 defenses
    def_df["tklw_per90"] = np.where(def_df["minutes"]>0, def_df["tklw"]/def_df["minutes"]*90, 0)
    def_df["int_per90"] = np.where(def_df["minutes"]>0, def_df["interceptions"]/def_df["minutes"]*90, 0)
    def_melt = def_df[["Player","tklw_per90","int_per90"]].melt(id_vars="Player", var_name="metric", value_name="value")
    def_melt["metric"] = def_melt["metric"].replace({"tklw_per90":"TklW / 90","int_per90":"Int / 90"})
    fig5 = px.bar(def_melt, x="Player", y="value", color="metric", barmode="group", title="Defensive per-90")
    st.plotly_chart(fig5, use_container_width=True)

# show aggregated table
with st.expander("Show aggregated stats table"):
    st.dataframe(agg_sel.sort_values("goals", ascending=False).reset_index(drop=True))

st.caption("Columns used: Player, Date, Min, Gls, Ast, Sh, SoT, TklW, Int. Replace `all_players.csv` or upload a CSV with matching column names.")
