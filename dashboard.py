import random

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(layout="wide", page_title="Superliga Top Three Player Comparison Dashboard")


@st.cache_data
def load_data(path):
    df = pd.read_csv(path, parse_dates=["Date"], dayfirst=False)
    # standardize column names (strip whitespace)
    df.columns = [c.strip() for c in df.columns]
    # numeric columns to coerce
    num_cols = ["Min", "Gls", "Ast", "PK", "PKatt", "Sh", "SoT", "CrdY", "CrdR",
                "Fls", "Fld", "Off", "Crs", "TklW", "Int", "OG", "PKwon", "PKcon"]
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

pos_col = "Pos"

if pos_col:
    positions = sorted(df[pos_col].dropna().unique())
    positions_options = ["All positions"] + positions
    # persist position filter
    if "sel_positions" not in st.session_state:
        st.session_state["sel_positions"] = ["All positions"]
    sel_positions = st.sidebar.multiselect(
        "Filter by position",
        positions_options,
        default=st.session_state["sel_positions"],
        key="sel_positions"
    )
else:
    sel_positions = ["All positions"]

# build available players based on position filter
if pos_col and sel_positions and "All positions" not in sel_positions:
    players_available = sorted(df[df[pos_col].isin(sel_positions)]["Player"].dropna().unique())
else:
    players_available = sorted(df["Player"].dropna().unique())

# initialize random default selection once (persisted in session_state)
if "sel_players" not in st.session_state:
    k = min(5, len(players_available))
    st.session_state["sel_players"] = random.sample(players_available, k) if k > 0 else []

# ensure stored selections are valid for the current position filter
current_sel = st.session_state.get("sel_players", [])
clamped = [p for p in current_sel if p in players_available]
if clamped != current_sel:
    st.session_state["sel_players"] = clamped

# player multiselect bound to session_state
sel_players = st.sidebar.multiselect(
    "Select players to compare",
    players_available,
    key="sel_players"
)

# choose visualizations
st.sidebar.markdown("### Charts")
show_time = st.sidebar.checkbox("Per-match goals (time series)", value=True)
show_cum = st.sidebar.checkbox("Cumulative goals", value=True)
show_per90 = st.sidebar.checkbox("Per-90 comparison (Gls/Ast/Sh)", value=True)
show_shots = st.sidebar.checkbox("Shots vs SoT", value=True)
show_def = st.sidebar.checkbox("Defensive (TklW / Int)", value=True)
show_results = st.sidebar.checkbox("Match Results", value=True)
show_minutes = st.sidebar.checkbox("Minutes", value=True)

if not sel_players:
    st.info("Pick one or more players from the sidebar to see comparisons.")
    st.stop()


# helper aggregated stats
def aggregate_stats(df):
    agg = df.groupby("Player").agg(
        matches=("Date", "count"),
        minutes=("Min", "sum"),
        goals=("Gls", "sum"),
        assists=("Ast", "sum"),
        shots=("Sh", "sum"),
        sot=("SoT", "sum"),
        tklw=("TklW", "sum"),
        interceptions=("Int", "sum")
    ).reset_index()
    # per90 safe
    agg["goals_per90"] = np.where(agg["minutes"] > 0, agg["goals"] / agg["minutes"] * 90, 0)
    agg["assists_per90"] = np.where(agg["minutes"] > 0, agg["assists"] / agg["minutes"] * 90, 0)
    agg["shots_per90"] = np.where(agg["minutes"] > 0, agg["shots"] / agg["minutes"] * 90, 0)
    agg["sot_pct"] = np.where(agg["shots"] > 0, agg["sot"] / agg["shots"] * 100, 0)
    return agg


agg = aggregate_stats(df)

# filter selected players data
df_sel = df[df["Player"].isin(sel_players)].copy()
agg_sel = agg[agg["Player"].isin(sel_players)].copy()

st.title("Superliga Top Three Player Comparison Dashboard")
st.subheader(", ".join(sel_players))

# layout: two columns for match-level charts
col1, col2 = st.columns(2)

if show_time:
    # per-match goals time series
    with col1:
        ts = df_sel.sort_values("Date")
        fig = px.line(ts, x="Date", y="Gls", color="Player", markers=True,
                      title="Goals per match", labels={"Gls": "Goals"})
        st.plotly_chart(fig, use_container_width=True)

if show_cum:
    with col2:
        # cumulative goals by date per player
        cum = df_sel.sort_values(["Player", "Date"]).groupby(["Player", "Date"], as_index=False)["Gls"].sum()
        cum = cum.copy()
        cum["cum_goals"] = cum.groupby("Player")["Gls"].cumsum()
        fig2 = px.line(cum, x="Date", y="cum_goals", color="Player",
                       title="Cumulative goals", labels={"cum_goals": "Cumulative goals"})
        st.plotly_chart(fig2, use_container_width=True)

# full width per-90 comparison
if show_per90:
    metrics = ["goals_per90", "assists_per90", "shots_per90"]
    per90 = agg_sel[["Player"] + metrics].melt(id_vars="Player", var_name="metric", value_name="value")
    per90["metric"] = per90["metric"].replace({
        "goals_per90": "Goals / 90",
        "assists_per90": "Assists / 90",
        "shots_per90": "Shots / 90"
    })
    fig3 = px.bar(per90, x="Player", y="value", color="metric", barmode="group",
                  title="Per-90 comparisons", labels={"value": "Per 90"})
    st.plotly_chart(fig3, use_container_width=True)

if show_shots:
    st.markdown("### Shots and Shots on Target")
    shots_df = agg_sel[["Player", "shots", "sot"]].melt(id_vars="Player", var_name="type", value_name="count")
    fig4 = px.bar(shots_df, x="Player", y="count", color="type", barmode="group",
                  title="Shots and Shots on Target")
    st.plotly_chart(fig4, use_container_width=True)
    # SOT%
    sot_pct = agg_sel[["Player", "sot_pct"]].sort_values("sot_pct", ascending=False)
    fig4b = px.bar(sot_pct, x="Player", y="sot_pct", title="Shots on Target %", labels={"sot_pct": "SOT %"})
    st.plotly_chart(fig4b, use_container_width=True)

if show_def:
    st.markdown("### Defensive summary")
    def_df = agg_sel[["Player", "tklw", "interceptions", "minutes"]].copy()
    # per90 defenses
    def_df["tklw_per90"] = np.where(def_df["minutes"] > 0, def_df["tklw"] / def_df["minutes"] * 90, 0)
    def_df["int_per90"] = np.where(def_df["minutes"] > 0, def_df["interceptions"] / def_df["minutes"] * 90, 0)
    def_melt = def_df[["Player", "tklw_per90", "int_per90"]].melt(id_vars="Player", var_name="metric",
                                                                  value_name="value")
    def_melt["metric"] = def_melt["metric"].replace({"tklw_per90": "TklW / 90", "int_per90": "Int / 90"})
    fig5 = px.bar(def_melt, x="Player", y="value", color="metric", barmode="group", title="Defensive per-90")
    st.plotly_chart(fig5, use_container_width=True)

if show_results:
    st.markdown("### Results Analysis")

    if len(df_sel) == 0 or 'Result' not in df_sel.columns or df_sel['Result'].isnull().all():
        st.info("No result data available for the selected players.")
        st.stop()

    # Prepare results data
    def analyze_results(df):
        # Create a copy of player-match data with result info
        player_results = df.copy()
        # Extract result type (W, D, L) from Result column
        player_results['result_type'] = player_results['Result'].str[0]

        # Group by player and result type
        results_count = player_results.groupby(['Player', 'result_type']).size().unstack(fill_value=0)

        # Calculate total matches
        results_count['total'] = results_count.sum(axis=1)

        # Calculate percentages
        for col in ['W', 'D', 'L']:
            if col in results_count.columns:
                results_count[f'{col}_pct'] = round(results_count[col] / results_count['total'] * 100, 1)

        return results_count.reset_index()


    # Get results for selected players
    results_df = analyze_results(df_sel)

    # Display results in two columns
    col_results1, col_results2 = st.columns(2)

    # Column 1: Results count
    with col_results1:
        # Melt the count data for visualization
        results_melt = results_df[['Player', 'W', 'D', 'L']].melt(
            id_vars='Player', var_name='Result', value_name='Count'
        )
        fig_results = px.bar(
            results_melt,
            x='Player',
            y='Count',
            color='Result',
            color_discrete_map={'W': 'green', 'D': 'gray', 'L': 'red'},
            title='Matches by Result Type'
        )
        st.plotly_chart(fig_results, use_container_width=True)

    # Column 2: Results percentage
    with col_results2:
        # Melt the percentage data for visualization
        if all(col in results_df.columns for col in ['W_pct', 'D_pct', 'L_pct']):
            pct_cols = ['Player', 'W_pct', 'D_pct', 'L_pct']
        else:
            # Handle case where some result types might be missing
            pct_cols = ['Player'] + [f'{col}_pct' for col in ['W', 'D', 'L'] if f'{col}_pct' in results_df.columns]

        results_pct = results_df[pct_cols].melt(
            id_vars='Player',
            var_name='Result',
            value_name='Percentage'
        )
        # Clean up result type names
        results_pct['Result'] = results_pct['Result'].str.replace('_pct', '')

        fig_pct = px.bar(
            results_pct,
            x='Player',
            y='Percentage',
            color='Result',
            color_discrete_map={'W': 'green', 'D': 'gray', 'L': 'red'},
            title='Results Distribution (%)'
        )
        fig_pct.update_layout(yaxis_range=[0, 100])
        st.plotly_chart(fig_pct, use_container_width=True)

if show_minutes:
    st.markdown("### Minutes Analysis")
    if len(df_sel) == 0 or df_sel['Min'].sum() == 0:
        st.info("No minutes data available for the selected players.")
        st.stop()
    # Prepare minutes data
    minutes_df = df_sel.groupby('Player').agg(
        total_minutes=('Min', 'sum'),
        matches_played=('Min', lambda x: (x > 0).sum()),
        avg_minutes=('Min', lambda x: round(x.mean(), 1)),
        min_minutes=('Min', lambda x: x[x > 0].min() if x[x > 0].size > 0 else 0),
        max_minutes=('Min', 'max'),
        full_matches=('Min', lambda x: (x >= 90).sum())
    ).reset_index()

    # Add percentage of possible minutes
    minutes_df['full_match_pct'] = round(minutes_df['full_matches'] / minutes_df['matches_played'] * 100, 1)

    # Display minutes in two columns
    col_min1, col_min2 = st.columns(2)

    # Column 1: Total minutes
    with col_min1:
        fig_total = px.bar(
            minutes_df,
            x='Player',
            y='total_minutes',
            title='Total Minutes Played',
            labels={'total_minutes': 'Minutes'}
        )
        st.plotly_chart(fig_total, use_container_width=True)

    # Column 2: Average minutes per match
    with col_min2:
        fig_avg = px.bar(
            minutes_df,
            x='Player',
            y='avg_minutes',
            title='Average Minutes per Match',
            labels={'avg_minutes': 'Minutes'}
        )
        fig_avg.update_layout(yaxis_range=[0, 100])
        st.plotly_chart(fig_avg, use_container_width=True)

    # Minutes distribution (min, avg, max)
    minutes_range = minutes_df[['Player', 'min_minutes', 'avg_minutes', 'max_minutes']].melt(
        id_vars='Player',
        var_name='Stat',
        value_name='Minutes'
    )
    minutes_range['Stat'] = minutes_range['Stat'].map({
        'min_minutes': 'Minimum',
        'avg_minutes': 'Average',
        'max_minutes': 'Maximum'
    })

    fig_range = px.bar(
        minutes_range,
        x='Player',
        y='Minutes',
        color='Stat',
        barmode='group',
        title='Minutes Distribution per Match'
    )
    st.plotly_chart(fig_range, use_container_width=True)

    # Full matches percentage
    fig_full = px.bar(
        minutes_df,
        x='Player',
        y='full_match_pct',
        title='Percentage of Matches Played Full Time (90+ mins)',
        labels={'full_match_pct': 'Percentage (%)'}
    )
    fig_full.update_layout(yaxis_range=[0, 100])
    st.plotly_chart(fig_full, use_container_width=True)

# show aggregated table
with st.expander("Show aggregated stats table"):
    st.dataframe(agg_sel.sort_values("goals", ascending=False).reset_index(drop=True))

st.caption(
    "Columns used: Player, Date, Min, Gls, Ast, Sh, SoT, TklW, Int. Replace `all_players.csv` or upload a CSV with matching column names.")
