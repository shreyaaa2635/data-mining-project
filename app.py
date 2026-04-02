
import streamlit as st
import pandas as pd
import numpy as np
import time
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from data.ghcn_loader import load_climate_data
from sting.sting_algorithm import STINGAnalyzer, LEVEL_CONFIG, MIN_COUNT_THRESHOLDS, TEMP_TOLERANCE
from sting.visualizer import (
    build_pydeck,
    plot_temperature_distribution,
    plot_cluster_summary,
    plot_level_stats,
)

st.set_page_config(
    page_title="STING — Global Climate Analysis",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;800&display=swap');

  :root {
    --bg:       #0a0e1a;
    --surface:  #0d1117;
    --border:   #1e2433;
    --accent:   #4fc3f7;
    --warm:     #ff7043;
    --cool:     #4fc3f7;
    --text:     #c9d1d9;
    --muted:    #6e7681;
    --green:    #56d364;
  }

  html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'Syne', sans-serif;
  }

  [data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border);
  }

  h1, h2, h3 { font-family: 'Syne', sans-serif; font-weight: 800; }

  .metric-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 14px 18px;
    text-align: center;
  }
  .metric-val  { font-size: 2rem; font-weight: 800; font-family: 'Space Mono', monospace; color: var(--accent); }
  .metric-lbl  { font-size: 0.72rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.12em; margin-top: 4px; }

  .badge {
    display:inline-block; padding: 2px 10px; border-radius: 99px;
    font-size: 0.75rem; font-weight: 700; font-family: 'Space Mono', monospace;
    margin: 2px;
  }
  .badge-blue  { background:#1c3d5a; color:#4fc3f7; }
  .badge-red   { background:#4d1f1a; color:#ff7043; }
  .badge-green { background:#1a3a28; color:#56d364; }
  .badge-gray  { background:#1e2433; color:#8b949e; }

  .level-pill {
    padding:4px 12px; border-radius:6px; margin:2px 0;
    font-family:'Space Mono',monospace; font-size:0.78rem;
    background:#111827; border:1px solid #1e2433;
    display:flex; justify-content:space-between; align-items:center;
  }

  .stSlider > div { color: var(--text) !important; }
  .stSelectbox label, .stSlider label { color: var(--muted) !important; font-size:0.8rem; }

  div[data-testid="stHorizontalBlock"] > div { gap: 10px; }

  footer { display: none; }
</style>
""", unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def get_climate_data() -> pd.DataFrame:
    return load_climate_data()


@st.cache_resource(show_spinner=False)
def get_analyzer(data_hash: int) -> STINGAnalyzer:
    df = get_climate_data()
    analyzer = STINGAnalyzer(df)
    analyzer.build_hierarchy()
    return analyzer

st.markdown("""
<div style='padding:20px 0 8px 0'>
  <h1 style='margin:0;font-size:2.4rem;letter-spacing:-0.5px'>
    🌍 STING <span style='color:#4fc3f7'>Climate</span> Analyzer
  </h1>
  <p style='color:#6e7681;margin:4px 0 0;font-size:0.9rem;font-family:Space Mono,monospace'>
    Statistical Information Grid · 5-Level Spatial Hierarchy · GHCN Dataset
  </p>
</div>
""", unsafe_allow_html=True)
st.markdown("---")

with st.sidebar:
    st.markdown("### ⚙️ Query Parameters")

    level_labels = [f"L{i}: {cfg[2]}" for i, cfg in enumerate(LEVEL_CONFIG)]
    selected_level_label = st.selectbox(
        "Grid Resolution Level",
        options=level_labels,
        index=2,
        help="Higher levels = finer spatial resolution",
    )
    selected_level = level_labels.index(selected_level_label)

    st.markdown("---")
    st.markdown("**Temperature Range (°C)**")
    temp_min, temp_max = st.slider(
        "Temperature window",
        min_value=-40.0, max_value=50.0,
        value=(-5.0, 10.0), step=0.5,
        label_visibility="collapsed"
    )

    min_cluster_size = st.slider(
        "Min cluster size (cells)",
        min_value=1, max_value=20, value=2,
        help="Discard clusters with fewer cells than this",
    )

    st.markdown("---")
    st.markdown("### 🗺️ Map Style")
    map_style = st.selectbox(
        "Basemap",
        options=["dark", "light", "satellite", "terrain"],
        index=0,
        label_visibility="collapsed",
    )

    st.markdown("### 🔬 Layer Options")
    show_clusters  = st.checkbox("Show cluster overlay",  value=True)
    show_heatmap   = st.checkbox("Show raw station heatmap", value=False)

    st.markdown("---")
    st.markdown("### 📊 Threshold Info")
    for i, (_, _, lbl) in enumerate(LEVEL_CONFIG):
        st.markdown(f"""
        <div class='level-pill'>
          <span>L{i} {lbl}</span>
          <span>
            <span class='badge badge-blue'>n≥{MIN_COUNT_THRESHOLDS[i]}</span>
            <span class='badge badge-red'>±{TEMP_TOLERANCE[i]}°C</span>
          </span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    run_btn = st.button("▶ Run STING Query", use_container_width=True, type="primary")

with st.spinner("Loading GHCN climate data..."):
    df = get_climate_data()

data_hash = hash(tuple(df.columns.tolist() + [len(df)]))

with st.spinner("Building 5-level STING hierarchy..."):
    analyzer = get_analyzer(data_hash)

if "query_run" not in st.session_state:
    st.session_state["query_run"] = False
    st.session_state["query_params"] = {}

if run_btn or not st.session_state["query_run"]:
    t0 = time.perf_counter()
    relevant_cells, all_cells = analyzer.query(
        temp_min=temp_min,
        temp_max=temp_max,
        level=selected_level,
        min_cluster_size=min_cluster_size,
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000

    all_cells_df     = analyzer.level_summary(selected_level)
    cluster_stats    = analyzer.cluster_statistics(selected_level)
    n_clusters       = int(all_cells_df["cluster_id"].max() + 1) \
                       if not all_cells_df.empty and "cluster_id" in all_cells_df else 0

    st.session_state.update({
        "query_run": True,
        "all_cells_df": all_cells_df,
        "cluster_stats": cluster_stats,
        "n_relevant": len(relevant_cells),
        "n_total": len(all_cells),
        "n_clusters": n_clusters,
        "elapsed_ms": elapsed_ms,
        "selected_level": selected_level,
        "temp_min": temp_min,
        "temp_max": temp_max,
    })

all_cells_df  = st.session_state.get("all_cells_df", pd.DataFrame())
cluster_stats = st.session_state.get("cluster_stats", pd.DataFrame())
n_relevant    = st.session_state.get("n_relevant", 0)
n_total       = st.session_state.get("n_total", 0)
n_clusters    = st.session_state.get("n_clusters", 0)
elapsed_ms    = st.session_state.get("elapsed_ms", 0)

c1, c2, c3, c4, c5, c6 = st.columns(6)
metrics = [
    (c1, f"{len(df):,}",          "Weather Stations",   "#4fc3f7"),
    (c2, f"{n_total:,}",          "Grid Cells (Level)", "#7c4dff"),
    (c3, f"{n_relevant:,}",       "Relevant Cells",     "#56d364"),
    (c4, f"{n_clusters}",         "Clusters Found",     "#ff7043"),
    (c5, f"{elapsed_ms:.1f} ms",  "Query Time",         "#ffd54f"),
    (c6, f"L{selected_level}",    "Active Level",       "#4fc3f7"),
]
for col, val, lbl, clr in metrics:
    with col:
        st.markdown(f"""
        <div class='metric-card'>
          <div class='metric-val' style='color:{clr}'>{val}</div>
          <div class='metric-lbl'>{lbl}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

map_col, chart_col = st.columns([3, 2], gap="medium")

with map_col:
    st.markdown(f"""
    #### 🗺️ Global Temperature Grid
    <span class='badge badge-blue'>Level {selected_level}: {LEVEL_CONFIG[selected_level][2]}</span>
    <span class='badge badge-red'>T ∈ [{temp_min}°C, {temp_max}°C]</span>
    <span class='badge badge-green'>{n_clusters} cluster(s)</span>
    """, unsafe_allow_html=True)

    if not all_cells_df.empty:
        deck = build_pydeck(
            all_cells_df=all_cells_df,
            stations_df=df,
            show_clusters=show_clusters,
            show_heatmap=show_heatmap,
            map_style=map_style,
        )
        st.pydeck_chart(deck, use_container_width=True)
    else:
        st.info("No populated cells at this level. Try a coarser level.")

    st.markdown("""
    <div style='display:flex;align-items:center;gap:8px;margin-top:6px;font-size:0.75rem;color:#6e7681;font-family:Space Mono,monospace'>
      <span>−40°C</span>
      <div style='flex:1;height:8px;border-radius:4px;
        background:linear-gradient(to right,#1e6eb5,#90caf9,#fff,#ff7043,#b71c1c)'></div>
      <span>+50°C</span>
    </div>
    """, unsafe_allow_html=True)


with chart_col:
    tab1, tab2, tab3 = st.tabs(["📊 Distribution", "🔵 Clusters", "📐 Hierarchy"])

    with tab1:
        if not all_cells_df.empty:
            st.plotly_chart(
                plot_temperature_distribution(all_cells_df),
                use_container_width=True, config={"displayModeBar": False}
            )
        else:
            st.info("No data available.")

    with tab2:
        if not cluster_stats.empty:
            st.plotly_chart(
                plot_cluster_summary(cluster_stats),
                use_container_width=True, config={"displayModeBar": False}
            )
            st.markdown("**Cluster Table**")
            st.dataframe(
                cluster_stats.style.format({
                    "mean_temp": "{:.1f}°C",
                    "max_temp": "{:.1f}°C",
                    "min_temp": "{:.1f}°C",
                    "mean_confidence": "{:.2f}",
                }),
                use_container_width=True, hide_index=True,
            )
        else:
            st.info("No clusters for current query. Try widening the temperature range.")

    with tab3:
        st.plotly_chart(
            plot_level_stats(analyzer),
            use_container_width=True, config={"displayModeBar": False}
        )
        st.markdown("**Level Configuration**")
        level_df = pd.DataFrame([
            {
                "Level": f"L{i}",
                "Label": LEVEL_CONFIG[i][2],
                "Grid": f"{LEVEL_CONFIG[i][0]}×{LEVEL_CONFIG[i][1]}",
                "Cell Size": f"{180/LEVEL_CONFIG[i][0]:.1f}°×{360/LEVEL_CONFIG[i][1]:.1f}°",
                "Min Count": MIN_COUNT_THRESHOLDS[i],
                "Temp Tol": f"±{TEMP_TOLERANCE[i]}°C",
            }
            for i in range(5)
        ])
        st.dataframe(level_df, use_container_width=True, hide_index=True)

st.markdown("---")
with st.expander(f"📋 Relevant Cells Detail ({n_relevant} cells)", expanded=False):
    rel_df = all_cells_df[all_cells_df.get("is_relevant", False) == True] \
             if "is_relevant" in all_cells_df.columns else all_cells_df
    if rel_df.empty:
        st.info("No relevant cells found for this query.")
    else:
        display_cols = ["level", "lat_min", "lat_max", "lon_min", "lon_max",
                        "count", "temp_mean", "temp_max", "temp_min",
                        "temp_std", "cluster_id", "confidence"]
        display_cols = [c for c in display_cols if c in rel_df.columns]
        st.dataframe(
            rel_df[display_cols].sort_values("cluster_id").style.format({
                "temp_mean": "{:.2f}",
                "temp_max": "{:.2f}",
                "temp_min": "{:.2f}",
                "temp_std": "{:.2f}",
                "lat_min": "{:.1f}",
                "lat_max": "{:.1f}",
                "lon_min": "{:.1f}",
                "lon_max": "{:.1f}",
                "confidence": "{:.3f}",
            }),
            use_container_width=True, height=320, hide_index=True,
        )

st.markdown("""
<div style='text-align:center;color:#3d4451;font-size:0.72rem;
  font-family:Space Mono,monospace;margin-top:24px'>
  STING Algorithm · GHCN Dataset · Built with Streamlit + PyDeck
</div>
""", unsafe_allow_html=True)