

import numpy as np
import pandas as pd
import pydeck as pdk
import plotly.graph_objects as go
import plotly.express as px
from typing import List, Optional
from .grid_cell import STINGCell

def _temp_to_rgba(temp: float, vmin: float = -30, vmax: float = 40,
                  alpha: int = 180) -> List[int]:

    if np.isnan(temp):
        return [128, 128, 128, 80]
    t = np.clip((temp - vmin) / max(vmax - vmin, 1e-6), 0, 1)

    if t < 0.5:
        s = t * 2
        r = int(30 + 220 * s)
        g = int(100 + 155 * s)
        b = int(220 - 100 * s)
    else:
        s = (t - 0.5) * 2
        r = int(250)
        g = int(255 - 200 * s)
        b = int(120 - 100 * s)

    return [r, g, b, alpha]


def _cluster_color(cluster_id: int, alpha: int = 200) -> List[int]:
    palette = [
        [255, 87, 51],   # Vivid red
        [46, 196, 182],  # Teal
        [255, 209, 102], # Amber
        [106, 76, 147],  # Purple
        [61, 220, 132],  # Green
        [255, 140, 0],   # Orange
        [0, 149, 255],   # Blue
        [255, 73, 148],  # Pink
        [100, 210, 80],  # Lime
        [180, 120, 60],  # Brown
    ]
    if cluster_id < 0:
        return [60, 60, 60, 60]
    return palette[cluster_id % len(palette)] + [alpha]


def build_all_cells_layer(cells_df: pd.DataFrame, vmin: float = -30,
                           vmax: float = 40) -> pdk.Layer:
    """Polygon layer for ALL populated cells coloured by temperature."""
    if cells_df.empty:
        return pdk.Layer("PolygonLayer", data=[], get_polygon="polygon")

    cells_df = cells_df.copy()
    cells_df["fill_color"] = cells_df["temp_mean"].apply(
        lambda t: _temp_to_rgba(t, vmin, vmax, alpha=150)
    )

    def make_polygon(row):
        lo, la = row["lon_min"], row["lat_min"]
        hi_lo, hi_la = row["lon_max"], row["lat_max"]
        return [[lo, la], [hi_lo, la], [hi_lo, hi_la], [lo, hi_la], [lo, la]]

    cells_df["polygon"] = cells_df.apply(make_polygon, axis=1)

    return pdk.Layer(
        "PolygonLayer",
        data=cells_df,
        get_polygon="polygon",
        get_fill_color="fill_color",
        get_line_color=[80, 80, 80, 60],
        line_width_min_pixels=0.3,
        pickable=True,
        filled=True,
        stroked=True,
    )


def build_cluster_layer(cells_df: pd.DataFrame) -> pdk.Layer:
    """Polygon layer for RELEVANT clustered cells coloured by cluster ID."""
    rel = cells_df[cells_df["is_relevant"] == True].copy()
    if rel.empty:
        return pdk.Layer("PolygonLayer", data=[], get_polygon="polygon")

    rel["fill_color"] = rel["cluster_id"].apply(
        lambda cid: _cluster_color(cid, 210)
    )

    def make_polygon(row):
        lo, la = row["lon_min"], row["lat_min"]
        hi_lo, hi_la = row["lon_max"], row["lat_max"]
        return [[lo, la], [hi_lo, la], [hi_lo, hi_la], [lo, hi_la], [lo, la]]

    rel["polygon"] = rel.apply(make_polygon, axis=1)

    return pdk.Layer(
        "PolygonLayer",
        data=rel,
        get_polygon="polygon",
        get_fill_color="fill_color",
        get_line_color=[255, 255, 255, 120],
        line_width_min_pixels=0.8,
        pickable=True,
        filled=True,
        stroked=True,
    )


def build_centroid_layer(cells_df: pd.DataFrame) -> pdk.Layer:
    """Scatter layer showing cell centroids scaled by data count."""
    rel = cells_df[cells_df.get("is_relevant", False) == True].copy() \
        if "is_relevant" in cells_df.columns else cells_df.copy()
    if rel.empty:
        return pdk.Layer("ScatterplotLayer", data=[], get_position="position")

    rel["position"] = list(zip(rel["lon_center"], rel["lat_center"]))
    rel["radius"] = (np.sqrt(rel["count"].clip(1)) * 8000).clip(5000, 200000)
    rel["color"] = rel["cluster_id"].apply(lambda c: _cluster_color(c, 180)) \
        if "cluster_id" in rel.columns else [[255,200,0,180]] * len(rel)

    return pdk.Layer(
        "ScatterplotLayer",
        data=rel,
        get_position="position",
        get_radius="radius",
        get_fill_color="color",
        get_line_color=[255, 255, 255, 200],
        line_width_min_pixels=1,
        pickable=True,
    )


def build_heatmap_layer(stations_df: pd.DataFrame) -> pdk.Layer:
    """Heatmap layer directly from raw station points."""
    df = stations_df[["lon", "lat", "temp_mean"]].dropna().copy()
    df["position"] = list(zip(df["lon"], df["lat"]))
    df["weight"] = ((df["temp_mean"] - df["temp_mean"].min()) /
                    (df["temp_mean"].max() - df["temp_mean"].min() + 1e-6)).clip(0, 1)
    return pdk.Layer(
        "HeatmapLayer",
        data=df,
        get_position="position",
        get_weight="weight",
        radiusPixels=30,
        intensity=1,
        threshold=0.05,
        aggregation="MEAN",
    )


def build_pydeck(
    all_cells_df: pd.DataFrame,
    stations_df: Optional[pd.DataFrame] = None,
    show_clusters: bool = True,
    show_heatmap: bool = False,
    map_style: str = "dark",
) -> pdk.Deck:
    """Compose all layers into a PyDeck deck."""

    vmin = all_cells_df["temp_min"].min() if not all_cells_df.empty else -30
    vmax = all_cells_df["temp_max"].max() if not all_cells_df.empty else 40

    layers = []

    if show_heatmap and stations_df is not None:
        layers.append(build_heatmap_layer(stations_df))

    layers.append(build_all_cells_layer(all_cells_df, vmin, vmax))

    if show_clusters:
        layers.append(build_cluster_layer(all_cells_df))

    STYLES = {
        "dark": "mapbox://styles/mapbox/dark-v10",
        "satellite": "mapbox://styles/mapbox/satellite-streets-v11",
        "light": "mapbox://styles/mapbox/light-v10",
        "terrain": "mapbox://styles/mapbox/outdoors-v11",
    }

    return pdk.Deck(
        layers=layers,
        initial_view_state=pdk.ViewState(
            latitude=20,
            longitude=10,
            zoom=1.4,
            pitch=30,
            bearing=0,
        ),
        map_style=STYLES.get(map_style, STYLES["dark"]),
        tooltip={
            "html": """
                <div style='font-family:monospace;font-size:12px;padding:6px'>
                  <b>Level {level}</b><br/>
                  Lat: [{lat_min:.1f}, {lat_max:.1f}]°<br/>
                  Lon: [{lon_min:.1f}, {lon_max:.1f}]°<br/>
                  Stations: {count}<br/>
                  T̄ = {temp_mean:.1f}°C<br/>
                  T_max = {temp_max:.1f}°C<br/>
                  T_min = {temp_min:.1f}°C<br/>
                  Cluster: {cluster_id}
                </div>
            """,
            "style": {
                "backgroundColor": "#0d1117",
                "color": "#c9d1d9",
                "border": "1px solid #30363d",
                "borderRadius": "6px",
            },
        },
    )

def plot_temperature_distribution(cells_df: pd.DataFrame) -> go.Figure:
    """Histogram of mean temperatures across cells."""
    df = cells_df.dropna(subset=["temp_mean"])
    fig = px.histogram(
        df, x="temp_mean", nbins=50,
        color_discrete_sequence=["#4fc3f7"],
        labels={"temp_mean": "Mean Temperature (°C)"},
        title="Temperature Distribution Across Grid Cells",
    )
    fig.update_layout(
        plot_bgcolor="#0d1117",
        paper_bgcolor="#0d1117",
        font_color="#c9d1d9",
        xaxis=dict(gridcolor="#21262d"),
        yaxis=dict(gridcolor="#21262d"),
        bargap=0.05,
    )
    return fig


def plot_cluster_summary(cluster_stats: pd.DataFrame) -> go.Figure:
    """Bar chart of cluster sizes and mean temperatures."""
    if cluster_stats.empty:
        return go.Figure()

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=cluster_stats["cluster_id"].astype(str),
        y=cluster_stats["total_stations"],
        name="Stations",
        marker_color="#4fc3f7",
        yaxis="y",
    ))
    fig.add_trace(go.Scatter(
        x=cluster_stats["cluster_id"].astype(str),
        y=cluster_stats["mean_temp"],
        name="Mean Temp (°C)",
        mode="lines+markers",
        line=dict(color="#ff7043", width=2),
        marker=dict(size=8),
        yaxis="y2",
    ))
    fig.update_layout(
        title="Cluster Summary",
        plot_bgcolor="#0d1117",
        paper_bgcolor="#0d1117",
        font_color="#c9d1d9",
        yaxis=dict(title="Total Stations", gridcolor="#21262d"),
        yaxis2=dict(title="Mean Temperature (°C)", overlaying="y",
                    side="right", gridcolor="#21262d"),
        xaxis=dict(title="Cluster ID", gridcolor="#21262d"),
        legend=dict(bgcolor="#161b22", bordercolor="#30363d"),
    )
    return fig


def plot_level_stats(analyzer) -> go.Figure:
    """Line chart: cells and coverage across hierarchy levels."""
    records = []
    for lvl_idx, label in enumerate(analyzer.level_labels):
        df = analyzer.level_summary(lvl_idx)
        records.append({
            "level": f"L{lvl_idx}: {label}",
            "cell_count": len(df),
            "mean_temp": df["temp_mean"].mean() if not df.empty else np.nan,
            "coverage_pct": len(df) / (
                analyzer.level_configs[lvl_idx][0] *
                analyzer.level_configs[lvl_idx][1]
            ) * 100,
        })
    df_rec = pd.DataFrame(records)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df_rec["level"], y=df_rec["cell_count"],
        name="Populated Cells", marker_color="#7c4dff",
    ))
    fig.add_trace(go.Scatter(
        x=df_rec["level"], y=df_rec["coverage_pct"],
        name="Coverage %", mode="lines+markers",
        line=dict(color="#69f0ae", width=2),
        marker=dict(size=8),
        yaxis="y2",
    ))
    fig.update_layout(
        title="STING Hierarchy — Cells per Level",
        plot_bgcolor="#0d1117",
        paper_bgcolor="#0d1117",
        font_color="#c9d1d9",
        yaxis=dict(title="Populated Cells", gridcolor="#21262d"),
        yaxis2=dict(title="Coverage (%)", overlaying="y",
                    side="right", gridcolor="#21262d"),
        xaxis=dict(gridcolor="#21262d"),
        legend=dict(bgcolor="#161b22"),
    )
    return fig