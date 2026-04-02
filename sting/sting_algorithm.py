import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from collections import deque
import logging

from .grid_cell import STINGCell

logger = logging.getLogger(__name__)

LEVEL_CONFIG = [
    # (lat_divisions, lon_divisions, label)
    (12,  24,  "Continental (15°)"),
    (24,  48,  "Regional (7.5°)"),
    (45,  90,  "Sub-Regional (4°)"),
    (90,  180, "National (2°)"),
    (180, 360, "Local (1°)"),
]

MIN_COUNT_THRESHOLDS = [3, 2, 2, 1, 1]

TEMP_TOLERANCE = [6.0, 4.5, 4.0, 3.0, 2.5]

NEIGHBOURS = [
    (-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1),
    (-2,0),(2,0),(0,-2),(0,2),(-2,-2),(-2,2),(2,-2),(2,2)
]


class STINGAnalyzer:

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.levels: List[List[List[Optional[STINGCell]]]] = []  
        self.level_configs = LEVEL_CONFIG
        self._built = False

    def build_hierarchy(self):
        """Partition data into all 5 grid levels and compute cell statistics."""
        logger.info("Building STING hierarchy (5 levels)...")
        self.levels = []

        for lvl_idx, (n_lat, n_lon, label) in enumerate(LEVEL_CONFIG):
            logger.info(f"  Level {lvl_idx}: {label} ({n_lat}×{n_lon} grid)")
            grid = self._build_level(lvl_idx, n_lat, n_lon)
            self.levels.append(grid)

        self._link_children()
        self._built = True
        logger.info("Hierarchy complete.")

    def _build_level(self, level: int, n_lat: int, n_lon: int) \
            -> List[List[Optional[STINGCell]]]:
        """Create n_lat × n_lon cells and populate with statistics."""

        lat_edges = np.linspace(-90, 90, n_lat + 1)
        lon_edges = np.linspace(-180, 180, n_lon + 1)

        lat_idx = np.searchsorted(lat_edges[1:], self.df["lat"].values, side="left")
        lon_idx = np.searchsorted(lon_edges[1:], self.df["lon"].values, side="left")
        lat_idx = np.clip(lat_idx, 0, n_lat - 1)
        lon_idx = np.clip(lon_idx, 0, n_lon - 1)

        grid: List[List[Optional[STINGCell]]] = [
            [None] * n_lon for _ in range(n_lat)
        ]
        self.df["_lat_idx"] = lat_idx
        self.df["_lon_idx"] = lon_idx

        grouped = self.df.groupby(["_lat_idx", "_lon_idx"])

        for (ri, ci), group in grouped:
            temps = group["temp_mean"].dropna().values
            if len(temps) == 0:
                continue

            cell = STINGCell(
                level=level,
                lat_min=float(lat_edges[ri]),
                lat_max=float(lat_edges[ri + 1]),
                lon_min=float(lon_edges[ci]),
                lon_max=float(lon_edges[ci + 1]),
                count=len(temps),
                temp_mean=float(np.mean(temps)),
                temp_max=float(group["temp_max"].max()) if "temp_max" in group else float(np.max(temps)),
                temp_min=float(group["temp_min"].min()) if "temp_min" in group else float(np.min(temps)),
                temp_std=float(np.std(temps)) if len(temps) > 1 else 0.0,
            )
            grid[ri][ci] = cell

        self.df.drop(columns=["_lat_idx", "_lon_idx"], inplace=True, errors="ignore")
        return grid

    def _link_children(self):
        """For each cell at level L, find its sub-cells at level L+1."""
        for lvl_idx in range(len(self.levels) - 1):
            parent_grid = self.levels[lvl_idx]
            child_grid  = self.levels[lvl_idx + 1]
            n_lat_c = len(child_grid)
            n_lon_c = len(child_grid[0])
            lat_step_c = 180.0 / n_lat_c
            lon_step_c = 360.0 / n_lon_c

            for row in parent_grid:
                for cell in row:
                    if cell is None:
                        continue

                    r_start = int((cell.lat_min + 90) / lat_step_c)
                    r_end   = int((cell.lat_max + 90) / lat_step_c)
                    c_start = int((cell.lon_min + 180) / lon_step_c)
                    c_end   = int((cell.lon_max + 180) / lon_step_c)

                    for cr in range(max(0, r_start), min(n_lat_c, r_end)):
                        for cc in range(max(0, c_start), min(n_lon_c, c_end)):
                            child = child_grid[cr][cc]
                            if child is not None:
                                cell.children.append(child)

    def query(
        self,
        temp_min: float,
        temp_max: float,
        level: int = 2,
        min_cluster_size: int = 1
    ) -> Tuple[List[STINGCell], List[STINGCell]]:
        if not self._built:
            raise RuntimeError("Call build_hierarchy() first.")

        grid = self.levels[level]
        min_count = MIN_COUNT_THRESHOLDS[level]
        query_center = (temp_min + temp_max) / 2
        tol = TEMP_TOLERANCE[level]

        # Mark relevant cells
        all_cells: List[STINGCell] = []
        relevant_cells: List[STINGCell] = []

        for row in grid:
            for cell in row:
                if cell is None:
                    continue
                cell.cluster_id = -1
                cell.is_relevant = False
                cell.confidence = 0.0
                all_cells.append(cell)

                if cell.count < min_count:
                    continue

                # Check temperature overlap
                cell_lo = cell.temp_mean - cell.temp_std
                cell_hi = cell.temp_mean + cell.temp_std

                overlap = (cell_lo <= temp_max) and (cell_hi >= temp_min)
                within_tol = abs(cell.temp_mean - query_center) <= tol

                if overlap or within_tol:
                    cell.is_relevant = True
                    # Confidence: inverse normalised distance from query center
                    dist = max(0.0, abs(cell.temp_mean - query_center) - (temp_max - temp_min) / 2)
                    cell.confidence = float(np.exp(-dist / max(tol, 0.5)))
                    relevant_cells.append(cell)

        # BFS clustering on relevant cells
        self._cluster_bfs(grid, level, min_cluster_size)

        return relevant_cells, all_cells

    def _cluster_bfs(self, grid, level: int, min_cluster_size: int):
        """BFS over adjacent relevant cells to assign cluster IDs."""
        n_lat = len(grid)
        n_lon = len(grid[0])

        # Build fast lookup: (ri, ci) → cell
        cell_map: Dict[Tuple[int,int], STINGCell] = {}
        for ri, row in enumerate(grid):
            for ci, cell in enumerate(row):
                if cell is not None and cell.is_relevant:
                    cell_map[(ri, ci)] = cell

        visited = set()
        cluster_id = 0

        for (ri, ci), cell in cell_map.items():
            if (ri, ci) in visited:
                continue

            # BFS
            queue = deque([(ri, ci)])
            cluster_members = []
            visited.add((ri, ci))

            while queue:
                r, c = queue.popleft()
                cluster_members.append((r, c))

                for dr, dc in NEIGHBOURS:
                    nr, nc = r + dr, c + dc
                    if (nr, nc) in cell_map and (nr, nc) not in visited:
                        visited.add((nr, nc))
                        queue.append((nr, nc))

            if len(cluster_members) >= min_cluster_size:
                for (r, c) in cluster_members:
                    cell_map[(r, c)].cluster_id = cluster_id
                cluster_id += 1

    def level_summary(self, level: int) -> pd.DataFrame:
        """Return a DataFrame summary of all populated cells at a level."""
        if not self._built:
            raise RuntimeError("Call build_hierarchy() first.")
        rows = []
        for row in self.levels[level]:
            for cell in row:
                if cell is not None and cell.is_populated():
                    rows.append(cell.to_dict())
        return pd.DataFrame(rows)

    def get_all_cells_df(self, level: int) -> pd.DataFrame:
        """All cells at a level as a flat DataFrame (for map rendering)."""
        return self.level_summary(level)

    def cluster_statistics(self, level: int) -> pd.DataFrame:
        """Aggregate stats per cluster at a given level."""
        df = self.level_summary(level)
        df = df[df["cluster_id"] >= 0]
        if df.empty:
            return pd.DataFrame()
        return df.groupby("cluster_id").agg(
            cell_count=("count", "size"),
            total_stations=("count", "sum"),
            mean_temp=("temp_mean", "mean"),
            max_temp=("temp_max", "max"),
            min_temp=("temp_min", "min"),
            mean_confidence=("confidence", "mean"),
        ).reset_index()

    @property
    def level_labels(self) -> List[str]:
        return [cfg[2] for cfg in LEVEL_CONFIG]