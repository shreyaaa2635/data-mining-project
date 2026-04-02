"""
run_analysis.py — CLI runner for STING (no UI required)

Usage:
  python run_analysis.py --temp-min -5 --temp-max 10 --level 2
"""

import argparse
import time
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from data.ghcn_loader import load_climate_data
from sting.sting_algorithm import STINGAnalyzer, LEVEL_CONFIG


def main():
    parser = argparse.ArgumentParser(description="STING Climate Analyzer CLI")
    parser.add_argument("--temp-min", type=float, default=-5.0)
    parser.add_argument("--temp-max", type=float, default=10.0)
    parser.add_argument("--level", type=int, default=2, choices=range(5))
    parser.add_argument("--min-cluster-size", type=int, default=1)
    args = parser.parse_args()

    print("\n══════════════════════════════════════════════")
    print("  STING Global Climate Analysis")
    print("══════════════════════════════════════════════")

    print("\n[1/3] Loading climate data...")
    t0 = time.time()
    df = load_climate_data()
    print(f"      Loaded {len(df):,} stations in {time.time()-t0:.2f}s")

    print("\n[2/3] Building 5-level STING hierarchy...")
    t0 = time.time()
    analyzer = STINGAnalyzer(df)
    analyzer.build_hierarchy()
    print(f"      Hierarchy built in {time.time()-t0:.2f}s")

    print(f"\n[3/3] Running query:")
    print(f"      Level   = {args.level} ({LEVEL_CONFIG[args.level][2]})")
    print(f"      T range = [{args.temp_min}°C, {args.temp_max}°C]")

    t0 = time.time()
    relevant, all_cells = analyzer.query(
        temp_min=args.temp_min,
        temp_max=args.temp_max,
        level=args.level,
        min_cluster_size=args.min_cluster_size,
    )
    print(f"      Query completed in {(time.time()-t0)*1000:.1f}ms")

    print(f"\n── Results ────────────────────────────────────")
    print(f"  Total populated cells : {len(all_cells)}")
    print(f"  Relevant cells        : {len(relevant)}")

    cluster_stats = analyzer.cluster_statistics(args.level)
    if not cluster_stats.empty:
        print(f"  Clusters found        : {len(cluster_stats)}")
        print(f"\n  Cluster Detail:")
        print(cluster_stats.to_string(index=False))
    else:
        print("  Clusters found        : 0")

    print("\n── Hierarchy Summary ──────────────────────────")
    for lvl in range(5):
        df_lvl = analyzer.level_summary(lvl)
        n_lat, n_lon, lbl = LEVEL_CONFIG[lvl]
        coverage = len(df_lvl) / (n_lat * n_lon) * 100
        mean_t = df_lvl["temp_mean"].mean() if not df_lvl.empty else float("nan")
        print(f"  L{lvl} {lbl:<25} "
              f"cells={len(df_lvl):>5}  "
              f"coverage={coverage:5.1f}%  "
              f"mean_T={mean_t:+.1f}°C")

    print("\n══════════════════════════════════════════════\n")


if __name__ == "__main__":
    main()