
"""
data/ghcn_loader.py
Loads and preprocesses local GHCN-M v1 data from ghcn-m-v1.csv only.
No online or synthetic fallback.
"""

def ensure_cache_dir():
    pass  # No cache needed


import pandas as pd
import re
import os

def _parse_lat_band(lat_band):
    # Example: '85-90N', '0-5S'
    match = re.match(r"([\-\d]+)-([\-\d]+)([NS])", lat_band)
    if not match:
        raise ValueError(f"Invalid latitude band: {lat_band}")
    low, high, hemi = match.groups()
    low, high = float(low), float(high)
    # Use midpoint
    val = (low + high) / 2.0
    if hemi == 'S':
        val = -val
    return val

def _parse_lon_band(col):
    # Example: 'lon_175_180W', 'lon_0_5E'
    match = re.match(r"lon_([\d]+)_([\d]+)([EW])", col)
    if not match:
        return None
    low, high, hemi = match.groups()
    low, high = float(low), float(high)
    val = (low + high) / 2.0
    if hemi == 'W':
        val = -val
    return val

def load_climate_data() -> pd.DataFrame:
    """
    Loads and preprocesses the local GHCN-M v1 dataset (ghcn-m-v1.csv).
    Returns a DataFrame with columns: year, month, lat, lon, value
    """
    csv_path = os.path.join(os.path.dirname(__file__), '..', 'ghcn-m-v1.csv')
    df = pd.read_csv(csv_path)

    # Identify longitude columns
    lon_cols = [col for col in df.columns if col.startswith('lon_')]

    records = []
    for _, row in df.iterrows():
        year = int(row['year'])
        month = int(row['month'])
        lat = _parse_lat_band(row['lat'])
        for col in lon_cols:
            val = row[col]
            if val == -9999 or pd.isna(val):
                continue
            lon = _parse_lon_band(col)
            if lon is None:
                continue
            records.append({
                'year': year,
                'month': month,
                'lat': lat,
                'lon': lon,
                'value': val
            })
    df_out = pd.DataFrame(records)
    # Add temp_mean column for STING compatibility
    df_out["temp_mean"] = df_out["value"]
    return df_out


if __name__ == "__main__":
    df = load_climate_data()
    print(df.head())
    print(f"Total records: {len(df)}")