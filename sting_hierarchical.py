import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_ghcn_stations(file_path='ghcnd-stations.txt'):
    # loads station metadata and simulates temperature
    col_specs = [(0, 11), (12, 21), (21, 31)]
    col_names = ['station_id', 'latitude', 'longitude']
    df = pd.read_fwf(file_path, colspecs=col_specs, names=col_names)
    
    # simulate temperature based on latitude
    df['temp'] = 25 - 0.5 * np.abs(df['latitude']) + np.random.normal(0, 5, len(df))
    return df

class STINGHierarchy:
    # models a hierarchical Statistical Information Grid
    def __init__(self, df, levels=[30, 25, 20, 10, 5]):
        self.df = df
        self.levels = sorted(levels, reverse=True)
        self.grids = {}
        self._build_hierarchy()
        
    def _build_hierarchy(self):
        for res in self.levels:
            # grid grouping
            self.df['lat_grid'] = np.floor(self.df['latitude'] / res) * res
            self.df['lon_grid'] = np.floor(self.df['longitude'] / res) * res
            
            stats = self.df.groupby(['lat_grid', 'lon_grid']).agg(
                mean_temp=('temp', 'mean'),
                min_temp=('temp', 'min'),
                max_temp=('temp', 'max'),
                std_temp=('temp', 'std'),
                count=('temp', 'count')
            ).reset_index()
            
            # filling NaN for std_temp when count is 1
            stats['std_temp'] = stats['std_temp'].fillna(0.0)
            
            self.grids[res] = stats
            print(f"\n--- Grid Stats Sample ({res}° Resolution) ---")
            print(stats.head(5).to_string(index=True))
            print(f"Total Clusters: {len(stats)}")

def plot_hierarchy(hierarchy):
    # saves individual scatter plots for each resolution level
    for res in hierarchy.levels:
        grid = hierarchy.grids[res]
        plt.figure(figsize=(10, 6))
        
        scatter = plt.scatter(
            grid['lon_grid'], grid['lat_grid'], 
            c=grid['mean_temp'], cmap='magma', s=40, edgecolors='none'
        )
        plt.title(f'Resolution: {res}° (STING)')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.colorbar(scatter, label='Mean Temp')
        
        filename = f'sting_{res}deg.png'
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()
        print(f"Plot saved as '{filename}'")

if __name__ == "__main__":
    print("Initializing Hierarchical STING Analysis...")
    try:
        data = load_ghcn_stations()
        sting = STINGHierarchy(data)
        plot_hierarchy(sting)
    except Exception as e:
        print(f"Error: {e}")
