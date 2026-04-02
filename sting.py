import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#load data
df = pd.read_csv("ghcn-m-v1.csv")

print("Columns:", df.columns)

#reshape data

id_vars = ['year', 'month', 'lat']
value_vars = [col for col in df.columns if col.startswith('lon_')]

df_melted = df.melt(
    id_vars=id_vars,
    value_vars=value_vars,
    var_name='lon_range',
    value_name='TEMP'
)

#extract longitudinal values
def extract_lon(lon_str):
    parts = lon_str.replace('lon_', '').replace('W', '').replace('E', '').split('_')
    mid = (float(parts[0]) + float(parts[1])) / 2
    
    if 'W' in lon_str:
        return -mid
    else:
        return mid

df_melted['lon'] = df_melted['lon_range'].apply(extract_lon)

#rename lat column
df_melted.rename(columns={'lat': 'LATITUDE'}, inplace=True)
df_melted.rename(columns={'lon': 'LONGITUDE'}, inplace=True)

#drop missing values
df_melted.dropna(inplace=True)

print("\nReshaped Data:")
print(df_melted.head())

def lat_to_mid(lat_str):
    lat_str = lat_str.replace('N','').replace('S','')
    parts = lat_str.split('-')
    mid = (float(parts[0]) + float(parts[1])) / 2
    if 'S' in lat_str:
        mid = -mid #southern hemisphere negative
    return mid

df_melted['LATITUDE'] = df_melted['LATITUDE'].apply(lat_to_mid)

#apply sting algorithm
grid_size = 5

df_melted['lat_grid'] = (df_melted['LATITUDE'] // grid_size) * grid_size
df_melted['lon_grid'] = (df_melted['LONGITUDE'] // grid_size) * grid_size

grid_stats = df_melted.groupby(['lat_grid', 'lon_grid']).agg(
    mean_temp=('TEMP', 'mean'),
    min_temp=('TEMP', 'min'),
    max_temp=('TEMP', 'max'),
    count=('TEMP', 'count')
).reset_index()

print("\nSTING Output:")
print(grid_stats.head())

#visualization
plt.figure(figsize=(10, 6))

scatter = plt.scatter(
    grid_stats['lon_grid'],
    grid_stats['lat_grid'],
    c=grid_stats['mean_temp'],
    cmap='coolwarm',
    s=50
)

plt.colorbar(scatter, label='Mean Temperature')
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("STING on Pre-Gridded Climate Data")

plt.show()

#save output
grid_stats.to_csv("sting_output.csv", index=False)

print("\nOutput saved.")