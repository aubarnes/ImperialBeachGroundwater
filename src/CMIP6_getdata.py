'''
Cal-Adapt Analytics Engine Data Catalog and Access
--------------------------------------------------
CMIP 6 dynamically downscaled (WRF) Evapotranspiration and Precipitation Data Extraction
Author: Austin Barnes, 2024
This script is used to download CMIP6 data from the cloud using the intake-esm library.
The data is then processed to extract the evapotranspiration and precipitation data for a specific grid point.
The data is saved to a dictionary and then saved to a file.
'''
#%%
import intake
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pylab as P
%matplotlib

## Outputs
path_to_cmip6_ensemble = '../data/cmip6_ensemble.pkl'

#%%
cat = intake.open_esm_datastore('https://cadcat.s3.amazonaws.com/cae-collection.json')
cat
#%%
cat.df.head()
#%%
# Show the unique column values which are useful to query on.
# cat.unique(columns=set(cat.df.columns) - {'path'})
cat.unique()
#%%
cat.df['source_id'].unique()

#%%
cat.df['experiment_id'].unique()

#%%
cat.df['table_id'].unique()
#%%
cat.df['variable_id'].unique()
#%%
# Use a dictionary of column names and values to query for datasets.
query = {
    'activity_id': 'WRF', # Downscaling method
    'institution_id': 'UCLA', # UCSD, CAE, or UCLA
    # Uncomment this to search for an individual GCM, otherwise all are loaded.
    #'source_id': 'CESM2',
    # 'experiment_id': ['historical', 'ssp370'], # time period - historical or emissions scenario
    'experiment_id': ['ssp370'], # time period - historical or emissions scenario
    'variable_id': ['prec','etrans_sfc'], # variable
    # 'table_id': '1hr', # hourly time resolution
    'table_id': 'day', # daily time resolution
    'grid_label': 'd03' # grid resolution: d01 = 45km, d02 = 9km, d03 = 3km
}
# subset catalog and get some metrics grouped by 'source_id'
cat_subset = cat.search(require_all_on=['source_id'], **query)
cat_subset

#%%
# Open each zarr store in a dictionary of Xarray Dataset objects.
dsets = cat_subset.to_dataset_dict(zarr_kwargs={'consolidated': True}, 
                                   storage_options={'anon': True})

#%%
# List all the dataset keys.
list(dsets)

## Create an array of the keys for looping
keys = list(dsets)

#%% Define Latitude and Longitude of Imperial Beach Point of Interest
## Matches the lat/lon of the historical evapotranspiration grid point and location of precipitation gauge
lat_ib = 32.5756
lon_ib = -117.1262
#%% Loop through each dataset to verify that the same grid point is selected

## Create a dictionary to store the time, etrans, and prec data for each dataset
data_dict = {}

for key in keys:
    ds = dsets[key]

    # Calculate the distance to the target lat/lon for each grid point
    distances = np.sqrt((ds['lat'] - lat_ib)**2 + (ds['lon'] - lon_ib)**2)

    # Find the indices of the minimum distance
    min_dist_idx = np.unravel_index(np.argmin(distances.values), distances.shape)

    ## Print the indices of the minimum distance
    print(f"Dataset: {key}")
    print(f"Minimum distance index: {min_dist_idx}")
    
    selected_lat = ds['lat'].isel(y=min_dist_idx[0], x=min_dist_idx[1])
    selected_lon = ds['lon'].isel(y=min_dist_idx[0], x=min_dist_idx[1])
    print(f"IB grid point: lat={lat_ib}, lon={lon_ib}")
    print(f"Selected grid point: lat={selected_lat.values}, lon={selected_lon.values}")

    time_data = ds['time'].values
    print(f"Dims of time data: {time_data.shape}")
    print(f"Time data start: {time_data[0]}")
    print(f"Time data end: {time_data[-1]}")

    ## Extract the data for the closest grid point
    etrans_sfc_data = ds['etrans_sfc'].isel(y=min_dist_idx[0], x=min_dist_idx[1]).values
    prec_data = ds['prec'].isel(y=min_dist_idx[0], x=min_dist_idx[1]).values
    ## Reduce dimensions
    etrans_sfc_data = np.squeeze(etrans_sfc_data)
    prec_data = np.squeeze(prec_data)

    ## Save time, etrans, and prec data from each dataset
    data_dict[key] = {'time': time_data, 'etrans_sfc': etrans_sfc_data, 'prec': prec_data}

# %%
## Between dataset 1 and 2, find the time entries that are different
time1 = data_dict[keys[0]]['time']
time2 = data_dict[keys[1]]['time']

unique_to_time1 = list(set(time1) - set(time2))

# Find values in time2 that are not in time1
unique_to_time2 = list(set(time2) - set(time1))

# Display the unique values
print("Values in time1 but not in time2:", unique_to_time1)
print("Values in time2 but not in time1:", unique_to_time2)

#%%
## Save the data dictionary to a file
import pickle

with open(path_to_cmip6_ensemble, 'wb') as f:
    pickle.dump(data_dict, f)

# %%
