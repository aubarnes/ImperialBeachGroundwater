"""
Load and prepare the reference evapotranspiration (ETo) data for use in the Pastas model
Data from CIMIS (California Irrigation Management Information System)
Website: https://cimis.water.ca.gov/WSNReportCriteria.aspx
Station data for San Diego II (station 184) 
Spatial gridded data (2km res) available from 2/20/2003-Present
Used address/coords: 337 Caspian Way, Imperial Beach, CA 91932, United States(32.5756, -117.1262)

October 2024
Austin Barnes
"""
#%% Imports
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib
import numpy as np
import pickle

## Paths for loading data
path_to_ETo_spatial = '../data/ETo_spatial_2003-2024.csv'
path_to_cmip6_ensemble = '../data/cmip6_ensemble.pkl'

## Paths for saving data
path_to_ETo_2100 = '../data/ETo_2100.h5'

#%% Load evapotranspiration SPATIAL data (mm/day)
## Spatial gridded data (2km res) available from 2/20/2003-Present
## Used address/coords: 337 Caspian Way, Imperial Beach, CA 91932, United States(32.5756, -117.1262)
## NOTE: 2003 has many '0' values that I believe are missing, suggest starting 11/24/03 or later

SD_ET = pd.read_csv(path_to_ETo_spatial, header=0)
## Create new column 'timestamp' in SD_ET dataframe by converting 'DateTimeStamp' column to datetime
SD_ET['timestamp'] = pd.to_datetime(SD_ET['Date'])
## Set 'timestamp' column as index
SD_ET.set_index('timestamp', inplace=True)
## create new dataframe just for the evapotranspiration data
ETo_1day = SD_ET['ETo (mm/day)'] # ['PM ETo (mm)']
## Rename the column to 'ETo'
ETo_1day = ETo_1day.rename('ETo')
## Make ETo index a datetime index
ETo_1day.index = pd.to_datetime(ETo_1day.index)

## Truncate evapotranspiration data to start on 2003-10-01 and end on 2024-09-30
ETo_1day = ETo_1day.loc['2003-10-01':'2024-09-30']

## Save the evapotranspiration data to an HDF5 file
ETo_1day.to_hdf(path_to_ETo_obs, key='ETo_obs')

#%% Plot the evapotranspiration data
%matplotlib qt
plt.figure()
ETo_1day.plot()
plt.ylabel('ETo (mm/day)')
plt.title('Evapotranspiration Data')
plt.show()
#%% Load the CMIP6 ensemble data
with open(path_to_cmip6_ensemble, 'rb') as f:
    cmip6_ensemble = pickle.load(f)

#%% Plot the evapotrans from spatial historical data and the CMIP6 ensemble
## Historical data
%matplotlib qt
plt.figure()
plt.plot(ETo_1day,label='Historical ETo')
## CMIP6 ensemble data
for key in cmip6_ensemble.keys():
    ## Convert time to timestamp
    timestamps = pd.to_datetime(cmip6_ensemble[key]['time'])
    plt.plot(timestamps, cmip6_ensemble[key]['etrans_sfc'], label=key)

plt.ylabel('ETo (mm/day)')
plt.title('Historical ETo and CMIP6 Ensemble')
plt.legend()
plt.show()

#%% Create input time series from 2003-10-01 to 2100-08-31
## Historical Observations through 2024-09-30
## 8 Unique CMIP6 Dynamically Downscaled Models out to 2100-08-31
## (1 month earlier than a full water year b/c of end of CMIP6 data)
## Each ET from CMIP6 is scaled by a unique factor for each model based on
## Ratio of CMIP6 ET from 2014-09-01 to 2024-09-30 to the historical ET from 2014-09-01 to 2024-09-30

ETo_data = ETo_1day

## Historical data
ETo_hist = ETo_data.loc['2003-10-01':'2024-09-30']

## CMIP6 data
cmip6_time = pd.to_datetime(cmip6_ensemble[list(cmip6_ensemble.keys())[1]]['time'])
ETo_cmip6 = pd.DataFrame(index=cmip6_time)
ETo_cmip6_scaled = pd.DataFrame(index=cmip6_time)
for key in cmip6_ensemble.keys():
    ## Extract the ET data
    etrans_sfc = cmip6_ensemble[key]['etrans_sfc']
    ## If length of etrans_sfc < 31411 (days beteen 2014-09-01 and 2100-08-31), pad with the last value
    ## Does not matter - we will not be using any values in calendar year 2100
    if len(etrans_sfc) < 31411:
        etrans_sfc = np.pad(etrans_sfc, (0, 31411 - len(etrans_sfc)), 'mean')
    cmip6_series = pd.Series(etrans_sfc, index=cmip6_time)
    ETo_cmip6[key] = etrans_sfc

    ## Scale the ET data
    ## Calculate the scale factor that would give the best R^2 between the historical and CMIP6 data from 2014-09-01 to 2024-09-30
    obs2014 = ETo_hist.loc['2014-09-01':'2024-09-30']
    cmip2014 = cmip6_series.loc['2014-09-01':'2024-09-30']

    scale_factor = np.sum(obs2014 * cmip2014) / np.sum(cmip2014**2)
    scale_factor_mean = np.mean(obs2014) / np.mean(cmip2014)
    ETo_cmip6_scaled[key] = etrans_sfc * scale_factor
    print(f"Scale factor for {key}: {scale_factor}")
    print(f"R^2: {np.corrcoef(obs2014, cmip2014 * scale_factor)[0,1]**2}")
    print(f"Mean scale factor: {scale_factor_mean}")

    # ## Plot the scaled data
    # plt.figure()
    # plt.plot(obs2014, label='Historical')
    # plt.plot(ETo_cmip6_scaled[key].index,ETo_cmip6_scaled[key], label='Scaled CMIP6')
    # plt.ylabel('ETo (mm/day)')
    # plt.title(key)
    # plt.legend()
    # plt.show()

#%% Find the linear trends for each of the scaled CMIP6 models
## Calculate the linear trend for each of the scaled CMIP6 models
trends = {}
for key in ETo_cmip6_scaled.keys():
    ## Calculate the linear trend
    x = np.arange(len(ETo_cmip6_scaled[key]))
    y = ETo_cmip6_scaled[key]
    m, b = np.polyfit(x, y, 1)
    trends[key] = m
    print(f"Linear trend for {key}: {m}")

#%% Concatenate the historical data 2003-10-01 to 2024-09-31 with each of the scaled CMIP6 models from 2024-10-01 to 2100-08-31
## Create a new dataframe
ETo_2100 = pd.DataFrame(index=pd.date_range(start='2003-10-01', end='2100-08-31', freq='D'))
## Create 8 columns
for key in ETo_cmip6_scaled.keys():
    ETo_2100[key] = np.nan  # Initialize with NaNs
    ETo_2100.loc['2003-10-01':'2024-09-30', key] = ETo_hist.values  # Copy historical data
    ETo_2100.loc['2024-10-01':'2100-08-31', key] = ETo_cmip6_scaled.loc['2024-10-01':'2100-08-31', key].values  # Copy scaled CMIP6 data


#%% Plot ETo_2100
%matplotlib qt
ETo_2100.plot()
plt.ylabel('ETo (mm/day)')
plt.title('Evapotranspiration 2100')
plt.show()

#%% Save the evapotranspiration data to an HDF5 file
ETo_2100.to_hdf(path_to_ETo_2100, key='ETo_2100')
# %%
