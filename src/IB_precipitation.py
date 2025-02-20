"""
Load and prepare the precipitation data for use in the Pastas model

Observed precipitation from the Tijuana River Estuary Rain Gauge
Basin precipitation for Otay Basin, Tijuana River Basin, and both combined
using spatially gridded data from Google Earth Engine

October 2024
Austin Barnes
Code to get precipitation data via Google Earth Engine from Morgan Levy
"""
#%% Imports
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib
import numpy as np
import pickle

## Paths for loading data
path_to_TJRTLMET = '/Users/austinbarnes/Documents/UCSD SIO/IB Groundwater/ImperialBeach/data/precip_IB/TJRTLMET_full.csv'
path_to_otaybasin_prcp = '/Users/austinbarnes/Documents/UCSD SIO/IB Groundwater/ImperialBeach/data/precip_watershed/otaybasin_prcp.csv'
path_to_tjbasin_prcp = '/Users/austinbarnes/Documents/UCSD SIO/IB Groundwater/ImperialBeach/data/precip_watershed/tjbasin_prcp.csv'
path_to_cmip6_ensemble = '/Users/austinbarnes/Documents/UCSD SIO/IB Groundwater/ImperialBeach/data/cmip6_ensemble.pkl'

## Paths for saving data
path_to_IB_precip = '/Users/austinbarnes/Documents/UCSD SIO/IB Groundwater/ImperialBeach/data/precip_IB/IB_precip.h5'
path_to_otaybasin_precip = '/Users/austinbarnes/Documents/UCSD SIO/IB Groundwater/ImperialBeach/data/precip_watershed/otaybasin_precip.h5'
path_to_tjbasin_precip = '/Users/austinbarnes/Documents/UCSD SIO/IB Groundwater/ImperialBeach/data/precip_watershed/tjbasin_precip.h5'
path_to_combinedbasin_precip = '/Users/austinbarnes/Documents/UCSD SIO/IB Groundwater/ImperialBeach/data/precip_watershed/combinedbasin_precip.h5'
path_to_precip_2100 = '/Users/austinbarnes/Documents/UCSD SIO/IB Groundwater/ImperialBeach/data/precip_IB/precip_2100.h5'

#%% Load and prepare TJRTLMET precipitation data

## Load TJ River Estuary IB Sensor Precipitation Data (Station)
TJRTLMET = pd.read_csv(path_to_TJRTLMET, header=0)
## Create new column 'timestamp' in TJRTLMET dataframe by converting 'DateTimeStamp' column to datetime
TJRTLMET['timestamp'] = pd.to_datetime(TJRTLMET['DateTimeStamp'])
## Set 'timestamp' column as index
TJRTLMET.set_index('timestamp', inplace=True)
## Set timezone to Pacific Time
TJRTLMET.index = TJRTLMET.index.tz_localize('Etc/GMT+8')
## Convert to UTC
TJRTLMET.index = TJRTLMET.index.tz_convert('UTC')

## 15-minute precipitation(mm)
IB_precip_15min = TJRTLMET.TotPrcp
## Fill in missing values with 0
IB_precip_15min = IB_precip_15min.fillna(0)
## Make IB_precip_15min index a datetime index
IB_precip_15min.index = pd.to_datetime(IB_precip_15min.index)
grouped = IB_precip_15min.groupby(level=0)
IB_precip_15min = grouped.mean()
## Hourly TOTAL Precipitation (4*15 min measurements per hour)
IB_precip_1hour = IB_precip_15min.resample('H').sum()
## Daily TOTAL Precipitation (4*15 min measurements per hour * 24 hours per day)
IB_precip_1day = IB_precip_15min.resample('D').sum()
## 7-day running TOTAL Precipitation (4*15 min measurements per hour * 24 hours per day * 7 days)
IB_precip_7daysum = IB_precip_1hour.rolling(window=24*7).sum()
## 7-day running MEAN Precipitation (4*15 min measurements per hour * 24 hours per day * 7 days)
IB_precip_7daymean = IB_precip_1hour.rolling(window=24*7).mean()
## 30-day running TOTAL Precipitation (4*15 min measurements per hour * 24 hours per day * 30 days)
IB_precip_30daysum = IB_precip_1hour.rolling(window=24*30).sum()
## 30-day running MEAN Precipitation (4*15 min measurements per hour * 24 hours per day * 30 days)
IB_precip_30daymean = IB_precip_1hour.rolling(window=24*30).mean()

## Truncate IB_precip data to start on 2001-10-01 and end before 2024-10-01 00:00:00
IB_precip_15min = IB_precip_15min.loc['2001-10-01':'2024-09-30 23:45:00']
IB_precip_1hour = IB_precip_1hour.loc['2001-10-01':'2024-09-30 23:00:00']
IB_precip_1day = IB_precip_1day.loc['2001-10-01':'2024-09-30']
IB_precip_7daysum = IB_precip_7daysum.loc['2001-10-01':'2024-09-30 23:00:00']
IB_precip_7daymean = IB_precip_7daymean.loc['2001-10-01':'2024-09-30 23:00:00']
IB_precip_30daysum = IB_precip_30daysum.loc['2001-10-01':'2024-09-30 23:00:00']
IB_precip_30daymean = IB_precip_30daymean.loc['2001-10-01':'2024-09-30 23:00:00']
#%% Save the IB_precip data
IB_precip_storage = pd.HDFStore(path_to_IB_precip)
IB_precip_storage['IB_precip_15min'] = IB_precip_15min
IB_precip_storage['IB_precip_1hour'] = IB_precip_1hour
IB_precip_storage['IB_precip_1day'] = IB_precip_1day
IB_precip_storage['IB_precip_7daysum'] = IB_precip_7daysum
IB_precip_storage['IB_precip_7daymean'] = IB_precip_7daymean
IB_precip_storage['IB_precip_30daysum'] = IB_precip_30daysum
IB_precip_storage['IB_precip_30daymean'] = IB_precip_30daymean
IB_precip_storage.close()

#%% Load the IB_precip data
IB_precip_storage = pd.HDFStore(path_to_IB_precip)
IB_precip_15min = IB_precip_storage['IB_precip_15min']
IB_precip_1hour = IB_precip_storage['IB_precip_1hour']
IB_precip_1day = IB_precip_storage['IB_precip_1day']
IB_precip_7daysum = IB_precip_storage['IB_precip_7daysum']
IB_precip_7daymean = IB_precip_storage['IB_precip_7daymean']
IB_precip_30daysum = IB_precip_storage['IB_precip_30daysum']
IB_precip_30daymean = IB_precip_storage['IB_precip_30daymean']
IB_precip_storage.close()

#%% Load and prepare the Otay Basin precipitation data

## Load Otay River Basin Daily Precipitation Data
otaybasin_raw = pd.read_csv(path_to_otaybasin_prcp,header=0)
## Convert the 'system:index' column to datetime from format "YYYYMMDD"
otaybasin_raw['timestamp'] = pd.to_datetime(otaybasin_raw['date'], format='%Y%m%d')
## Create new series just for the precipitation data with 'timestamp' as index
otaybasin_1day = otaybasin_raw['prcp']
## Make otaybasin index a datetime index
otaybasin_1day.index = otaybasin_raw['timestamp']
## Resample to daily data (to get frequency set exactly for stress model)
otaybasin_1day = otaybasin_1day.resample('D').sum()
## Fill in missing values with 0
otaybasin_1day = otaybasin_1day.fillna(0)
## 7-day running TOTAL Precipitation
otaybasin_7daysum = otaybasin_1day.rolling(window=7).sum()
## 7-day running MEAN Precipitation
otaybasin_7daymean = otaybasin_1day.rolling(window=7).mean()
## 30-day running TOTAL Precipitation
otaybasin_30daysum = otaybasin_1day.rolling(window=30).sum()
## 30-day running MEAN Precipitation
otaybasin_30daymean = otaybasin_1day.rolling(window=30).mean()

#%% Save the otaybasin_precip data
otaybasin_precip_storage = pd.HDFStore(path_to_otaybasin_precip)
otaybasin_precip_storage['otaybasin_1day'] = otaybasin_1day
otaybasin_precip_storage['otaybasin_7daysum'] = otaybasin_7daysum
otaybasin_precip_storage['otaybasin_7daymean'] = otaybasin_7daymean
otaybasin_precip_storage['otaybasin_30daysum'] = otaybasin_30daysum
otaybasin_precip_storage['otaybasin_30daymean'] = otaybasin_30daymean
otaybasin_precip_storage.close()

#%% Load the otaybasin_precip data
otaybasin_precip_storage = pd.HDFStore(path_to_otaybasin_precip)
otaybasin_1day = otaybasin_precip_storage['otaybasin_1day']
otaybasin_7daysum = otaybasin_precip_storage['otaybasin_7daysum']
otaybasin_7daymean = otaybasin_precip_storage['otaybasin_7daymean']
otaybasin_30daysum = otaybasin_precip_storage['otaybasin_30daysum']
otaybasin_30daymean = otaybasin_precip_storage['otaybasin_30daymean']
otaybasin_precip_storage.close()

#%% Load and prepare the TJ River Estuary Basin precipitation data

## Load TJ River Estuary Basin Daily Precipitation Data
tjbasin_raw = pd.read_csv(path_to_tjbasin_prcp,header=0)
## Convert the 'system:index' column to datetime at 00:00 from format "YYYYMMDD"
tjbasin_raw['timestamp'] = pd.to_datetime(tjbasin_raw['date'], format='%Y%m%d')
## Create new series just for the precipitation data with 'timestamp' as index
tjbasin_1day = tjbasin_raw['prcp']
## Make tjbasin index a datetime index
tjbasin_1day.index = tjbasin_raw['timestamp']
## Resample to daily data (to get frequency set exactly for stress model)
tjbasin_1day = tjbasin_1day.resample('D').sum()
## Fill in missing values with 0
tjbasin_1day = tjbasin_1day.fillna(0)
## 7-day running TOTAL Precipitation
tjbasin_7daysum = tjbasin_1day.rolling(window=7).sum()
## 7-day running MEAN Precipitation
tjbasin_7daymean = tjbasin_1day.rolling(window=7).mean()
## 30-day running TOTAL Precipitation
tjbasin_30daysum = tjbasin_1day.rolling(window=30).sum()
## 30-day running MEAN Precipitation
tjbasin_30daymean = tjbasin_1day.rolling(window=30).mean()

#%% Save the tjbasin_precip data
tjbasin_precip_storage = pd.HDFStore(path_to_tjbasin_precip)
tjbasin_precip_storage['tjbasin_1day'] = tjbasin_1day
tjbasin_precip_storage['tjbasin_7daysum'] = tjbasin_7daysum
tjbasin_precip_storage['tjbasin_7daymean'] = tjbasin_7daymean
tjbasin_precip_storage['tjbasin_30daysum'] = tjbasin_30daysum
tjbasin_precip_storage['tjbasin_30daymean'] = tjbasin_30daymean
tjbasin_precip_storage.close()

#%% Load the tjbasin_precip data
tjbasin_precip_storage = pd.HDFStore(path_to_tjbasin_precip)
tjbasin_1day = tjbasin_precip_storage['tjbasin_1day']
tjbasin_7daysum = tjbasin_precip_storage['tjbasin_7daysum']
tjbasin_7daymean = tjbasin_precip_storage['tjbasin_7daymean']
tjbasin_30daysum = tjbasin_precip_storage['tjbasin_30daysum']
tjbasin_30daymean = tjbasin_precip_storage['tjbasin_30daymean']
tjbasin_precip_storage.close()

#%% Combine TJ and Otay Basin Precipitation Data
combinedbasin_1day = tjbasin_1day + otaybasin_1day
combinedbasin_7daysum = tjbasin_7daysum + otaybasin_7daysum
combinedbasin_7daymean = tjbasin_7daymean + otaybasin_7daymean
combinedbasin_30daysum = tjbasin_30daysum + otaybasin_30daysum
combinedbasin_30daymean = tjbasin_30daymean + otaybasin_30daymean

#%% Save the combinedbasin_precip data
combinedbasin_precip_storage = pd.HDFStore(path_to_combinedbasin_precip)
combinedbasin_precip_storage['combinedbasin_1day'] = combinedbasin_1day
combinedbasin_precip_storage['combinedbasin_7daysum'] = combinedbasin_7daysum
combinedbasin_precip_storage['combinedbasin_7daymean'] = combinedbasin_7daymean
combinedbasin_precip_storage['combinedbasin_30daysum'] = combinedbasin_30daysum
combinedbasin_precip_storage['combinedbasin_30daymean'] = combinedbasin_30daymean
combinedbasin_precip_storage.close()

#%% Load the combinedbasin_precip data
combinedbasin_precip_storage = pd.HDFStore(path_to_combinedbasin_precip)
combinedbasin_1day = combinedbasin_precip_storage['combinedbasin_1day']
combinedbasin_7daysum = combinedbasin_precip_storage['combinedbasin_7daysum']
combinedbasin_7daymean = combinedbasin_precip_storage['combinedbasin_7daymean']
combinedbasin_30daysum = combinedbasin_precip_storage['combinedbasin_30daysum']
combinedbasin_30daymean = combinedbasin_precip_storage['combinedbasin_30daymean']
combinedbasin_precip_storage.close()

#%% Create arrays of 64 days starting on 2021-12-15
startdate = '2021-12-15'
enddate = '2022-02-17'
# Note: several small precipitation events, but might be best continuous data

# IB_precip_64days = IB_precip_15min[(IB_precip_15min.index >= startdate) & (IB_precip_15min.index < enddate)]
IB_precip_64days = IB_precip_1day[(IB_precip_1day.index >= startdate) & (IB_precip_1day.index < enddate)]
otaybasin_64days = otaybasin_1day[(otaybasin_1day.index >= startdate) & (otaybasin_1day.index < enddate)]
tjbasin_64days = tjbasin_1day[(tjbasin_1day.index >= startdate) & (tjbasin_1day.index < enddate)]
combinedbasin_64days = combinedbasin_1day[(combinedbasin_1day.index >= startdate) & (combinedbasin_1day.index < enddate)]
#%%Plot all daily precipitation data
%matplotlib qt
fig, ax = plt.subplots(4, 1, figsize=(12, 12), sharex=True)
ax[0].plot(IB_precip_1day, label='IB')
ax[0].set_ylabel('Precipitation (mm)')
ax[0].set_title('IB Rain Gauge')
ax[1].plot(otaybasin_1day, label='Otay Basin')
ax[1].set_ylabel('Precipitation (mm)')
ax[1].set_title('Otay Basin')
ax[2].plot(tjbasin_1day, label='TJ Basin')
ax[2].set_ylabel('Precipitation (mm)')
ax[2].set_title('TJ Basin')
ax[3].plot(combinedbasin_1day, label='Combined Basin')
ax[3].set_ylabel('Precipitation (mm)')
ax[3].set_title('Combined Basin')
plt.show()
#%% Plot the 64-day precipitation data
%matplotlib qt
fig, ax = plt.subplots(4, 1, figsize=(12, 12), sharex=True)
ax[0].plot(IB_precip_64days.index, IB_precip_64days, label='IB')
ax[0].set_ylabel('Precipitation (mm)')
ax[0].set_title('IB Rain Gauge')
ax[1].plot(otaybasin_64days.index, otaybasin_64days, label='Otay Basin')
ax[1].set_ylabel('Precipitation (mm)')
ax[1].set_title('Otay Basin')
ax[2].plot(tjbasin_64days.index, tjbasin_64days, label='TJ Basin')
ax[2].set_ylabel('Precipitation (mm)')
ax[2].set_title('TJ Basin')
ax[3].plot(combinedbasin_64days.index, combinedbasin_64days, label='Combined Basin')
ax[3].set_ylabel('Precipitation (mm)')
ax[3].set_title('Combined Basin')
plt.show()

#%% Load the CMIP6 ensemble data
with open(path_to_cmip6_ensemble, 'rb') as f:
    cmip6_ensemble = pickle.load(f)

cmip6_df = pd.DataFrame(cmip6_ensemble)
#%% Plot the precip from IB historical data and the CMIP6 ensemble
## Historical data
%matplotlib qt
plt.figure()
## CMIP6 ensemble data
for key in cmip6_ensemble.keys():
    ## Convert time to timestamp
    timestamps = pd.to_datetime(cmip6_ensemble[key]['time'])
    plt.plot(timestamps, cmip6_ensemble[key]['prec'], label=key)

plt.plot(IB_precip_1day,label='Historical Daily Precipitation')

plt.ylabel('Precipitation (mm/day)')
plt.title('Historical Precip and CMIP6 Ensemble')
plt.legend()
plt.show()

#%% Compute historical and CMIP6 ensemble projections of annual precipitation
## Historical data
IB_precip_annual = IB_precip_1day.resample('AS-OCT').sum()
## CMIP6 ensemble data
cmip6_annual = {}

for key in cmip6_ensemble.keys():
    ## Resample to annual data
    cmip6_annual[key] = pd.Series(cmip6_df[key]['prec'],index = cmip6_df[key]['time']).resample('AS-OCT').sum()

#%% Plot the annual precipitation data
## Historical data
%matplotlib qt
plt.figure()
## CMIP6 ensemble data
for key in cmip6_annual.keys():
    ## Convert time to timestamp
    timestamps = pd.to_datetime(cmip6_ensemble[key]['time'])
    plt.plot(cmip6_annual[key].index, cmip6_annual[key], label=key)

plt.plot(IB_precip_annual, label='Historical Annual Precipitation', linewidth=2, color='black')
plt.ylabel('Annual Precipitation (mm)', fontsize=12)
plt.title('Historical and CMIP6 Ensemble Annual Precipitation')
plt.legend()
plt.show()

#%% Calculate the ratio of annual CMIP6 precip from 2014-10-01 to 2024-09-30 to historical precip
## Historical data from 2014-10-01 to 2024-09-30
IB_precip_hist = IB_precip_annual.loc['2014-10-01':'2024-09-30']
## Drop tz info
IB_precip_hist.index = IB_precip_hist.index.tz_localize(None)
## CMIP6 ensemble data from 2014-10-01 to 2024-09-30
cmip6_hist = {}
for key in cmip6_annual.keys():
    cmip6_hist[key] = cmip6_annual[key].loc['2014-10-01':'2024-09-30']
## Calculate the ratio of CMIP6 to historical precip
cmip6_ratio = {}
for key in cmip6_hist.keys():
    # cmip6_ratio[key] = np.mean(cmip6_hist[key] / IB_precip_hist)
    cmip6_ratio[key] = np.mean(cmip6_hist[key]) / np.mean(IB_precip_hist)

## Plot CMIP6 annual precipitation scaled by the ratio and compare to historical data
%matplotlib qt
plt.figure()
for key in cmip6_annual.keys():
    ## Convert time to timestamp
    timestamps = pd.to_datetime(cmip6_ensemble[key]['time'])
    plt.plot(cmip6_annual[key].index, cmip6_annual[key] / cmip6_ratio[key], label=key)

plt.plot(IB_precip_hist, label='Historical Annual Precipitation', linewidth=2, color='black')
plt.ylabel('Annual Precipitation (mm)', fontsize=12)
plt.title('Historical and CMIP6 Ensemble Annual Precipitation Scaled')
plt.legend()
#%% Create input time series from 2003-10-01 to 2100-08-31
## Historical Observations through 2024-09-30
## 8 Unique CMIP6 Dynamically Downscaled Models out to 2100-08-31
## (1 month earlier than a full water year b/c of end of CMIP6 data)
## Concatenate the historical data 2003-10-01 to 2024-09-31 with each of the scaled CMIP6 models from 2024-10-01 to 2100-08-31
precip_2100 = pd.DataFrame(index=pd.date_range(start='2003-10-01', end='2100-08-31', freq='H'))

## Choose precipitation data to use for historical data
precip_data = IB_precip_1hour #### HOURLY
# precip_data = IB_precip_1day #### DAILY
# precip_data = otaybasin_7daymean

precip_hist = precip_data.loc['2003-10-01':'2024-09-30'].copy()

## CMIP6 data
cmip6_time = pd.to_datetime(cmip6_ensemble[list(cmip6_ensemble.keys())[1]]['time'])

for key in cmip6_ensemble.keys():
    precip_2100[key] = np.nan  # Initialize with NaNs

    ## Extract the daily CMIP6 precipitation data
    prec = cmip6_ensemble[key]['prec']
    ## If length of prec < 31411 (days beteen 2014-09-01 and 2100-08-31), pad with the last value
    ## Does not matter - we will not be using any values in calendar year 2100
    if len(prec) < 31411:
        prec = np.pad(prec, (0, 31411 - len(prec)), 'mean')
    ## To adjust cmip to match observations, use scaling factor (/cmip6_ratio[key])
    cmip6_series = pd.Series(prec, index=cmip6_time)#/cmip6_ratio[key]
    prec_hourly = (cmip6_series/24).resample('H').ffill()
    
    precip_2100.loc['2003-10-01':'2024-09-30', key] = precip_hist.values  # Copy historical data
    precip_2100.loc['2024-10-01':'2100-08-31', key] = prec_hourly.loc['2024-10-01':'2100-08-31'].values  # Copy scaled CMIP6 data


#%% Plot precip_2100
%matplotlib qt
precip_2100.plot()
IB_precip_1hour.plot()
plt.ylabel('Precip (mm/hour)')
plt.title('Precipitation out to 2100')
plt.show()

#%% Save the precip_2100 data
precip_2100.to_hdf(path_to_precip_2100, 'precip_2100')
print('Saved precip_2100 data to', path_to_precip_2100)
# %%
