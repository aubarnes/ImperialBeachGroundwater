"""
La Jolla Tide 1: UTide Analysis
Austin Barnes 2024

Analyze La Jolla Tide Gauge Data with UTide
Store coefficients for tidal forecasts
Create tidal forecast out to 2100

NOTE: SA and SSA constituents must not contain the seasonal cycle not attributable to tides
"""
#%% Imports
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib
import numpy as np
import requests
from utide import solve, reconstruct
import datetime as dt
import pickle

path_to_ljtide_full = '../data/ljtide_1924.h5'
path_to_coef_lj = '../data/coef_lj.pkl'
path_to_ljtide_2100 = '../data/ljtide_2100.h5'
#%% Function: getTides - get tide data from NOAA
def getTides(date1, date2, station_id = 9410230, product = 'hourly_height', datum = 'NAVD', tz='GMT', units='metric', format='json'):
    ## If the date range is less than 1 year, can call download_tide_data once
    if (date2 - date1).days <= 365:
        begin_date = date1.strftime('%Y%m%d')
        end_date = date2.strftime('%Y%m%d')
        print('Returning NOAA Tide Data for the following date range:')
        print(f'Begin Date: {begin_date}')
        print(f'End Date: {end_date}')
        tide = download_tide_data(begin_date, end_date, station_id, product, datum, tz, units, format)
        tide.name = datum
    ## If the date range is greater than 1 year, need to call download_tide_data multiple times
    ## and concatenate the results
    else:
        ## Create date ranges for each year and for the fraction of the final year
        date_ranges = pd.date_range(date1, date2, freq='YS')
        ## Append the end date + 1 day to the date ranges
        date_ranges = date_ranges.append(pd.DatetimeIndex([date2 + pd.Timedelta(days=1)]))
        print('Attempting to return NOAA tide data for the following date range:')
        print(f'Begin Date: {date_ranges[0]}')
        print(f'End Date: {date_ranges[-1]}')
        tide = pd.Series()
        ## Loop through the date ranges and download the tide data for each interval
        for i in range(len(date_ranges)-1):
            begin_date = date_ranges[i].strftime('%Y%m%d')
            ## define end date as the first day of the next interval minus one day
            end_date = (date_ranges[i + 1] - pd.Timedelta(days=1)).strftime('%Y%m%d')
            ## Download the tide data for the interval and append it to the series
            tide = pd.concat([tide,download_tide_data(begin_date, end_date, station_id, product, datum, tz, units, format)])
            tide.name = datum
    print('Returning NOAA tide data for the following date range:')
    print(f'Begin Date: {tide.index[0]}')
    print(f'End Date: {tide.index[-1]}')
    print('Any truncation of the data is due to the NOAA API constraints or QA/QC data availability.')
    return tide
#%% Function: download_tide_data - helper function called by getTides
def download_tide_data(begin_date, end_date, station_id = 9410230, product = 'hourly_height', datum = 'NAVD', tz='GMT', units='metric', format='json'):
    ## For API options and constraints, see https://api.tidesandcurrents.noaa.gov/api/prod/
    url = f"https://api.tidesandcurrents.noaa.gov/api/prod/datagetter?product={product}&application=NOS.COOPS.TAC.WL&begin_date={begin_date}&end_date={end_date}&station={station_id}&datum={datum}&time_zone={tz}&units={units}&format={format}"
    print(url)
    response = requests.get(url)
    data = response.json()
    t = pd.to_datetime([item['t'] for item in data['data']], utc=True)
    vraw = [item['v'] for item in data['data']]
    # v = [float(item['v']) for item in data['data']]
    v = pd.to_numeric(vraw, errors='coerce')
    return pd.Series(v, index=t)
#%% Retrieve LJ Tide Gauge data
# date1 = dt.datetime(2000,1,1,tzinfo=dt.timezone.utc)
date1 = dt.datetime(2000,1,1,tzinfo=dt.timezone.utc)
date2 = dt.datetime(2000,12,31,tzinfo=dt.timezone.utc)

## Defaults for getTides
## station_id = 9410230 (La Jolla, CA)
## ALTERNATE station_id = 9410170 (San Diego, CA)
## product = 'hourly_height'
## ALTERNATE product = 'water_level' (6-minute data, can only pull 31 days at a time)
## datum = 'NAVD'
## ALTERNATE datum = 'MSL'
## tz = 'GMT'
## units = 'metric'
## format = 'json'

## See if ljtide_1924.h5 exists, if it does, load; if not - call getTides
try:
    ljtide_full = pd.read_hdf(path_to_ljtide_full, 'ljtide')
    print(['Loaded La Jolla Tide Gauge Data from '+path_to_ljtide_full])
except:
    print('La Jolla Tide Gauge Data not found, calling getTides...')
    ljtide_raw = getTides(date1, date2)
    ljtide_full = ljtide_raw[ljtide_raw.first_valid_index():]
    ## Save ljtide_raw to ljtide_1924.h5
    ljtide_full.to_hdf(path_to_ljtide_full, 'ljtide')
    print(['Saved La Jolla Tide Gauge Data to '+path_to_ljtide_full])
#%% Create arrays of 64 days starting on 2021-12-15 to match groundwater data with minimal precipitation
startdate = '2021-12-15'
enddate = '2022-02-17'
# Note: several small precipitation events, but might be best continuous data

ljtide_64days = ljtide_full[(ljtide_full.index >= startdate) & (ljtide_full.index < enddate)]
#%% U-Tide Tidal Fit

# ##64 Days of low precipitation
# t_lj = ljtide_64days.index
# u_lj = ljtide_64days.values

## Tidal Datum Analysis Period: 1983-01-01 - 2001-12-31
t_lj = ljtide_full.index
t_lj = t_lj[(t_lj >= '1983-01-01') & (t_lj < '2002-01-01')]
u_lj = ljtide_full.loc[t_lj].values

## Configure Tidal Constituents
# tidal_constit = ['M2','S2','N2','K1','O1','P1','K2','Q1']
# tidal_constit = ['M2','S2','N2','K1','O1','P1','K2','Q1','SA']
# tidal_constit = ['MM', 'MSF', 'ALP1', '2Q1', 'Q1', 'O1', 'NO1', 'K1', 'J1', 'OO1',
#        'UPS1', 'EPS2', 'MU2', 'N2', 'M2', 'L2', 'S2', 'ETA2', 'MO3', 'M3',
#        'MK3', 'SK3', 'MN4', 'M4', 'SN4', 'MS4', 'S4', '2MK5', '2SK5',
#        '2MN6', 'M6', '2MS6', '2SM6', '3MK7', 'M8', 'SA']
# tidal_constit = ['ALP1', '2Q1', 'Q1', 'O1', 'NO1', 'K1', 'J1', 'OO1',
#        'UPS1', 'EPS2', 'MU2', 'N2', 'M2', 'L2', 'S2', 'ETA2', 'MO3', 'M3', 
#        'SK3', 'MN4', 'SN4', 'MS4', 'S4', '2MK5', '2SK5',
#        '2MN6', '2MS6', '2SM6', '3MK7', 'M8', 'SA']
tidal_constit = 'auto'

# For LJ dataset
coef_lj = solve(t_lj, u_lj, v=None,
             lat=32.866667,
             nodal = True,
             trend = False,
             constit = tidal_constit,
             method = 'ols',
             conf_int='linear',
             order_constit='frequency',
             Rayleigh_min=1.00)

## Display constituents and amplitudes
for i in range(len(coef_lj['name'])):
    print(coef_lj['name'][i])
    print('lj: ' + str(coef_lj['A'][i]))

#%% Zero out the SA and SSA constituents
## Find the indices of the SA and SSA constituents
sa_idx = np.where(coef_lj['name']=='SA')
ssa_idx = np.where(coef_lj['name']=='SSA')

## Zero out
coef_lj['A'][sa_idx] = 0
coef_lj['A'][ssa_idx] = 0

## Display constituents and amplitudes
for i in range(len(coef_lj['name'])):
    print(coef_lj['name'][i])
    print('lj: ' + str(coef_lj['A'][i]))
#%% Save coef_lj with pickle

with open(path_to_coef_lj, 'wb') as f:
    pickle.dump(coef_lj, f)

#%% Load coef_lj with pickle
import pickle
with open(path_to_coef_lj, 'rb') as f:
    coef_lj = pickle.load(f)
#%% Compare tidal reconstruction with observations for 2000 to 2024
## Tidal reconstruction for 2000 to 2024
t_recon = ljtide_full[(ljtide_full.index >= '2000-01-01') & (ljtide_full.index < '2024-10-01')].index
u_recon = reconstruct(t_recon, coef_lj)['h']
# %% Plot tidal reconstruction against the data
%matplotlib qt
fig, ax = plt.subplots(1, 1, figsize=(12, 6))
ax.plot(t_recon, ljtide_full['2000-01-01':'2024-10-01']-u_recon, label='Observed - Reconstructed')
ax.set_ylabel('Water Level (m)')
ax.set_title('La Jolla Tide Gauge Data: 2000-2024')
ax.legend()
plt.show()
#%% Create UTide Reconstruction from 2024-10-01 to 2100-01-01
t_recon = pd.date_range('2000-01-01', '2100-09-30T23:00:00', freq='H')
u_recon = reconstruct(t_recon, coef_lj)['h']

## Save u_recon and t_recon to ljtide_2100.h5
ljtide_2100 = pd.Series(u_recon, index=t_recon)
ljtide_2100.to_hdf(path_to_ljtide_2100, 'ljtide')
# %%
