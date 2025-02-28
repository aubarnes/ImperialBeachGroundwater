"""
Import NASA Interagency Task Force sea level rise data for La Jolla Tide Gauge
Level the MSL projections (relative to 2000 MSL) to NAVD88
Create quadratic interpolation functions for each curve to get the mean sea level (NAVD88) at any given date

Import NOAA Tide Gauge data for La Jolla Tide Gauge
Get non-tidal residual using Cosine-Lanczos Squared Filter with 36-hour cutoff

Pull MOP wave data to calculate R2 and TWL (integrating tides from NOAA La Jolla Tide Gauge)
Compare IPA TWL with Tuned Stockdon06 TWL

Parts of code adapted from code sent by Mele Johnson in July 2024 by Austin Barnes
V2 - updated to include foreshore slopes derived directly from survey data
Updated September 2024 by Austin Barnes with code from Mele Johnson
"""
#%% Imports & Define Directory
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib
import requests
import datetime as dt
from scipy.interpolate import CubicSpline
import pickle

## Observations
path_to_ljtide_full = '../data/ljtide_1924.h5'
## SLR Projections
path_to_ljslritf = '../data/lj_slr_itf.csv'
## Output
path_to_slr_interp = '../data/slr_interp.pkl'

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
    print('Loaded La Jolla Tide Gauge Data from ljtide_1924.h5')
except:
    print('La Jolla Tide Gauge Data not found, calling getTides...')
    ljtide_raw = getTides(date1, date2)
    ljtide_full = ljtide_raw[ljtide_raw.first_valid_index():]
    ## Save ljtide_raw to ljtide_1924.h5
    ljtide_full.to_hdf(path_to_ljtide_full, 'ljtide')
    print('Saved La Jolla Tide Gauge Data to ljtide_1924.h5')

## La Jolla Tide Gauge 2000 average water level (NAVD88)
ljtide_2000mean_NAVD88 = ljtide_full['2000-01-01':'2000-12-31'].mean()
## Compute annual average water level for each year
ljtide_annualmean_NAVD88 = ljtide_full.resample('AS').mean()
#%% Load 2022 NASA Interagency Task Force sea level rise data & Create Sea Level Rise Curves

## Import SLR projection data for scenarios
## Projections are relative to 2000 annual average MSL at La Jolla Tide Gauge
slr_data = pd.read_csv(path_to_ljslritf)
## Create new dataframe that is slr_data from columns 3 to end
slr_projections = slr_data.iloc[:,3:]
## Divide all values in columns 2 to end by 1000 to convert from mm to m
slr_projections.iloc[:,2:] = slr_projections.iloc[:,2:].div(1000)

## Projections are relative to 2000 annual average MSL at La Jolla Tide Gauge
slr_projections.iloc[:,2:] = slr_projections.iloc[:,2:] + ljtide_2000mean_NAVD88

slr_low_17p = pd.Series(slr_projections.iloc[0,2:], index=slr_projections.columns[2:])
slr_low_17p.name = 'Low 17th Percentile'
slr_low_17p.index = pd.to_datetime(slr_low_17p.index)
## Fit a cubic spline interpolation to the low 17th percentile curve
slr_low_17p_interp = CubicSpline(slr_low_17p.index.to_julian_date(), slr_low_17p)

slr_low_50p = pd.Series(slr_projections.iloc[1,2:], index=slr_projections.columns[2:])
slr_low_50p.name = 'Low 50th Percentile'
slr_low_50p.index = pd.to_datetime(slr_low_50p.index)
slr_low_50p_interp = CubicSpline(slr_low_50p.index.to_julian_date(), slr_low_50p)

slr_low_83p = pd.Series(slr_projections.iloc[2,2:], index=slr_projections.columns[2:])
slr_low_83p.name = 'Low 83rd Percentile'
slr_low_83p.index = pd.to_datetime(slr_low_83p.index)
slr_low_83p_interp = CubicSpline(slr_low_83p.index.to_julian_date(), slr_low_83p)

slr_intlow_17p = pd.Series(slr_projections.iloc[3,2:], index=slr_projections.columns[2:])
slr_intlow_17p.name = 'IntLow 17th Percentile'
slr_intlow_17p.index = pd.to_datetime(slr_intlow_17p.index)
slr_intlow_17p_interp = CubicSpline(slr_intlow_17p.index.to_julian_date(), slr_intlow_17p)

slr_intlow_50p = pd.Series(slr_projections.iloc[4,2:], index=slr_projections.columns[2:])
slr_intlow_50p.name = 'IntLow 50th Percentile'
slr_intlow_50p.index = pd.to_datetime(slr_intlow_50p.index)
slr_intlow_50p_interp = CubicSpline(slr_intlow_50p.index.to_julian_date(), slr_intlow_50p)

slr_intlow_83p = pd.Series(slr_projections.iloc[5,2:], index=slr_projections.columns[2:])
slr_intlow_83p.name = 'IntLow 83rd Percentile'
slr_intlow_83p.index = pd.to_datetime(slr_intlow_83p.index)
slr_intlow_83p_interp = CubicSpline(slr_intlow_83p.index.to_julian_date(), slr_intlow_83p)

slr_int_17p = pd.Series(slr_projections.iloc[6,2:], index=slr_projections.columns[2:])
slr_int_17p.name = 'Int 17th Percentile'
slr_int_17p.index = pd.to_datetime(slr_int_17p.index)
slr_int_17p_interp = CubicSpline(slr_int_17p.index.to_julian_date(), slr_int_17p)

slr_int_50p = pd.Series(slr_projections.iloc[7,2:], index=slr_projections.columns[2:])
slr_int_50p.name = 'Int 50th Percentile'
slr_int_50p.index = pd.to_datetime(slr_int_50p.index)
slr_int_50p_interp = CubicSpline(slr_int_50p.index.to_julian_date(), slr_int_50p)

slr_int_83p = pd.Series(slr_projections.iloc[8,2:], index=slr_projections.columns[2:])
slr_int_83p.name = 'Int 83rd Percentile'
slr_int_83p.index = pd.to_datetime(slr_int_83p.index)
slr_int_83p_interp = CubicSpline(slr_int_83p.index.to_julian_date(), slr_int_83p)

slr_inthigh_17p = pd.Series(slr_projections.iloc[9,2:], index=slr_projections.columns[2:])
slr_inthigh_17p.name = 'IntHigh 17th Percentile'
slr_inthigh_17p.index = pd.to_datetime(slr_inthigh_17p.index)
slr_inthigh_17p_interp = CubicSpline(slr_inthigh_17p.index.to_julian_date(), slr_inthigh_17p)

slr_inthigh_50p = pd.Series(slr_projections.iloc[10,2:], index=slr_projections.columns[2:])
slr_inthigh_50p.name = 'IntHigh 50th Percentile'
slr_inthigh_50p.index = pd.to_datetime(slr_inthigh_50p.index)
slr_inthigh_50p_interp = CubicSpline(slr_inthigh_50p.index.to_julian_date(), slr_inthigh_50p)

slr_inthigh_83p = pd.Series(slr_projections.iloc[11,2:], index=slr_projections.columns[2:])
slr_inthigh_83p.name = 'IntHigh 83rd Percentile'
slr_inthigh_83p.index = pd.to_datetime(slr_inthigh_83p.index)
slr_inthigh_83p_interp = CubicSpline(slr_inthigh_83p.index.to_julian_date(), slr_inthigh_83p)

slr_high_17p = pd.Series(slr_projections.iloc[12,2:], index=slr_projections.columns[2:])
slr_high_17p.name = 'High 17th Percentile'
slr_high_17p.index = pd.to_datetime(slr_high_17p.index)
slr_high_17p_interp = CubicSpline(slr_high_17p.index.to_julian_date(), slr_high_17p)

slr_high_50p = pd.Series(slr_projections.iloc[13,2:], index=slr_projections.columns[2:])
slr_high_50p.name = 'High 50th Percentile'
slr_high_50p.index = pd.to_datetime(slr_high_50p.index)
slr_high_50p_interp = CubicSpline(slr_high_50p.index.to_julian_date(), slr_high_50p)

slr_high_83p = pd.Series(slr_projections.iloc[14,2:], index=slr_projections.columns[2:])
slr_high_83p.name = 'High 83rd Percentile'
slr_high_83p.index = pd.to_datetime(slr_high_83p.index)
slr_high_83p_interp = CubicSpline(slr_high_83p.index.to_julian_date(), slr_high_83p)

#%% Plot the sea level rise curves and observations from 2000 onwards
fontsize = 18
%matplotlib qt
fig, ax = plt.subplots(1, 1, figsize=(12, 8), sharex=True)

# Plot the SLR curves with scatter points and interpolated curves
ax.scatter(slr_low_50p.index, slr_low_50p, label='Low', color='blue', alpha=0.7)
ax.plot(slr_low_50p.index, slr_low_50p_interp(slr_low_50p.index.to_julian_date()), color='blue', alpha=0.7)

ax.scatter(slr_intlow_50p.index, slr_intlow_50p, label='Intermediate Low', color='green', alpha=0.7)
ax.plot(slr_intlow_50p.index, slr_intlow_50p_interp(slr_intlow_50p.index.to_julian_date()), color='green', alpha=0.7)

ax.scatter(slr_int_50p.index, slr_int_50p, label='Intermediate', color='orange', alpha=0.7)
ax.plot(slr_int_50p.index, slr_int_50p_interp(slr_int_50p.index.to_julian_date()), color='orange', alpha=0.7)

ax.scatter(slr_inthigh_50p.index, slr_inthigh_50p, label='Intermediate High', color='red', alpha=0.7)
ax.plot(slr_inthigh_50p.index, slr_inthigh_50p_interp(slr_inthigh_50p.index.to_julian_date()), color='red', alpha=0.7)

ax.scatter(slr_high_50p.index, slr_high_50p, label='High', color='purple', alpha=0.7)
ax.plot(slr_high_50p.index, slr_high_50p_interp(slr_high_50p.index.to_julian_date()), color='purple', alpha=0.7)

# Shade between the 83rd and 17th percentiles
ax.fill_between(slr_low_17p.index, slr_low_17p_interp(slr_low_17p.index.to_julian_date()), slr_low_83p_interp(slr_low_83p.index.to_julian_date()), color='blue', alpha=0.2)
ax.fill_between(slr_intlow_17p.index, slr_intlow_17p_interp(slr_intlow_17p.index.to_julian_date()), slr_intlow_83p_interp(slr_intlow_83p.index.to_julian_date()), color='green', alpha=0.2)
ax.fill_between(slr_int_17p.index, slr_int_17p_interp(slr_int_17p.index.to_julian_date()), slr_int_83p_interp(slr_int_83p.index.to_julian_date()), color='orange', alpha=0.2)
ax.fill_between(slr_inthigh_17p.index, slr_inthigh_17p_interp(slr_inthigh_17p.index.to_julian_date()), slr_inthigh_83p_interp(slr_inthigh_83p.index.to_julian_date()), color='red', alpha=0.2)
ax.fill_between(slr_high_17p.index, slr_high_17p_interp(slr_high_17p.index.to_julian_date()), slr_high_83p_interp(slr_high_83p.index.to_julian_date()), color='purple', alpha=0.2)

# Plot the Annual Average La Jolla Tide Gauge Water Level (NAVD88) from 2000 onwards
ax.plot(ljtide_annualmean_NAVD88, label='Observations', color='black', linewidth=4)

ax.set_ylabel('Annual Average Sea Level (m, NAVD88)', fontsize=fontsize)
ax.legend(fontsize=fontsize)
ax.set_title('NASA Interagency Task Force Sea Level Rise Projections for La Jolla', fontsize=fontsize)

ax.set_ylim(0.5, 6.0)
ax.set_xlim(pd.to_datetime('1920-01-01').tz_localize('UTC'), pd.to_datetime('2160-01-01').tz_localize('UTC'))

# Set font size for tick marks
ax.tick_params(axis='both', which='major', labelsize=fontsize)

plt.show()

#%% Save interp functions to file
slr_interp = {'low_17p':slr_low_17p_interp, 'low_50p':slr_low_50p_interp, 'low_83p':slr_low_83p_interp,
                'intlow_17p':slr_intlow_17p_interp, 'intlow_50p':slr_intlow_50p_interp, 'intlow_83p':slr_intlow_83p_interp,
                'int_17p':slr_int_17p_interp, 'int_50p':slr_int_50p_interp, 'int_83p':slr_int_83p_interp,
                'inthigh_17p':slr_inthigh_17p_interp, 'inthigh_50p':slr_inthigh_50p_interp, 'inthigh_83p':slr_inthigh_83p_interp,
                'high_17p':slr_high_17p_interp, 'high_50p':slr_high_50p_interp, 'high_83p':slr_high_83p_interp}

with open(path_to_slr_interp, 'wb') as f:
    pickle.dump(slr_interp, f)

#%% Load interp functions from file
import pickle
with open(path_to_slr_interp, 'rb') as f:
    slr_interp = pickle.load(f)

slr_low_17p_interp = slr_interp['low_17p']
slr_low_50p_interp = slr_interp['low_50p']
slr_low_83p_interp = slr_interp['low_83p']
slr_intlow_17p_interp = slr_interp['intlow_17p']
slr_intlow_50p_interp = slr_interp['intlow_50p']
slr_intlow_83p_interp = slr_interp['intlow_83p']
slr_int_17p_interp = slr_interp['int_17p']
slr_int_50p_interp = slr_interp['int_50p']
slr_int_83p_interp = slr_interp['int_83p']
slr_inthigh_17p_interp = slr_interp['inthigh_17p']
slr_inthigh_50p_interp = slr_interp['inthigh_50p']
slr_inthigh_83p_interp = slr_interp['inthigh_83p']
slr_high_17p_interp = slr_interp['high_17p']
slr_high_50p_interp = slr_interp['high_50p']
slr_high_83p_interp = slr_interp['high_83p']

# %%
