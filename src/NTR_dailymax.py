"""
Import NOAA Tide Gauge data for La Jolla Tide Gauge
Get non-tidal residual using Cosine-Lanczos Squared Filter with 36-hour cutoff

Updated October 2024 by Austin Barnes
"""
#%% Imports & Define Directory
import numpy as np
import pandas as pd
import netCDF4 as nc
import matplotlib.pyplot as plt
%matplotlib
import requests
import datetime as dt

path_to_ljtide_dailymax_hindcast = '/Users/austinbarnes/Documents/UCSD SIO/IB Groundwater/ImperialBeach/data/LJ_tide_data/ljtide_dailymax_hindcast.h5'
path_to_ljtide_dailymax_24 = '/Users/austinbarnes/Documents/UCSD SIO/IB Groundwater/ImperialBeach/data/LJ_tide_data/ljtide_dailymax_24.h5'

#%% FUNCTION: Cosine-Lanczos Squared Filter
def cosine_lanczos_squared_filter(cutoff_hours, fs, N=None):
    """
    Create a cosine-Lanczos squared filter with a specified cutoff.
    
    Parameters:
    cutoff_hours (float): Cutoff period in hours.
    fs (float): Sampling rate in samples per hour.
    N (int): Number of filter coefficients. If None, a default value is used.
    
    Returns:
    h (numpy.ndarray): The filter coefficients.
    """
    # Calculate the cutoff frequency in cycles per sample
    fc = 1 / cutoff_hours
    
    # If N is not provided, calculate a default value
    if N is None:
        N = int(4 * cutoff_hours * fs)
    
    # Generate the filter coefficients
    n = np.arange(-N // 2, N // 2 + 1)
    sinc_func = np.sinc(2 * fc * n / fs)
    lanczos_func = np.sinc(n / N)
    cosine_func = np.cos(np.pi * n / N)
    
    # Impulse response of the cosine-Lanczos squared filter
    h = sinc_func * (lanczos_func * cosine_func) ** 2
    
    # Normalize the filter coefficients
    h /= np.sum(h)
    
    return h
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
#%% Retrieve LJ Tide Gauge data & Filter for the non-tidal residual
date1 = dt.datetime(1924,1,1,tzinfo=dt.timezone.utc) # verified hourly water level starts 1924-08-01 0800 UTC; almost continuous since 1973-07-01 0800 UTC
# date1 = dt.datetime(2000,1,1,tzinfo=dt.timezone.utc)
# date1 = dt.datetime(1999,1,1,tzinfo=dt.timezone.utc) # Go back 1 year to remove filter effects
# date2 = dt.datetime(2023,12,31,tzinfo=dt.timezone.utc)
# date2 = dt.datetime(2024,1,1,tzinfo=dt.timezone.utc) + dt.timedelta(days=5) # Add 5 days to remove filter effects
date2 = dt.datetime(2024,10,15,tzinfo=dt.timezone.utc)

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
    ljtide_full = pd.read_hdf('/Users/austinbarnes/Documents/UCSD SIO/IB Groundwater/ImperialBeach/data/LJ_tide_data/ljtide_1924.h5', 'ljtide')
    print('Loaded La Jolla Tide Gauge Data from ljtide_1924.h5')
except:
    print('La Jolla Tide Gauge Data not found, calling getTides...')
    ljtide_raw = getTides(date1, date2)
    ljtide_full = ljtide_raw[ljtide_raw.first_valid_index():]
    ## Save ljtide_raw to ljtide_1924.h5
    ljtide_full.to_hdf('/Users/austinbarnes/Documents/UCSD SIO/IB Groundwater/ImperialBeach/data/LJ_tide_data/ljtide_1924.h5', 'ljtide')
    print('Saved La Jolla Tide Gauge Data to ljtide_1924.h5')

# ## Interpolate NaNs in the data by filling with the mean of the surrounding values
ljtide_mean = ljtide_full.mean()
ljtide_full = ljtide_full.fillna(ljtide_mean)

# ## Interpolate NaNs in the data using linear interpolation but ONLY fill in the NaNs
# ljtide_full = ljtide_full.interpolate(method='linear')

## Compute and remove linear trend from tide data
trend = np.polyfit(ljtide_full.index.to_julian_date(), ljtide_full, 1)
ljtide = ljtide_full - np.polyval(trend, ljtide_full.index.to_julian_date())

## Filter the tide data using Cosine-Lanczos Squared Filter
cutoff_hours = 36
fs = 1  # samples per hour

# Generate the filter coefficients
h = cosine_lanczos_squared_filter(cutoff_hours, fs)

## Apply the filter
## filtered_ is for the filtered signal (low pass)
## _CL stands for cosine-Lanczos squared tidal signal (high pass)
filtered_ljtide = np.convolve(ljtide, h, mode='same')
ljtide_CL = ljtide - filtered_ljtide

ljtide_filt_series = pd.Series(filtered_ljtide, index=ljtide.index)
# NaN out the first and last 5 days to remove filter effects
ljtide_filt_series[:5*24] = np.nan
# ljtide_filt_series[-5*24:] = np.nan

## Plot the filtered tide data
%matplotlib qt
fig, ax = plt.subplots(1,1,figsize=(12,8),sharex=True)
ax.plot(ljtide_full.index, ljtide_full, label='Tide', color='black', alpha=0.5)
ax.plot(ljtide_filt_series.index, ljtide_filt_series, label='Filtered Tide', color='blue', alpha=0.5)
ax.set_ylabel('Tide (m, NAVD88)')
ax.legend()
ax.set_title('La Jolla Tide Gauge Data')

# plt.show()

#%% Create LJ Tide Filtered Series as a time series for hindcast
jan2000 = '2000-01-01'

ljtide_filt_series_fordailymax = ljtide_filt_series[jan2000:] + np.polyval(trend, ljtide_full[jan2000:].index.to_julian_date())

ljtide_dailymax_hindcast = pd.Series(ljtide_filt_series_fordailymax.resample('D').max(), index=ljtide_filt_series_fordailymax[jan2000:].resample('D').max().index)

## Save ljtide_dailymax_hindcast to hdf5 file
ljtide_dailymax_hindcast.to_hdf(path_to_ljtide_dailymax_hindcast,'ljtide_dailymax_hindcast')
print('La Jolla Tide Gauge Daily Maximum Values saved to: ' + path_to_ljtide_dailymax_hindcast)
#%% Create LJ Tide Filtered Series Daily Maximum Values for 24 realizations
oct2000 = '2000-10-01'

ljtide_dailymax = pd.Series(ljtide_filt_series[oct2000:].resample('D').max(), index=ljtide_filt_series[oct2000:].resample('D').max().index)

## Create new dataframe ljtide_dailymax_24 with columns for each year
dates = pd.date_range(start='2000-10-01', end='2024-09-30', freq='D')

ljtide_dailymax.index = dates

# Create a new dataframe with 366 rows and 24 columns
years = [f'Oct{year}' for year in range(2000, 2024)]
ljtide_dailymax_24 = pd.DataFrame(index=range(0, 366), columns=years)

# Iterate through each year from 2000 to 2023
for year in range(2000, 2024):
    # Define the start and end dates for each "water year" (Oct 1 to Sep 30)
    start_date = f'{year}-10-01'
    end_date = f'{year+1}-09-30'

    # Create a date range for the current year
    current_year_dates = pd.date_range(start=start_date, end=end_date, freq='D')

    # Get the values from the original series for this date range
    year_values = ljtide_dailymax.loc[current_year_dates]

    # Assign values to the dataframe, accounting for leap years
    ljtide_dailymax_24[f'Oct{year}'] = pd.Series(year_values.values)

    ## If the current year is a leap year, move all values in that column from rows 59-365 to rows 60-366
    if np.shape(current_year_dates)[0] == 365:
        ljtide_dailymax_24[f'Oct{year}'][152:366] = ljtide_dailymax_24[f'Oct{year}'][151:365]
        ljtide_dailymax_24[f'Oct{year}'][151] = np.nan
        
## Save ljtide_dailymax_24 to hdf5 file
ljtide_dailymax_24.to_hdf(path_to_ljtide_dailymax_24,'ljtide_dailymax_24')
print('La Jolla Tide Gauge Daily Maximum Values by Year saved to: ' + path_to_ljtide_dailymax_24)

# %%
