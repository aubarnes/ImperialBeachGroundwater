"""
La Jolla Tide 2: Non-Tidal Residual Analysis
Austin Barnes 2024

1) Remove UTide reconstructed tide from record
2) Filter using Cosine-Lanczos Squared Filter to get non-tidal residual (NTR)
3) Create hindcast of NTR
4) Create ensemble of projections out to 2100 using observed NTR

NOTE: SA and SSA constituents must not contain the seasonal cycle not attributable to tides
"""
#%% Imports & Define Directory
import numpy as np
import pandas as pd
import netCDF4 as nc
import matplotlib.pyplot as plt
%matplotlib
from utide import reconstruct
import pickle

## input paths
path_to_ljtide_full = '../data/ljtide_1924.h5'
path_to_coef_lj = '../data/coef_lj.pkl'
path_to_ljtide_2100 = '../data/ljtide_2100.h5'

## output paths
path_to_ljntr_hindcast = '../data/ljntr_hindcast.h5'
path_to_ljntr_ensemble = '../data/ljntr_ensemble.h5'

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
#%% Load Full LJ Tide Data
ljtide_full = pd.read_hdf(path_to_ljtide_full, 'ljtide')
print('Loaded La Jolla Tide Gauge Data from: ' + path_to_ljtide_full)

#%% Load coef_lj with pickle
with open(path_to_coef_lj, 'rb') as f:
    coef_lj = pickle.load(f)
#%% UTide reconstruction for ljtide_full
t_recon = ljtide_full.index
u_recon = reconstruct(t_recon, coef_lj)['h']

#%% Create utide & ntr series
utide_full = pd.Series(u_recon, index=ljtide_full.index)
detide_full = ljtide_full - utide_full
#%% Plot tidal reconstruction against the data
%matplotlib qt
plt.figure()
plt.plot(ljtide_full,alpha=0.8)
plt.plot(utide_full,alpha=0.5)
plt.plot(detide_full,'k')
plt.show()

#%% Interpolate de-tided data, detrend, and filter to get NTR

## Interpolate NaNs in the data
## OPTION 1: Fill NaNs with mean
# detide_mean = detide_full.mean()
# detide_interp = detide_full.fillna(detide_mean)

## OPTION 2: Linear interpolation
detide_interp = detide_full.interpolate(method='linear')

## Compute and remove linear trend from tide data
trend = np.polyfit(detide_interp.index.to_julian_date(), detide_interp, 1)
detide_detrend = detide_interp - np.polyval(trend, detide_interp.index.to_julian_date())

## Filter the tide data using Cosine-Lanczos Squared Filter
cutoff_hours = 36
fs = 1  # samples per hour

# Generate the filter coefficients
h = cosine_lanczos_squared_filter(cutoff_hours, fs)

## Apply the filter
## filtered_ is for the filtered signal (low pass)
## _CL stands for cosine-Lanczos squared tidal signal (high pass)
ntr_full = np.convolve(detide_detrend, h, mode='same')

ntr_full_series = pd.Series(ntr_full, index=ljtide_full.index)
# NaN out the first and last 5 days to remove filter effects
# ljtide_filt_series[:5*24] = np.nan
# ljtide_filt_series[-5*24:] = np.nan

#%% Compute and plot the decorrelation time scale

def compute_decorrelation_timescale(series):
    # Compute the autocorrelation function
    autocorr = np.correlate(series - np.mean(series), series - np.mean(series), mode='full')
    autocorr = autocorr[autocorr.size // 2:]  # Keep only the positive lags
    autocorr /= autocorr[0]  # Normalize

    # Find the decorrelation timescale (integral of the autocorrelation function)
    decorrelation_timescale = np.sum(autocorr)
    return decorrelation_timescale, autocorr

## Create smaller series ntr_2000_2010
ntr_2000_2010 = ntr_full_series['2000-01-01':'2010-12-31']

# Assuming ntr_full_series is your pandas Series
# If you haven't loaded it yet, you would do so here
# ntr_full_series = pd.read_csv('your_data_file.csv', parse_dates=['date'], index_col='date')['ntr']

# Compute the decorrelation timescale
decorrelation_timescale, autocorr = compute_decorrelation_timescale(ntr_2000_2010.values)

autocorr_time = np.arange(0, len(autocorr))/24

# Plot the autocorrelation function
plt.figure(figsize=(10, 6))
plt.plot(autocorr_time,autocorr)
plt.axhline(y=1/np.e, color='r', linestyle='--', label='1/e threshold')
plt.title('Autocorrelation Function')
plt.xlabel('Lag (days)')
plt.ylabel('Autocorrelation')
plt.legend()

# Find the lag where autocorrelation drops below 1/e
e_folding_time = np.where(autocorr < 1/np.e)[0][0]

plt.text(0.7, 0.9, f'Decorrelation timescale: {decorrelation_timescale:.2f}', 
         transform=plt.gca().transAxes)
plt.text(0.7, 0.85, f'e-folding time: {e_folding_time}', 
         transform=plt.gca().transAxes)

plt.show()

print(f"Decorrelation timescale: {decorrelation_timescale:.2f}")
print(f"e-folding time: {e_folding_time}")

#%% Plot the filtered tide data
%matplotlib qt
fig, ax = plt.subplots(1,1,figsize=(12,8),sharex=True)
ax.plot(ljtide_full, label='Tide', color='black', alpha=0.5)
ax.plot(ntr_full_series, label='Non-tidal residual', color='blue', alpha=0.5)
ax.set_ylabel('Tide (m, NAVD88)')
ax.legend()
ax.set_title('La Jolla Tide Gauge Data')

plt.show()

#%% Save non-tidal residual time series for hindcast 2000-01-01 through 2024-09-30
jan2000 = '2000-01-01'

ntr_hindcast = ntr_full_series[jan2000:] + np.polyval(trend, ntr_full_series[jan2000:].index.to_julian_date())+utide_full.mean()

## Save ntr_hindcast to hdf5 file
ntr_hindcast.to_hdf(path_to_ljntr_hindcast,'ntr_hindcast')
print('La Jolla Tide Gauge Non-Tidal Residual Hindcast from 2000 to 2024 saved to: ' + path_to_ljntr_hindcast)
#%% Create LJ Tide Filtered Series Daily Maximum Values for 24 realizations
# oct2000 = '2000-10-01'

# ljtide_dailymax = pd.Series(ljtide_filt_series[oct2000:].resample('D').max(), index=ljtide_filt_series[oct2000:].resample('D').max().index)

## Create new dataframe ljtide_dailymax_24 with columns for each year
# dates = pd.date_range(start='2000-10-01', end='2024-09-30', freq='D')

# ljtide_dailymax.index = dates

# Create a new dataframe with 366 rows and columns for each valid year of NTR
years = np.concatenate([np.arange(1926, 1931), np.arange(1932, 1946), [1949], np.arange(1951, 1953), np.arange(1955, 1971), np.arange(1973, 1977), np.arange(1979, 2024)])
octyears = [f'Oct{year}' for year in years]
ntr_ensemble = pd.DataFrame(index=range(0, 366*24), columns=octyears)

# Iterate through each year
for year in years:
    # Define the start and end dates for each "water year" (Oct 1 to Sep 30)
    start_date = f'{year}-10-01'
    end_date = f'{year+1}-10-01'

    # Create a date range for the current year
    current_year_dates = pd.date_range(start=start_date, end=end_date, freq='H')
    ## Drop the last date
    current_year_dates = current_year_dates.drop(current_year_dates[-1])

    # Get the values from the original series for this date range
    year_values = ntr_full_series[start_date:end_date]

    # Assign values to the dataframe, accounting for leap years
    ntr_ensemble[f'Oct{year}'] = pd.Series(year_values.values)

    ## If the current year is a leap year, move all values in that column from rows 59-365 to rows 60-366
    if np.shape(current_year_dates)[0] == 366*24:
        ntr_ensemble[f'Oct{year}'][152*24:366*24] = ntr_ensemble[f'Oct{year}'][151*24:365*24]
        ntr_ensemble[f'Oct{year}'][151*24] = np.nan

#%% Save ntr_ensemble to hdf5 file
        
## Save ljtide_dailymax_24 to hdf5 file
ntr_ensemble.to_hdf(path_to_ljntr_ensemble,'ntr_ensemble')
print('La Jolla Tide Gauge Non-Tidal Residual Ensemble from 1926 to 2024 saved to: ' + path_to_ljntr_ensemble)

#%% Plot the ensemble of NTR

## Compute the mean of the ensemble
ntr_ensemble_mean = ntr_ensemble.mean(axis=1)
## Compute the standard deviation of the ensemble
ntr_ensemble_std = ntr_ensemble.std(axis=1)

%matplotlib qt
fig, ax = plt.subplots(1,1,figsize=(12,8),sharex=True)
ntr_ensemble.plot(ax=ax, legend=False, color='k', alpha=0.2)
# ax.plot(ntr_ensemble['Oct1930'], color='red', label='1930')
# ax.plot(ntr_ensemble['Oct1940'], color='red', label='1940')
# ax.plot(ntr_ensemble['Oct1957'], color='red', label='1957')
# ax.plot(ntr_ensemble['Oct1965'], color='red', label='1965')
# # ax.plot(ntr_ensemble['Oct1972'], color='red', label='1972') ## Not in record
# ax.plot(ntr_ensemble['Oct1982'], color='red', label='1982')
# ax.plot(ntr_ensemble['Oct1987'], color='red', label='1987')
# ax.plot(ntr_ensemble['Oct1997'], color='red', label='1997')
# ax.plot(ntr_ensemble['Oct1998'], color='purple', label='1998')
# ax.plot(ntr_ensemble['Oct2015'], color='red', label='2015')
ax.plot(ntr_ensemble['Oct2023'], color='red', label='2023')
ax.plot(ntr_ensemble_mean, color='blue', label='Mean')
ax.fill_between(ntr_ensemble.index, ntr_ensemble_mean - ntr_ensemble_std, ntr_ensemble_mean + ntr_ensemble_std, color='blue', alpha=0.2, label='1 Std. Dev.')
ax.set_ylabel('NTR (m)')
ax.set_title('La Jolla Tide Gauge Non-Tidal Residual Ensemble\nBy Water Year')
## Make 0 on x-axis show "October 1" and last value show "September 30, following year"
## Include monthly ticks and labels
ax.set_xticks(np.arange(0, 366*24, 366*24//12))
ax.set_xticklabels(['Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep'])
plt.show()


# %%
