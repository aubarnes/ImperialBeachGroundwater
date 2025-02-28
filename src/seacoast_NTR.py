"""
Filter Seacoast GW table using cosine-Lanczos squared filter to get non-tidal residual

October 2024
Austin Barnes
"""
#%% Imports
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib
import numpy as np

path_to_seacoastobs = '../data/seacoast_20240514_1124_QC.h5'
path_to_seacoast_NTR = '../data/seacoast_NTR.h5'

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

#%% Load the hydraulic head data
seacoast =  pd.read_hdf(path_to_seacoastobs).reset_index(drop=True)
#%% 6-min & Hourly averages of hydraulic head; converted to series seacoast_6min_series and seacoast_1hr_series
seacoast_6min = seacoast.resample('6T', on='Timestamps').mean().reset_index()
seacoast_6min_series = seacoast_6min.set_index('Timestamps')['NAVD88']

seacoast_1hr = seacoast.resample('H', on='Timestamps').mean().reset_index()
seacoast_1hr_series = seacoast_1hr.set_index('Timestamps')['NAVD88']
#%% Filter Seacoast & LJ Tide Gauge using the cosine-Lanczos squared filter
# cutoff_hours = 40
cutoff_hours = 36
fs = 1  # samples per hour

# Generate the filter coefficients
h = cosine_lanczos_squared_filter(cutoff_hours, fs)

## Apply the filter
## filtered_ is for the filtered signal (low pass)
## _CL stands for cosine-Lanczos squared tidal signal (high pass)
filtered_seacoast = np.convolve(seacoast_1hr_series, h, mode='same')
seacoast_tide_CL = seacoast_1hr_series - filtered_seacoast

seacoast_filt_series = pd.Series(filtered_seacoast, index=seacoast_1hr_series.index)
# NaN out the first and last 5 days
seacoast_filt_series[:5*24] = np.nan
seacoast_filt_series[-5*24:] = np.nan

## Name the filtered signal 'NAVD88'
seacoast_filt_series.name = 'NAVD88'

#%% Plot the filtered signal on top of the original signal
%matplotlib qt
fig, ax = plt.subplots(1, 1, figsize=(12, 6))
ax.plot(seacoast_1hr_series.index, seacoast_1hr_series, label='Original')
ax.plot(seacoast_filt_series.index, seacoast_filt_series, label='Filtered')
ax.set_ylabel('Water Level (m)')
ax.set_title('Seacoast Groundwater Table')
ax.legend()
plt.show()
#%% Save Seacoast Non-Tidal Residual
seacoast_filt_series.to_hdf(path_to_seacoast_NTR,'seacoast_NTR')
# %%
