"""
Use UTide to analyze the Seacoast groundwater data
Create a function based on tidal component amplitudes and phases to be able to predict tidal component of
Seacoast groundwater

October 2024
Austin Barnes
"""
#%% Imports
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib
import numpy as np
import utide
from utide import solve, reconstruct

path_to_seacoastobs = '/Users/austinbarnes/Documents/UCSD SIO/IB Groundwater/ImperialBeach/data/seacoast_20240514_1124_QC.h5'
path_to_coef_lj = '/Users/austinbarnes/Documents/UCSD SIO/IB Groundwater/ImperialBeach/data/LJ_tide_data/coef_lj.pkl'
path_to_ljtide_full = '/Users/austinbarnes/Documents/UCSD SIO/IB Groundwater/ImperialBeach/data/LJ_tide_data/ljtide_1924.h5'
path_to_ljtide_2024_2100 = '/Users/austinbarnes/Documents/UCSD SIO/IB Groundwater/ImperialBeach/data/LJ_tide_data/ljtide_2100.h5'
path_to_coef_seacoast = '/Users/austinbarnes/Documents/UCSD SIO/IB Groundwater/ImperialBeach/data/seacoast_tide/coef_seacoast.pkl'
path_to_coef_seacoast2 = '/Users/austinbarnes/Documents/UCSD SIO/IB Groundwater/ImperialBeach/data/seacoast_tide/coef_seacoast2.pkl'
path_to_seacoast_tide_hindcast = '/Users/austinbarnes/Documents/UCSD SIO/IB Groundwater/ImperialBeach/data/seacoast_tide/seacoast_tide_hindcast.h5'
path_to_seacoast_tide_projections = '/Users/austinbarnes/Documents/UCSD SIO/IB Groundwater/ImperialBeach/data/seacoast_tide/seacoast_tide_projections.h5'

#%% Load the hydraulic head data
seacoast =  pd.read_hdf(path_to_seacoastobs).reset_index(drop=True)
#%% Load coef_lj with pickle
import pickle
with open(path_to_coef_lj, 'rb') as f:
    coef_lj = pickle.load(f)

#%% 6-min & Hourly averages of hydraulic head; converted to series seacoast_6min_series and seacoast_1hr_series
seacoast_6min = seacoast.resample('6T', on='Timestamps').mean().reset_index()
seacoast_6min_series = seacoast_6min.set_index('Timestamps')['NAVD88']

seacoast_1hr = seacoast.resample('H', on='Timestamps').mean().reset_index()
seacoast_1hr_series = seacoast_1hr.set_index('Timestamps')['NAVD88']
#%% Array of 64 days starting on 2021-12-15 (best continuous data with minimal precipitation) for UTide analysis
## Note: still several small precipitation events
startdate = '2021-12-15'
enddate = '2022-02-17'

seacoast_6min_64days = seacoast_6min[(seacoast_6min['Timestamps'] >= startdate) & (seacoast_6min['Timestamps'] < enddate)]
seacoast_6min_64days = seacoast_6min_64days.set_index('Timestamps')
#%% TESTBED: Assigning SA and SSA amplitudes

# t_seacoast = seacoast_6min_64days['Timestamps']
# u_seacoast = seacoast_6min_64days['NAVD88']

# # Define parameters
# omega_sa = 2 * np.pi / (365.25 * 86400)  # Frequency of SA in radians per second
# omega_ssa = 2 * np.pi / (182.625 * 86400)  # Frequency of SSA in radians per second

# # Define known amplitudes and phases for SA and SSA (replace these with your known values)
# amplitude_sa = coef_lj['A'][coef_lj['name']=='SA'][0]  # Amplitude of SA
# phase_sa = 2 * np.pi / coef_lj['g'][coef_lj['name']=='SA'][0] # Phase of SA in radians (from known time of max)
# amplitude_ssa = coef_lj['A'][coef_lj['name']=='SSA'][0] # Amplitude of SSA
# phase_ssa = 2 * np.pi / coef_lj['g'][coef_lj['name']=='SSA'][0]     # Phase of SSA in radians (from known time of max)

# # Calculate time in seconds from start
# t_seconds = (seacoast_6min_64days['Time'] - seacoast_6min_64days['Time'].iloc[0])*24*60*60

# # Generate SA and SSA time series
# sa_component = amplitude_sa * np.cos(omega_sa * t_seconds + phase_sa)
# ssa_component = amplitude_ssa * np.cos(omega_ssa * t_seconds + phase_ssa)

# # Sum the components to get the total known signal
# known_tidal_signal = sa_component + ssa_component

# # Subtract known signal from original data to get residual
# residual_data = seacoast_6min_64days['NAVD88'] - known_tidal_signal

# # Perform UTide analysis on the residual data
# coef = solve(
#     t_seacoast,
#     residual_data.values,
#     lat=...  # Latitude if relevant, else set to None
# )
#%% U-Tide Tidal Fits - 64 Days of low precipitation

t_seacoast = seacoast_6min_64days.index
u_seacoast = seacoast_6min_64days['NAVD88']

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

## Seacoast
coef_seacoast = solve(t_seacoast, u_seacoast, v=None,
             lat= 32.566627,
             nodal = True,
             trend = False,
             constit = tidal_constit,
             method = 'ols',
             conf_int='linear',
             order_constit='frequency',
             Rayleigh_min=0.95)

## Display constituents and amplitudes
for i in range(len(coef_seacoast['name'])):
    print(coef_seacoast['name'][i])
    print(str(coef_seacoast['A'][i]))

reconstruct_seacoast = reconstruct(seacoast_6min_series.index, coef_seacoast)
utide_seacoast = reconstruct_seacoast['h']-u_seacoast.iloc[0]
utide_seacoast_series = pd.Series(utide_seacoast, index=seacoast_6min_series.index)
seacoast_detide1 = seacoast_6min_series - utide_seacoast_series

tidal_constit = ['M2','S2','N2','K1','O1','P1','K2','Q1']

coef_seacoast2 = solve(t_seacoast, u_seacoast - reconstruct(t_seacoast, coef_seacoast)['h'], v=None,
             lat= 32.566627,
             nodal = True,
             trend = False,
             constit = tidal_constit,
             method = 'ols',
             conf_int='linear',
             Rayleigh_min=0.95)

## Display constituents and amplitudes
print('Seacoast Second Tidal Fit')
for i in range(len(coef_seacoast2['name'])):
    print(coef_seacoast2['name'][i])
    print(coef_seacoast2['A'][i])

reconstruct_seacoast2 = reconstruct(seacoast_6min_series.index, coef_seacoast2)
utide_seacoast2 = reconstruct_seacoast2['h']
utide_seacoast_series2 = pd.Series(utide_seacoast2, index=seacoast_6min_series.index)
seacoast_detide2 = seacoast_6min_series - utide_seacoast_series - utide_seacoast_series2

#%% Plot the Seacoast Detided Series to see improvement in residuals
%matplotlib qt
plt.figure()
seacoast_detide1.plot()
seacoast_detide2.plot()
plt.title('Seacoast Detide Series')
plt.show()

#%% Save the Utide coefficients
import pickle
with open(path_to_coef_seacoast, 'wb') as f:
    pickle.dump(coef_seacoast, f)

with open(path_to_coef_seacoast2, 'wb') as f:
    pickle.dump(coef_seacoast2, f)
#%% HINDCAST: Load the full La Jolla Tide Gauge Data, select 2000-01-01 to 2024-09-30
ljtide_full = pd.read_hdf(path_to_ljtide_full, 'ljtide')
print('Loaded La Jolla Tide Gauge Data from ' + path_to_ljtide_full)

ljtide_2000_2024 = ljtide_full[(ljtide_full.index >= pd.to_datetime('2000-01-01').tz_localize('UTC')) & (ljtide_full.index < pd.to_datetime('2024-10-01').tz_localize('UTC'))]

## Create series of the reconstructed tide with both sets of coefficients, remove mean
seacoast_tide_2000_2024 = reconstruct(ljtide_2000_2024.index, coef_seacoast)['h']
seacoast_tide_2000_2024 = seacoast_tide_2000_2024 - seacoast_tide_2000_2024.mean()
seacoast_tide2_2000_2024 = reconstruct(ljtide_2000_2024.index, coef_seacoast2)['h']
seacoast_tide2_2000_2024 = seacoast_tide2_2000_2024 - seacoast_tide2_2000_2024.mean()

seacoast_tide_2000_2024_series = pd.Series(seacoast_tide_2000_2024+seacoast_tide2_2000_2024, index=ljtide_2000_2024.index)

#%% HINDCAST: Save the hindcast seacoast tide
seacoast_tide_hindcast = seacoast_tide_2000_2024_series
seacoast_tide_hindcast.to_hdf(path_to_seacoast_tide_hindcast, 'seacoast_tide_hindcast')
#%% PROJECTED: Load Predicted LJ Tide
ljtide_2024_2100 = pd.read_hdf(path_to_ljtide_2024_2100, 'ljtide')
print('Loaded Predicted LJ Tide data from ' + path_to_ljtide_2024_2100)

seacoast_tide_2024_2100 = reconstruct(ljtide_2024_2100.index, coef_seacoast)['h']
seacoast_tide_2024_2100 = seacoast_tide_2024_2100 - seacoast_tide_2024_2100.mean()
seacoast_tide2_2024_2100 = reconstruct(ljtide_2024_2100.index, coef_seacoast2)['h']
seacoast_tide2_2024_2100 = seacoast_tide2_2024_2100 - seacoast_tide2_2024_2100.mean()

seacoast_tide_2024_2100_series = pd.Series(seacoast_tide_2024_2100+seacoast_tide2_2024_2100, index=ljtide_2024_2100.index)

#%% PROJECTED: Save the projected seacoast tide
seacoast_tide_projections = seacoast_tide_2024_2100_series
seacoast_tide_projections.to_hdf(path_to_seacoast_tide_projections, 'seacoast_tide_projections')