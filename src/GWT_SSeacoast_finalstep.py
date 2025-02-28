"""
Create the Hindcast and Projected Groundwater Table
MOP 38: S Seacoast Drive

Hindcast: 2000-01-01 to 2024-09-30
Projected: 2024-10-01 to 2100-01-01

Hindcast Groundwater Table Elevation = predicted GW tide + Pastas(observed LJ Tide Gauge non-tidal residual + observed precip - observed ETo)
200xProjected Groundwater Table Elevation = annual mean sea level + predicted GW tide + 25xPastas([NTR] + 8x(projected precip - projected ETo)
"""
#%% Imports & Define Directory
from os import error
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib qt
import pastas as ps
import matplotlib.dates as mdates
import matplotlib.patheffects as pe
from scipy.stats import norm
import pickle

## Values derived from 2016 DEM
roadelevation_sseacoast = 1.9 # meters (1.846 according to GPS at well location)
roadelevation_encanto = 2.08 # meters (approx low from DEM)
roadelevation_descanso = 2.27 # meters (approx low from DEM)
roadelevation_cortez = 2.13 # meters (approx low from DEM)
roadelevation_palm = 2.0 # meters (approx low from DEM)
roadelevation_carnation = 1.5 # meters (approx low from DEM)

## Observations
path_to_gwt_observations = '../data/seacoast_20240514_1124_QC.h5'
path_to_ljtide_full = '../data/ljtide_1924.h5'

## Used in Hincast & Projections
path_to_ljtide_2100 = '../data/ljtide_2100.h5'
path_to_ETo_2100 = '../data/ETo_2100.h5'
path_to_precip = '../data/precip_2100.h5'

## Hindcast Only
path_to_ljntr_hindcast = '../data/ljntr_hindcast.h5'

## Projections Only
path_to_slr_interp = '../data/slr_interp.pkl'
path_to_ljntr_ensemble = '../data/ljntr_ensemble.h5'
path_to_cmip6_ensemble = '../data/cmip6_ensemble.pkl'

#%% Observations: Load SSeacoast gwt obs and full LJ tide gauge
gwt_obs_data = pd.read_hdf(path_to_gwt_observations)
gwt_NAVD88 = pd.Series(gwt_obs_data['NAVD88'])
gwt_NAVD88.index = pd.to_datetime(gwt_obs_data['Timestamps'])

gwt_NAVD88 = gwt_NAVD88.resample('H').mean()

## Load the full LJ tide gauge data
ljtide_full = pd.read_hdf(path_to_ljtide_full)
ljtide_full = ljtide_full.resample('H').mean()
## Drop time-zone information
ljtide_full.index = ljtide_full.index.tz_localize(None)

# ## Plot the Groundwater Table Observations
# %matplotlib qt
# fig, ax = plt.subplots(1,1,figsize=(12,8),sharex=True)
# ax.plot(gwt_NAVD88, label='Groundwater Table', color='C0')
# ax.set_ylabel('(m, NAVD88)')
# ax.legend()
# ax.set_title('Seacoast Drive Groundwater Table Observations')
# plt.show()
#%% Hindcast & Projections: Load Precipitation, ETo, and LJ Tide
precip_data = pd.read_hdf(path_to_precip)
## Drop time-zone information
precip_data.index = precip_data.index.tz_localize(None)
## Observed precipitation data from any column of precip_data
precip_obs = precip_data['2003-10-01':'2024-09-30'].iloc[:,0]
## Name the series 'precip'
precip_obs.name = 'precip'

ETo_data = pd.read_hdf(path_to_ETo_2100)

## Create an hourly time series of ETo_data that downsamples the daily ETo_data to hourly by spreading the daily value across the hours
ETo_data_hourly = (ETo_data/24).resample('H').ffill()
## Spatial ETo data from any column of ETo_data_hourly
ETo_obs_hourly = ETo_data_hourly['2003-10-01':'2024-09-30'].iloc[:,0]
## Name the series 'ETo'
ETo_obs_hourly.name = 'ETo'

ljtide_2100 = pd.read_hdf(path_to_ljtide_2100)
#%% Hindcast: Load the LJ NTR Hindcast
ljntr_hindcast = pd.read_hdf(path_to_ljntr_hindcast)
#%% Projections: Load SLR Curve Interpolation Functions
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
#%% Projections: Create SLR Curves for 2024-2100
t_2024_2100 = ljtide_2100['2024-10-01':'2100-09-30'].index

slr_low_17p_2100 = slr_low_17p_interp(t_2024_2100.to_julian_date())
slr_low_17p_2100 = pd.Series(slr_low_17p_2100, index=t_2024_2100)
slr_low_50p_2100 = slr_low_50p_interp(t_2024_2100.to_julian_date())
slr_low_50p_2100 = pd.Series(slr_low_50p_2100, index=t_2024_2100)
slr_low_83p_2100 = slr_low_83p_interp(t_2024_2100.to_julian_date())
slr_low_83p_2100 = pd.Series(slr_low_83p_2100, index=t_2024_2100)

slr_intlow_17p_2100 = slr_intlow_17p_interp(t_2024_2100.to_julian_date())
slr_intlow_17p_2100 = pd.Series(slr_intlow_17p_2100, index=t_2024_2100)
slr_intlow_50p_2100 = slr_intlow_50p_interp(t_2024_2100.to_julian_date())
slr_intlow_50p_2100 = pd.Series(slr_intlow_50p_2100, index=t_2024_2100)
slr_intlow_83p_2100 = slr_intlow_83p_interp(t_2024_2100.to_julian_date())
slr_intlow_83p_2100 = pd.Series(slr_intlow_83p_2100, index=t_2024_2100)

slr_int_17p_2100 = slr_int_17p_interp(t_2024_2100.to_julian_date())
slr_int_17p_2100 = pd.Series(slr_int_17p_2100, index=t_2024_2100)
slr_int_50p_2100 = slr_int_50p_interp(t_2024_2100.to_julian_date())
slr_int_50p_2100 = pd.Series(slr_int_50p_2100, index=t_2024_2100)
slr_int_83p_2100 = slr_int_83p_interp(t_2024_2100.to_julian_date())
slr_int_83p_2100 = pd.Series(slr_int_83p_2100, index=t_2024_2100)

slr_inthigh_17p_2100 = slr_inthigh_17p_interp(t_2024_2100.to_julian_date())
slr_inthigh_17p_2100 = pd.Series(slr_inthigh_17p_2100, index=t_2024_2100)
slr_inthigh_50p_2100 = slr_inthigh_50p_interp(t_2024_2100.to_julian_date())
slr_inthigh_50p_2100 = pd.Series(slr_inthigh_50p_2100, index=t_2024_2100)
slr_inthigh_83p_2100 = slr_inthigh_83p_interp(t_2024_2100.to_julian_date())
slr_inthigh_83p_2100 = pd.Series(slr_inthigh_83p_2100, index=t_2024_2100)

slr_high_17p_2100 = slr_high_17p_interp(t_2024_2100.to_julian_date())
slr_high_17p_2100 = pd.Series(slr_high_17p_2100, index=t_2024_2100)
slr_high_50p_2100 = slr_high_50p_interp(t_2024_2100.to_julian_date())
slr_high_50p_2100 = pd.Series(slr_high_50p_2100, index=t_2024_2100)
slr_high_83p_2100 = slr_high_83p_interp(t_2024_2100.to_julian_date())
slr_high_83p_2100 = pd.Series(slr_high_83p_2100, index=t_2024_2100)

## Plot the SLR Curves for 2024-2100
%matplotlib qt
fig, ax = plt.subplots(1,1,figsize=(12,8),sharex=True)
ax.plot(t_2024_2100, slr_low_50p_2100, label='Low', color='blue')
ax.plot(t_2024_2100, slr_intlow_50p_2100, label='Intermediate Low', color='green')
ax.plot(t_2024_2100, slr_int_50p_2100, label='Intermediate', color='orange')
ax.plot(t_2024_2100, slr_inthigh_50p_2100, label='Intermediate High', color='red')
ax.plot(t_2024_2100, slr_high_50p_2100, label='High', color='purple')
ax.set_ylabel('Mean Sea Level (m, NAVD88)')
ax.legend()
ax.set_title('Projected SLR Curves from 2024-2100')
plt.show()

#%% Befus (2020): MHHW and LMSL
## Create time vector from 2020 to 2100 with annual frequency
t_2020_2100 = pd.date_range(start='2020-01-01', end='2100-01-01', freq='A')
slr_int_50p_2020_2100 = slr_int_50p_interp(t_2020_2100.to_julian_date())
## Supplemental
slr_high_50p_2020_2100 = slr_high_50p_interp(t_2020_2100.to_julian_date())
slr_inthigh_50p_2020_2100 = slr_inthigh_50p_interp(t_2020_2100.to_julian_date())
slr_intlow_50p_2020_2100 = slr_intlow_50p_interp(t_2020_2100.to_julian_date())
slr_low_50p_2020_2100 = slr_low_50p_interp(t_2020_2100.to_julian_date())

## Befus predicts linear increase with SLR @ Seacoast
## Starts at 1.56 m in 2020
befus_mhhw_int = 1.56 + slr_int_50p_2020_2100 - slr_int_50p_2020_2100[0]
befus_mhhw_int_series = pd.Series(befus_mhhw_int, index=t_2020_2100)

## Supplemental
befus_mhhw_high = 1.56 + slr_high_50p_2020_2100 - slr_high_50p_2020_2100[0]
befus_mhhw_high_series = pd.Series(befus_mhhw_high, index=t_2020_2100)
befus_mhhw_inthigh = 1.56 + slr_inthigh_50p_2020_2100 - slr_inthigh_50p_2020_2100[0]
befus_mhhw_inthigh_series = pd.Series(befus_mhhw_inthigh, index=t_2020_2100)
befus_mhhw_intlow = 1.56 + slr_intlow_50p_2020_2100 - slr_intlow_50p_2020_2100[0]
befus_mhhw_intlow_series = pd.Series(befus_mhhw_intlow, index=t_2020_2100)
befus_mhhw_low = 1.56 + slr_low_50p_2020_2100 - slr_low_50p_2020_2100[0]
befus_mhhw_low_series = pd.Series(befus_mhhw_low, index=t_2020_2100)

#%% Projections: Load the ensemble of Non-Tidal Residuals
ljntr_ensemble = pd.read_hdf(path_to_ljntr_ensemble)
print('Loaded ensemble of LJ Non-Tidal Residuals, # full years:',len(ljntr_ensemble.columns))
#%% Projections: each column is a realization of Non-Tidal Residual
ensemble_size_ntr = 25
ensemble_size_CMIP = 8
ensemble_size_tot = ensemble_size_ntr * ensemble_size_CMIP

ljntr_2024_2100 = pd.DataFrame(index=t_2024_2100, columns=range(ensemble_size_ntr))

## For leap years, we need to interpolate the NaNs in ntr_dailymax
ljntr_leapyr = ljntr_ensemble.copy()
## Fill in NaNs in ntr_dailymax using linear interpolation
ljntr_leapyr.interpolate(method='linear', inplace=True,axis=0, limit_direction='both')
## Drop column names and replace with integers
ljntr_leapyr.columns = range(ljntr_ensemble.shape[1])

## For non-leap years, we need to drop Feb 29 rows
ljntr_nonleapyr = ljntr_ensemble.copy()
## Drop rows 151*24 to 151*24+24
ljntr_nonleapyr.drop(ljntr_nonleapyr.index[151*24:151*24+24], inplace=True)
## Drop column names and replace with integers
ljntr_nonleapyr.columns = range(ljntr_ensemble.shape[1])

### Create ensemble by concatenating random draws of the ljntr_ensemble['Octyear'] for each year
for year in range(2024,2100):
    print(year)
    ## Pull indices for Oct 1 of year to Sep 30 of year + 1
    ljntr_2024_2100_indices = ljntr_2024_2100.index[
        (ljntr_2024_2100.index >= np.datetime64(f'{year}-10-01')) &
        (ljntr_2024_2100.index < np.datetime64(f'{year + 1}-10-01'))]
    ## if leap year (in practice 1 year ahead of the start year since we start Oct 1)
    ## A year is a leap year if it is divisible by 4, but not divisible by 100, unless it is also divisible by 400
    if (year + 1) % 4 == 0 and ((year + 1) % 100 != 0 or (year + 1) % 400 == 0):
        for i in range(ensemble_size_ntr):
            random_i = np.random.randint(0, len(ljntr_leapyr.columns))
            ntr = ljntr_leapyr.iloc[:, random_i]
            ## Make day_values.index = each day starting on Oct 1 of year and ending on Sep 30 of year + 1
            ntr.index = pd.date_range(start=f'10-01-{year}', end=f'10-01-{year+1}', freq='H').drop(pd.to_datetime(f'{year+1}-10-01'))
            ljntr_2024_2100.loc[ljntr_2024_2100_indices, i] = ntr.values
    ## Non-leap years (We skip leap years if the centurial year is not divisible by 400)
    else:
        for i in range(ensemble_size_ntr):
            random_i = np.random.randint(0, len(ljntr_nonleapyr.columns))
            ntr = ljntr_nonleapyr.iloc[:, random_i]
            ## Make day_values.index = each day starting on Oct 1 of year and ending on Sep 30 of year + 1
            ntr.index = pd.date_range(start=f'10-01-{year}', end=f'10-01-{year+1}', freq='H').drop(pd.to_datetime(f'{year+1}-10-01'))
            ljntr_2024_2100.loc[ljntr_2024_2100_indices, i] = ntr.values

## Make all values into floats
ljntr_2024_2100 = ljntr_2024_2100.astype(float)
## Localize to UTC time zone
ljntr_2024_2100.index = ljntr_2024_2100.index.tz_localize('UTC')

## In each column, concatenate the ntr_dailymax_hindcast
ljntr_2000_2100 = pd.DataFrame(index=ljntr_hindcast.index.append(ljntr_2024_2100.index).unique(), columns=range(ensemble_size_ntr))
for i in range(ensemble_size_ntr):
    ljntr_2000_2100[i] = pd.concat([ljntr_hindcast, ljntr_2024_2100[i]])
#%% Model: Create Stress Models

## Stress Model - Recharge (Precipitation + Evapotranspiration)
sm_recharge = ps.RechargeModel(
    prec = precip_obs,
    evap = ETo_obs_hourly,
    rfunc=ps.Gamma(),
    name="recharge",
    # recharge = ps.rch.FlexModel(),
    recharge = ps.rch.Linear(),
    # recharge = ps.rch.Peterson(),
    # recharge = ps.rch.Berendrecht(),
    settings=("prec", "evap"))

## Stress Model - Precipitation ONLY
sm_precip = ps.StressModel(
    stress = precip_obs,
    rfunc=ps.Gamma(),
    name="precip",
    up=True,  # head goes up if it rains
    settings="prec")

## Stress Model - Evapotranspiration ONLY
sm_ETo = ps.StressModel(
    stress = ETo_obs_hourly, # - ETo_data.min(),
    rfunc=ps.Gamma(),
    name="ETo",
    up=False,  # head goes down if it is hot
    settings="evap")

## Stress Model - Full LJ Tide
sm_ljtide = ps.StressModel(
    stress = ljtide_full,
    rfunc=ps.Gamma(),
    name="ljtide",
    settings="waterlevel")

## Stress Model - Non-Tidal Residual (HINDCAST)
sm_ntr = ps.StressModel(
    stress = ljntr_hindcast.tz_localize(None), # - ntr_dailymax_hindcast.min(),
    # rfunc=ps.Gamma(),
    rfunc=ps.Exponential(),
    # rfunc = ps.One(),
    name="ljntr",
    settings="waterlevel")

## Stress Model - LJ Tide Hindcast + Projection
sm_ljtide_2100 = ps.StressModel(
    stress = ljtide_2100,
    rfunc=ps.Exponential(),
    # rfunc = ps.One(),
    name="ljtide_2100",
    settings="waterlevel")
#%% Model: Create model, configure stresses, and solve
ml = ps.Model(gwt_NAVD88,name='gwt_NAVD88')
ml.settings["freq"] = "1H"

## Model: Configure stress models
ml.add_stressmodel(sm_recharge) ## OPTION 1
# ml.add_stressmodel(sm_precip) ## OPTION 2
# ml.add_stressmodel(sm_ETo) ## OPTION 2
# ml.add_stressmodel(sm_ljtide) ## FULL LJ TIDE GAUGE (ntr and tide)
ml.add_stressmodel(sm_ntr) ## NTR
ml.add_stressmodel(sm_ljtide_2100) ## LJ TIDE 2100

## Model: Solve the model
calib_start = "2021-12-08" ## Beginning of all data
# calib_start = "2022-01-01" ## Beginning of first 4 month rainy period
# calib_start = "2022-11-01" ## Beginning of last 50% of data
# calib_start = "2023-12-15" ## Beginning of last 4 month rainy period
# calib_start = "2024-01-01" ## Beginning of last 25% of data

# calib_end = "2022-04-21" ## End of first 25% of data
# calib_end = "2022-04-30" ## End of first 4 month rainy period
# calib_end = "2022-10-30" ## End of first 50% of data
# calib_end = "2024-04-15" ## End of last 4 month rainy period
calib_end = "2024-05-15" ## End of all data

## Days between 2001-10-01 and 2021-12-08
# warmupdays = (pd.to_datetime(calib_start) - pd.to_datetime("2001-10-01")).days
warmupdays = 365*3
ml.solve(warmup = warmupdays, tmin=calib_start,tmax=calib_end,freq=ml.settings["freq"], report=True)
%matplotlib qt
ml.plots.results() 

## Add noisemodel for comparison
# ml.add_noisemodel(ps.ArNoiseModel())
# ml.solve(initial=False)
# %matplotlib qt
# ml.plots.results() 
#%% Model: Create time series of the calibration and hindcast components
hindcast_freq = "1H"
# hindcast_freq = "1D"

data4comparison = gwt_NAVD88.tz_localize('UTC')

hindcast_start = "2021-12-08" ## Beginning of all data
hindcast_end = "2024-09-30" ## End of all data

hindcast = ml.simulate(warmup = warmupdays,tmin=hindcast_start,tmax=hindcast_end,freq=hindcast_freq)
## Localize to UTC time zone
hindcast = hindcast.tz_localize('UTC')

## Individual Stress Components

## Tide
tide_stress = ml.get_contributions(split=False)[2]
tide_simulation = tide_stress - tide_stress.mean()

## Recharge
recharge_stress = ml.get_contributions(split=False)[0]
recharge_simulation = recharge_stress - recharge_stress.min()

## Non-Tidal Residual
ntr_stress = ml.get_contributions(split=False)[1]
ntr_simulation = ntr_stress + ml.parameters['optimal']['constant_d'] + tide_stress.mean() + recharge_stress.min()

## Calculate RMSE and R^2
mse = np.mean((data4comparison - hindcast)**2)
rmse = np.sqrt(mse)
print(f'RMSE: {rmse}')
r2 = 1 - np.sum((data4comparison - hindcast)**2) / np.sum((data4comparison - data4comparison.mean())**2)
print(f'R^2: {r2}')
## Maximum residual
max_resid = np.max(np.abs(data4comparison - hindcast))
print(f'Max Residual: {max_resid}')
## Calculate NMSE and 1-NMSE
nmse = mse / np.var(data4comparison)
print(f'NMSE: {nmse}')
print(f'1-NMSE: {1 - nmse}')

#%% Figure 4: Plot Hindcast, Observations, Residuals, and Contributions from Stresses
fontsize = 16
%matplotlib qt
fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, figsize=(12, 16), sharex=True, gridspec_kw={'height_ratios': [2, 1, 1, 1, 1]})

## Modeled and Observed Groundwater Table Elevation (NAV88)
ax1.plot(hindcast, label='Modeled',color = 'black', alpha=1.0, linewidth=0.3)
ax1.plot(data4comparison, label='Observed', alpha=0.6, color = 'C0', linewidth=0.5)
# ax1.plot(tide_simulation + ntr_simulation + recharge_simulation, label='Modeled Tide + NTR + Recharge', color='red', alpha=0.2)
## Shade in the calibration period
# ax1.axvspan(pd.to_datetime(calib_start), pd.to_datetime(calib_end), color='gray', alpha=0.2)
# ax1.text(pd.to_datetime(calib_start), 0.45, 'Calibration Period', verticalalignment='top', horizontalalignment='left', color='k', fontsize=12)
ax1.set_ylabel('Groundwater Table\nElevation (m, NAVD88)', fontsize=fontsize-2)
ax1.set_title('Well 1 Hourly Groundwater Table Model', fontsize=fontsize)
legend = ax1.legend(loc='upper center', ncol=2, fontsize=fontsize-2)
legend.legend_handles[0].set_linewidth(3)
legend.legend_handles[1].set_linewidth(3)
ax1.grid(True)
ax1.set_ylim([0.3, 2.0])
ax1.text(-0.08, 1.05, 'a', transform=ax1.transAxes, fontsize=fontsize, fontweight='bold', va='top', ha='right')

## Model Residuals
# ax2.scatter((data4comparison - hindcast).index, data4comparison - hindcast, color='k', label='Residual', s=2)
ax2.plot(data4comparison - hindcast, color='k', label='Residual', linewidth=0.3)
# ax3 = ax2.twinx()
# ax3.plot(R2)
ax2.grid(which='both', ls='dotted')
ax2.set_ylim([-0.2, 0.2])
ax2.set_ylabel('(m)', fontsize=fontsize-2)
ax2.set_title('Model Residuals (Observation - Modeled)', fontsize=fontsize-2)
ax2.yaxis.set_minor_locator(plt.MultipleLocator(0.1))
ax2.text(-0.08, 1.05, 'b', transform=ax2.transAxes, fontsize=fontsize, fontweight='bold', va='top', ha='right')

## Tide Contribution
ax3.plot(tide_simulation, label='Tide Simulation', color='k', linewidth=0.3)
# ax3.scatter(tide_simulation.index,tide_simulation, label='Tide Simulation', color='k')
ax3.grid(which='both')
ax3.set_ylabel('(m)', fontsize=fontsize-2)
ax3.set_title('Ocean Tide Contribution', fontsize=fontsize-2)
ax3.text(-0.08, 1.05, 'c', transform=ax3.transAxes, fontsize=fontsize, fontweight='bold', va='top', ha='right')

## Non-tidal Contribution
ax4.plot(ntr_simulation, label='NTR Simulation', color='k')
ax4.set_ylabel('(m, NAVD88)', fontsize=fontsize-2)
ax4.set_title('Non-Tidal Sea Level Contribution', fontsize=fontsize-2)
ax4.yaxis.set_major_locator(plt.MultipleLocator(0.25))
# ax4.yaxis.set_minor_locator(plt.MultipleLocator(0.25))
ax4.set_ylim([0.7, 1.3])
ax4.grid(which='both')
ax4.text(-0.08, 1.05, 'd', transform=ax4.transAxes, fontsize=fontsize, fontweight='bold', va='top', ha='right')

## Recharge Contribution
ax5.plot(recharge_simulation, label='Recharge Simulation', color='k')
ax5.set_ylabel('(m)', fontsize=fontsize-2)
ax5.set_xlabel('Date', fontsize=fontsize-2)
ax5.set_title('Recharge Contribution', fontsize=fontsize-2)
ax5.set_ylim([0, 0.3])
ax5.yaxis.set_major_locator(plt.MultipleLocator(0.1))
# ax5.yaxis.set_minor_locator(plt.MultipleLocator(0.1))
ax5.grid(which='both')
ax5.text(-0.08, 1.05, 'e', transform=ax5.transAxes, fontsize=fontsize, fontweight='bold', va='top', ha='right')

## Restrict x-axis
# plt.xlim([data4comparison.index[0], data4comparison.index[-1]])
plt.xlim(pd.Timestamp('2021-12-01'), pd.Timestamp('2024-06-01'))
plt.gcf().autofmt_xdate()

# Set xticks and xtick labels
# ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
ax5.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

# Set tick label font size
ax1.tick_params(axis='both', which='major', labelsize=fontsize-2)
ax2.tick_params(axis='both', which='major', labelsize=fontsize-2)
ax3.tick_params(axis='both', which='major', labelsize=fontsize-2)
ax4.tick_params(axis='both', which='major', labelsize=fontsize-2)
ax5.tick_params(axis='both', which='major', labelsize=fontsize-2)

fig.tight_layout()

plt.show()
#%% Compare High Tide between model and data
# resample_freq = '6H'
# resample_freq = '12H'
resample_freq = '1D'
## Resample maxima of the data
seacoast_max = data4comparison.resample(resample_freq).max()
## Resample maxima of the model
hindcast_max = hindcast.resample(resample_freq).max()
## Truncate hindcast_max to match the length of seacoast_max
hindcast_max = hindcast_max.loc[seacoast_max.index]

## Compute RMSE
rmse = np.sqrt(np.mean((seacoast_max - hindcast_max)**2))
print(f'RMSE of DAILY maxima: {rmse}')
## Compute R^2
r2 = 1 - np.sum((seacoast_max - hindcast_max)**2) / np.sum((seacoast_max - np.mean(seacoast_max))**2)
print(f'R^2 of DAILY maxima: {r2}')
## Scatter plot of data maxima vs model maxima
%matplotlib
plt.figure(figsize=(8,8))
plt.scatter(seacoast_max,hindcast_max)
plt.plot([0,2],[0,2],color='red')
plt.xlabel('Seacoast Data Maxima (m)')
plt.ylabel('Seacoast Model Maxima (m)')
plt.title('Seacoast Data vs Model Maxima')
## Restrict x and y axes to range [0.7 1.9]
# plt.xlim([0.7,1.9])
# plt.ylim([0.7,1.9])
plt.show()

#%% Hindcast: Create full hindcast groundwater table
hindcast_freq = "1H"

data4comparison = gwt_NAVD88.tz_localize('UTC')

# hindcast_2003 = ml.simulate(warmup = warmupdays,tmin="2001-10-01",tmax="2024-09-30T23:00:00",freq=hindcast_freq)
hindcast_2003 = ml.simulate(tmin="2003-10-01",tmax="2024-09-30T23:00:00",freq=hindcast_freq)
## Localize to UTC time zone
hindcast_2003 = hindcast_2003.tz_localize('UTC')
#%% Projections: Create ensemble of non-tidal residuals for 2024-2100
ml.settings["freq"] = "1H"

## Create 5 scenarios of ntr + annual mean sea level
ntr_low_17p = ljntr_2000_2100.copy()
ntr_low = ljntr_2000_2100.copy()
ntr_low_83p = ljntr_2000_2100.copy()

ntr_intlow_17p = ljntr_2000_2100.copy()
ntr_intlow = ljntr_2000_2100.copy()
ntr_intlow_83p = ljntr_2000_2100.copy()

ntr_int_17p = ljntr_2000_2100.copy()
ntr_int = ljntr_2000_2100.copy()
ntr_int_83p = ljntr_2000_2100.copy()

ntr_inthigh_17p = ljntr_2000_2100.copy()
ntr_inthigh = ljntr_2000_2100.copy()
ntr_inthigh_83p = ljntr_2000_2100.copy()

ntr_high_17p = ljntr_2000_2100.copy()
ntr_high = ljntr_2000_2100.copy()
ntr_high_83p = ljntr_2000_2100.copy()

for i in range(ensemble_size_ntr):
    ntr_low_17p[i] = slr_low_17p_2100.tz_localize('UTC').reindex(ntr_low_17p.index, method='ffill').add(ntr_low_17p[i], fill_value=0)
    ntr_low[i] = slr_low_50p_2100.tz_localize('UTC').reindex(ntr_low.index, method='ffill').add(ntr_low[i], fill_value=0)
    ntr_low_83p[i] = slr_low_83p_2100.tz_localize('UTC').reindex(ntr_low_83p.index, method='ffill').add(ntr_low_83p[i], fill_value=0)

    ntr_intlow_17p[i] = slr_intlow_17p_2100.tz_localize('UTC').reindex(ntr_intlow_17p.index, method='ffill').add(ntr_intlow_17p[i], fill_value=0)
    ntr_intlow[i] = slr_intlow_50p_2100.tz_localize('UTC').reindex(ntr_intlow.index, method='ffill').add(ntr_intlow[i], fill_value=0)
    ntr_intlow_83p[i] = slr_intlow_83p_2100.tz_localize('UTC').reindex(ntr_intlow_83p.index, method='ffill').add(ntr_intlow_83p[i], fill_value=0)

    ntr_int_17p[i] = slr_int_17p_2100.tz_localize('UTC').reindex(ntr_int_17p.index, method='ffill').add(ntr_int_17p[i], fill_value=0)
    ntr_int[i] = slr_int_50p_2100.tz_localize('UTC').reindex(ntr_int.index, method='ffill').add(ntr_int[i], fill_value=0)
    ntr_int_83p[i] = slr_int_83p_2100.tz_localize('UTC').reindex(ntr_int_83p.index, method='ffill').add(ntr_int_83p[i], fill_value=0)

    ntr_inthigh_17p[i] = slr_inthigh_17p_2100.tz_localize('UTC').reindex(ntr_inthigh_17p.index, method='ffill').add(ntr_inthigh_17p[i], fill_value=0)
    ntr_inthigh[i] = slr_inthigh_50p_2100.tz_localize('UTC').reindex(ntr_inthigh.index, method='ffill').add(ntr_inthigh[i], fill_value=0)
    ntr_inthigh_83p[i] = slr_inthigh_83p_2100.tz_localize('UTC').reindex(ntr_inthigh_83p.index, method='ffill').add(ntr_inthigh_83p[i], fill_value=0)

    ntr_high_17p[i] = slr_high_17p_2100.tz_localize('UTC').reindex(ntr_high_17p.index, method='ffill').add(ntr_high_17p[i], fill_value=0)
    ntr_high[i] = slr_high_50p_2100.tz_localize('UTC').reindex(ntr_high.index, method='ffill').add(ntr_high[i], fill_value=0)
    ntr_high_83p[i] = slr_high_83p_2100.tz_localize('UTC').reindex(ntr_high_83p.index, method='ffill').add(ntr_high_83p[i], fill_value=0)

## Make all values into floats
ntr_low_17p = ntr_low_17p.astype(float)
ntr_low = ntr_low.astype(float)
ntr_low_83p = ntr_low_83p.astype(float)

ntr_intlow_17p = ntr_intlow_17p.astype(float)
ntr_intlow = ntr_intlow.astype(float)
ntr_intlow_83p = ntr_intlow_83p.astype(float)

ntr_int_17p = ntr_int_17p.astype(float)
ntr_int = ntr_int.astype(float)
ntr_int_83p = ntr_int_83p.astype(float)

ntr_inthigh_17p = ntr_inthigh_17p.astype(float)
ntr_inthigh = ntr_inthigh.astype(float)
ntr_inthigh_83p = ntr_inthigh_83p.astype(float)

ntr_high_17p = ntr_high_17p.astype(float)
ntr_high = ntr_high.astype(float)
ntr_high_83p = ntr_high_83p.astype(float)

model_low_17p = pd.DataFrame(index=t_2024_2100, columns=range(ensemble_size_tot))
model_low = pd.DataFrame(index=t_2024_2100, columns=range(ensemble_size_tot))
model_low_83p = pd.DataFrame(index=t_2024_2100, columns=range(ensemble_size_tot))

model_intlow_17p = pd.DataFrame(index=t_2024_2100, columns=range(ensemble_size_tot))
model_intlow = pd.DataFrame(index=t_2024_2100, columns=range(ensemble_size_tot))
model_intlow_83p = pd.DataFrame(index=t_2024_2100, columns=range(ensemble_size_tot))

model_int_17p = pd.DataFrame(index=t_2024_2100, columns=range(ensemble_size_tot))
model_int = pd.DataFrame(index=t_2024_2100, columns=range(ensemble_size_tot))
model_int_83p = pd.DataFrame(index=t_2024_2100, columns=range(ensemble_size_tot))

model_inthigh_17p = pd.DataFrame(index=t_2024_2100, columns=range(ensemble_size_tot))
model_inthigh = pd.DataFrame(index=t_2024_2100, columns=range(ensemble_size_tot))
model_inthigh_83p = pd.DataFrame(index=t_2024_2100, columns=range(ensemble_size_tot))

model_high_17p = pd.DataFrame(index=t_2024_2100, columns=range(ensemble_size_tot))
model_high = pd.DataFrame(index=t_2024_2100, columns=range(ensemble_size_tot))
model_high_83p = pd.DataFrame(index=t_2024_2100, columns=range(ensemble_size_tot))

gwt_low_17p = pd.DataFrame(index=t_2024_2100, columns=range(ensemble_size_tot))
gwt_low = pd.DataFrame(index=t_2024_2100, columns=range(ensemble_size_tot))
gwt_low_83p = pd.DataFrame(index=t_2024_2100, columns=range(ensemble_size_tot))

gwt_intlow_17p = pd.DataFrame(index=t_2024_2100, columns=range(ensemble_size_tot))
gwt_intlow = pd.DataFrame(index=t_2024_2100, columns=range(ensemble_size_tot))
gwt_intlow_83p = pd.DataFrame(index=t_2024_2100, columns=range(ensemble_size_tot))

gwt_int_17p = pd.DataFrame(index=t_2024_2100, columns=range(ensemble_size_tot))
gwt_int = pd.DataFrame(index=t_2024_2100, columns=range(ensemble_size_tot))
gwt_int_83p = pd.DataFrame(index=t_2024_2100, columns=range(ensemble_size_tot))

gwt_inthigh_17p = pd.DataFrame(index=t_2024_2100, columns=range(ensemble_size_tot))
gwt_inthigh = pd.DataFrame(index=t_2024_2100, columns=range(ensemble_size_tot))
gwt_inthigh_83p = pd.DataFrame(index=t_2024_2100, columns=range(ensemble_size_tot))

gwt_high_17p = pd.DataFrame(index=t_2024_2100, columns=range(ensemble_size_tot))
gwt_high = pd.DataFrame(index=t_2024_2100, columns=range(ensemble_size_tot))
gwt_high_83p = pd.DataFrame(index=t_2024_2100, columns=range(ensemble_size_tot))

#%% Projections: Pastas rolloff effect

stress_ntr_low = slr_low_50p_2100
stress_ntr_low = stress_ntr_low.tz_localize(None)
ml.stressmodels['ljntr'] = ps.StressModel(
    stress=stress_ntr_low,
    # rfunc=ps.Gamma(),
    # rfunc=ps.Exponential(),
    rfunc = ps.One(),
    name="ljntr_low",
    settings="waterlevel")
model_slronly_low = ml.simulate(warmup=warmupdays, tmin="2024-10-01", tmax="2100-10-01", freq=ml.settings["freq"]).reindex(model_low.index)
ntr_slronly_low = ml.get_contribution("ljntr", tmin="2024-10-01", tmax="2100-10-01")
offset_low = stress_ntr_low - ntr_slronly_low
offset_low = offset_low - offset_low.min()
offset_low = offset_low.drop(offset_low.index[-1])

stress_ntr_intlow = slr_intlow_50p_2100
stress_ntr_intlow = stress_ntr_intlow.tz_localize(None)
ml.stressmodels['ljntr'] = ps.StressModel(
    stress=stress_ntr_intlow,
    # rfunc=ps.Gamma(),
    # rfunc=ps.Exponential(),
    rfunc = ps.One(),
    name="ljntr_intlow",
    settings="waterlevel")
model_slronly_intlow = ml.simulate(warmup=warmupdays, tmin="2024-10-01", tmax="2100-10-01", freq=ml.settings["freq"]).reindex(model_intlow.index)
ntr_slronly_intlow = ml.get_contribution("ljntr", tmin="2024-10-01", tmax="2100-10-01")
offset_intlow = stress_ntr_intlow - ntr_slronly_intlow
offset_intlow = offset_intlow - offset_intlow.min()
offset_intlow = offset_intlow.drop(offset_intlow.index[-1])

stress_ntr_int = slr_int_50p_2100
stress_ntr_int = stress_ntr_int.tz_localize(None)
ml.stressmodels['ljntr'] = ps.StressModel(
    stress=stress_ntr_int,
    # rfunc=ps.Gamma(),
    # rfunc=ps.Exponential(),
    rfunc = ps.One(),
    name="ljntr_int",
    settings="waterlevel")
model_slronly_int = ml.simulate(warmup=warmupdays, tmin="2024-10-01", tmax="2100-10-01", freq=ml.settings["freq"]).reindex(model_int.index)
ntr_slronly_int = ml.get_contribution("ljntr", tmin="2024-10-01", tmax="2100-10-01")
offset_int = stress_ntr_int - ntr_slronly_int
offset_int = offset_int - offset_int.min()
offset_int = offset_int.drop(offset_int.index[-1])

stress_ntr_inthigh = slr_inthigh_50p_2100
stress_ntr_inthigh = stress_ntr_inthigh.tz_localize(None)
ml.stressmodels['ljntr'] = ps.StressModel(
    stress=stress_ntr_inthigh,
    # rfunc=ps.Gamma(),
    # rfunc=ps.Exponential(),
    rfunc = ps.One(),
    name="ljntr_inthigh",
    settings="waterlevel")
model_slronly_inthigh = ml.simulate(warmup=warmupdays, tmin="2024-10-01", tmax="2100-10-01", freq=ml.settings["freq"]).reindex(model_inthigh.index)
ntr_slronly_inthigh = ml.get_contribution("ljntr", tmin="2024-10-01", tmax="2100-10-01")
offset_inthigh = stress_ntr_inthigh - ntr_slronly_inthigh
offset_inthigh = offset_inthigh - offset_inthigh.min()
offset_inthigh = offset_inthigh.drop(offset_inthigh.index[-1])

stress_ntr_high = slr_high_50p_2100
stress_ntr_high = stress_ntr_high.tz_localize(None)
ml.stressmodels['ljntr'] = ps.StressModel(
    stress=stress_ntr_high,
    # rfunc=ps.Gamma(),
    # rfunc=ps.Exponential(),
    rfunc = ps.One(),
    name="ljntr_high",
    settings="waterlevel")
model_slronly_high = ml.simulate(warmup=warmupdays, tmin="2024-10-01", tmax="2100-10-01", freq=ml.settings["freq"]).reindex(model_high.index)
ntr_slronly_high = ml.get_contribution("ljntr", tmin="2024-10-01", tmax="2100-10-01")
offset_high = stress_ntr_high - ntr_slronly_high
offset_high = offset_high - offset_high.min()
offset_high = offset_high.drop(offset_high.index[-1])

#%% Projections: Run model for realizations of projected groundwater table (Freq = 1H)

for ntr_num in range(ensemble_size_ntr):
    for cmip_num in range(ensemble_size_CMIP):
        i = ntr_num * ensemble_size_CMIP + cmip_num

        ## New Recharge stressmodel with CMIP6 projection realization
        stress_ETo = pd.Series(ETo_data_hourly.iloc[:, cmip_num].values, index=pd.to_datetime(ETo_data_hourly.index))
        stress_precip = pd.Series(precip_data.iloc[:, cmip_num].values, index=pd.to_datetime(precip_data.index))
        ml.stressmodels['recharge'] = ps.RechargeModel(
            prec=stress_precip,
            evap=stress_ETo,
            name="recharge",
            rfunc=ps.Gamma(),
            recharge=ps.rch.Linear(),
            settings=("prec", "evap"))
        
        ## New NTR stressmodel with projection realization for each SLR scenario
        
        stress_ntr = pd.Series(ntr_low.iloc[:, ntr_num].values, index=pd.to_datetime(ntr_low.index))
        stress_ntr = stress_ntr.tz_localize(None)
        ml.stressmodels['ljntr'] = ps.StressModel(
            stress=stress_ntr,
            rfunc=ps.Exponential(),
            name="ljntr_low",
            settings="waterlevel")
        model_low.iloc[:, i] = ml.simulate(tmin="2024-10-01", tmax="2100-10-01", freq=ml.settings["freq"]).reindex(model_low.index)
        gwt_low.iloc[:, i] = model_low.iloc[:, i] + offset_low

        stress_ntr = pd.Series(ntr_intlow.iloc[:, ntr_num].values, index=pd.to_datetime(ntr_intlow.index))
        stress_ntr = stress_ntr.tz_localize(None)
        ml.stressmodels['ljntr'] = ps.StressModel(
            stress=stress_ntr,
            rfunc=ps.Exponential(),
            name="ljntr_intlow",
            settings="waterlevel")
        model_intlow.iloc[:, i] = ml.simulate(tmin="2024-10-01", tmax="2100-10-01", freq=ml.settings["freq"]).reindex(model_intlow.index)
        gwt_intlow.iloc[:, i] = model_intlow.iloc[:, i] + offset_intlow

        stress_ntr = pd.Series(ntr_int.iloc[:, ntr_num].values, index=pd.to_datetime(ntr_int.index))
        stress_ntr = stress_ntr.tz_localize(None)
        ml.stressmodels['ljntr'] = ps.StressModel(
            stress=stress_ntr,
            rfunc=ps.Exponential(),
            name="ljntr_int",
            settings="waterlevel")
        model_int.iloc[:, i] = ml.simulate(warmup=warmupdays, tmin="2024-10-01", tmax="2100-10-01", freq=ml.settings["freq"]).reindex(model_int.index)
        gwt_int.iloc[:, i] = model_int.iloc[:, i] + offset_int

        stress_ntr = pd.Series(ntr_inthigh.iloc[:, ntr_num].values, index=pd.to_datetime(ntr_inthigh.index))
        stress_ntr = stress_ntr.tz_localize(None)
        ml.stressmodels['ljntr'] = ps.StressModel(
            stress=stress_ntr,
            rfunc=ps.Exponential(),
            name="ljntr_inthigh",
            settings="waterlevel")
        model_inthigh.iloc[:, i] = ml.simulate(tmin="2024-10-01", tmax="2100-10-01", freq=ml.settings["freq"]).reindex(model_inthigh.index)
        gwt_inthigh.iloc[:, i] = model_inthigh.iloc[:, i] + offset_inthigh

        stress_ntr = pd.Series(ntr_high.iloc[:, ntr_num].values, index=pd.to_datetime(ntr_high.index))
        stress_ntr = stress_ntr.tz_localize(None)
        ml.stressmodels['ljntr'] = ps.StressModel(
            stress=stress_ntr,
            rfunc=ps.Exponential(),
            name="ljntr_high",
            settings="waterlevel")
        model_high.iloc[:, i] = ml.simulate(tmin="2024-10-01", tmax="2100-10-01", freq=ml.settings["freq"]).reindex(model_high.index)
        gwt_high.iloc[:, i] = model_high.iloc[:, i] + offset_high

    ## Make all values into floats
    gwt_low = gwt_low.astype(float)
    gwt_intlow = gwt_intlow.astype(float)
    gwt_int = gwt_int.astype(float)
    gwt_inthigh = gwt_inthigh.astype(float)
    gwt_high = gwt_high.astype(float)

#%% Projections: Resample the projected groundwater table to the daily max
gwt_hindcast_dailymax = hindcast_2003.resample('D').max()
gwt_hindcast_dailymax_annualmean = gwt_hindcast_dailymax.resample('Y').mean()
gwt_hindcast_monthlymax_annualmean = hindcast_2003.resample('M').max().resample('Y').mean()
gwt_hindcast_annualmax = hindcast_2003.resample('Y').max()

gwt_low_dailymax = gwt_low.resample('D').max()
gwt_intlow_dailymax = gwt_intlow.resample('D').max()
gwt_int_dailymax = gwt_int.resample('D').max()
gwt_inthigh_dailymax = gwt_inthigh.resample('D').max()
gwt_high_dailymax = gwt_high.resample('D').max()

gwt_low_dailymax_mean = gwt_low_dailymax.mean(axis=1, numeric_only=True)
gwt_low_dailymax_std = gwt_low_dailymax.std(axis=1, numeric_only=True)
gwt_intlow_dailymax_mean = gwt_intlow_dailymax.mean(axis=1, numeric_only=True)
gwt_intlow_dailymax_std = gwt_intlow_dailymax.std(axis=1, numeric_only=True)
gwt_int_dailymax_mean = gwt_int_dailymax.mean(axis=1, numeric_only=True)
gwt_int_dailymax_std = gwt_int_dailymax.std(axis=1, numeric_only=True)
gwt_inthigh_dailymax_mean = gwt_inthigh_dailymax.mean(axis=1, numeric_only=True)
gwt_inthigh_dailymax_std = gwt_inthigh_dailymax.std(axis=1, numeric_only=True)
gwt_high_dailymax_mean = gwt_high_dailymax.mean(axis=1, numeric_only=True)
gwt_high_dailymax_std = gwt_high_dailymax.std(axis=1, numeric_only=True)

## Annual average of daily maximum groundwater table ANNUAL MEAN OF DAILY MAXIMUM OF EACH ENSEMBLE
gwt_low_dailymax_annualmean = gwt_low_dailymax.resample('Y').mean()
gwt_intlow_dailymax_annualmean = gwt_intlow_dailymax.resample('Y').mean()
gwt_int_dailymax_annualmean = gwt_int_dailymax.resample('Y').mean()
gwt_inthigh_dailymax_annualmean = gwt_inthigh_dailymax.resample('Y').mean()
gwt_high_dailymax_annualmean = gwt_high_dailymax.resample('Y').mean()

#%% Uncertainty Estimation: Hindcast & Projections 17th & 83rd Percentiles

## Hindcast
## error1 = var(residuals) + error_obs
residuals = data4comparison - hindcast
error_obs = 0.04**2 ## 0.04 m estimated observational standard deviation
error1 = np.nanvar(residuals) + error_obs
twosigma_hindcast = 2*np.sqrt(error1)


## Projections (not including SLR curves) - ANNUAL MEAN DAILY MAXIMUM
## Variance over the ensembles at each time point and average of these variances over time
## error2 = mean(var(y_forecast))
error2_low_dailymax_annualmean = np.mean(np.var(gwt_low_dailymax_annualmean, axis = 1))
twosigma_projection_low = 2*np.sqrt(error1 + error2_low_dailymax_annualmean)
error2_intlow_dailymax_annualmean = np.mean(np.var(gwt_intlow_dailymax_annualmean, axis = 1))
twosigma_projection_intlow = 2*np.sqrt(error1 + error2_intlow_dailymax_annualmean)
error2_int_dailymax_annualmean = np.mean(np.var(gwt_int_dailymax_annualmean, axis = 1))
twosigma_projection_int = 2*np.sqrt(error1 + error2_int_dailymax_annualmean)
error2_inthigh_dailymax_annualmean = np.mean(np.var(gwt_inthigh_dailymax_annualmean, axis = 1))
twosigma_projection_inthigh = 2*np.sqrt(error1 + error2_inthigh_dailymax_annualmean)
error2_high_dailymax_annualmean = np.mean(np.var(gwt_high_dailymax_annualmean, axis = 1))
twosigma_projection_high = 2*np.sqrt(error1 + error2_high_dailymax_annualmean)

p83_ensemble_low = norm.ppf(0.83, loc=0, scale=twosigma_projection_low/2)
p83_ensemble_intlow = norm.ppf(0.83, loc=0, scale=twosigma_projection_intlow/2)
p83_ensemble_int = norm.ppf(0.83, loc=0, scale=twosigma_projection_int/2)
p83_ensemble_inthigh = norm.ppf(0.83, loc=0, scale=twosigma_projection_inthigh/2)
p83_ensemble_high = norm.ppf(0.83, loc=0, scale=twosigma_projection_high/2)

## SLR Curves: 83rd and 17th percentiles - ANNUAL MEAN DAILY MAXIMUM
p83_slr_low = (slr_low_83p_2100 - slr_low_50p_2100).resample('Y').mean()
p17_slr_low = (slr_low_50p_2100 - slr_low_17p_2100).resample('Y').mean()
p83_slr_intlow = (slr_intlow_83p_2100 - slr_intlow_50p_2100).resample('Y').mean()
p17_slr_intlow = (slr_intlow_50p_2100 - slr_intlow_17p_2100).resample('Y').mean()
p83_slr_int = (slr_int_83p_2100 - slr_int_50p_2100).resample('Y').mean()
p17_slr_int = (slr_int_50p_2100 - slr_int_17p_2100).resample('Y').mean()
p83_slr_inthigh = (slr_inthigh_83p_2100 - slr_inthigh_50p_2100).resample('Y').mean()
p17_slr_inthigh = (slr_inthigh_50p_2100 - slr_inthigh_17p_2100).resample('Y').mean()
p83_slr_high = (slr_high_83p_2100 - slr_high_50p_2100).resample('Y').mean()
p17_slr_high = (slr_high_50p_2100 - slr_high_17p_2100).resample('Y').mean()

## Final 83rd and 17th percentile of Hindcast and Projections - ANNUAL MEAN DAILY MAXIMUM
p83_hindcast = gwt_hindcast_dailymax_annualmean + norm.ppf(0.83, loc=0, scale=twosigma_hindcast/2)
p17_hindcast = gwt_hindcast_dailymax_annualmean + norm.ppf(0.17, loc=0, scale=twosigma_hindcast/2)

p50_projection_low = gwt_low_dailymax_annualmean.median(axis=1)
p83_projection_low = gwt_low_dailymax_annualmean.median(axis=1) + p83_ensemble_low + p83_slr_low
p17_projection_low = gwt_low_dailymax_annualmean.median(axis=1) -p83_ensemble_low - p17_slr_low

p50_projection_intlow = gwt_intlow_dailymax_annualmean.median(axis=1)
p83_projection_intlow = gwt_intlow_dailymax_annualmean.median(axis=1) + p83_ensemble_intlow + p83_slr_intlow
p17_projection_intlow = gwt_intlow_dailymax_annualmean.median(axis=1) -p83_ensemble_intlow - p17_slr_intlow

p50_projection_int = gwt_int_dailymax_annualmean.median(axis=1)
p83_projection_int = gwt_int_dailymax_annualmean.median(axis=1) + p83_ensemble_int + p83_slr_int
p17_projection_int = gwt_int_dailymax_annualmean.median(axis=1) -p83_ensemble_int - p17_slr_int

p50_projection_inthigh = gwt_inthigh_dailymax_annualmean.median(axis=1)
p83_projection_inthigh = gwt_inthigh_dailymax_annualmean.median(axis=1) + p83_ensemble_inthigh + p83_slr_inthigh
p17_projection_inthigh = gwt_inthigh_dailymax_annualmean.median(axis=1) -p83_ensemble_inthigh - p17_slr_inthigh

p50_projection_high = gwt_high_dailymax_annualmean.median(axis=1)
p83_projection_high = gwt_high_dailymax_annualmean.median(axis=1) + p83_ensemble_high + p83_slr_high
p17_projection_high = gwt_high_dailymax_annualmean.median(axis=1) -p83_ensemble_high - p17_slr_high

## Projections (not including SLR curves) - DAILY MAXIMUM
## Variance over the ensembles at each time point and average of these variances over time
## error2 = mean(var(y_forecast))
error2_low_dailymax = np.mean(np.var(gwt_low_dailymax, axis = 1))
twosigma_projection_low_DM = 2*np.sqrt(error1 + error2_low_dailymax)
error2_intlow_dailymax = np.mean(np.var(gwt_intlow_dailymax, axis = 1))
twosigma_projection_intlow_DM = 2*np.sqrt(error1 + error2_intlow_dailymax)
error2_int_dailymax = np.mean(np.var(gwt_int_dailymax, axis = 1))
twosigma_projection_int_DM = 2*np.sqrt(error1 + error2_int_dailymax)
error2_inthigh_dailymax = np.mean(np.var(gwt_inthigh_dailymax, axis = 1))
twosigma_projection_inthigh_DM = 2*np.sqrt(error1 + error2_inthigh_dailymax)
error2_high_dailymax = np.mean(np.var(gwt_high_dailymax, axis = 1))
twosigma_projection_high_DM = 2*np.sqrt(error1 + error2_high_dailymax)

p83_ensemble_low_DM = norm.ppf(0.83, loc=0, scale=twosigma_projection_low_DM/2)
p83_ensemble_intlow_DM = norm.ppf(0.83, loc=0, scale=twosigma_projection_intlow_DM/2)
p83_ensemble_int_DM = norm.ppf(0.83, loc=0, scale=twosigma_projection_int_DM/2)
p83_ensemble_inthigh_DM = norm.ppf(0.83, loc=0, scale=twosigma_projection_inthigh_DM/2)
p83_ensemble_high_DM = norm.ppf(0.83, loc=0, scale=twosigma_projection_high_DM/2)

## SLR Curves: 83rd and 17th percentiles - DAILY MAXIMUM
p83_slr_low_DM = (slr_low_83p_2100 - slr_low_50p_2100).resample('D').mean()
p17_slr_low_DM = (slr_low_50p_2100 - slr_low_17p_2100).resample('D').mean()
p83_slr_intlow_DM = (slr_intlow_83p_2100 - slr_intlow_50p_2100).resample('D').mean()
p17_slr_intlow_DM = (slr_intlow_50p_2100 - slr_intlow_17p_2100).resample('D').mean()
p83_slr_int_DM = (slr_int_83p_2100 - slr_int_50p_2100).resample('D').mean()
p17_slr_int_DM = (slr_int_50p_2100 - slr_int_17p_2100).resample('D').mean()
p83_slr_inthigh_DM = (slr_inthigh_83p_2100 - slr_inthigh_50p_2100).resample('D').mean()
p17_slr_inthigh_DM = (slr_inthigh_50p_2100 - slr_inthigh_17p_2100).resample('D').mean()
p83_slr_high_DM = (slr_high_83p_2100 - slr_high_50p_2100).resample('D').mean()
p17_slr_high_DM = (slr_high_50p_2100 - slr_high_17p_2100).resample('D').mean()

## Final 83rd and 17th percentile of Hindcast and Projections - DAILY MAXIMUM
p83_hindcast_DM = gwt_hindcast_dailymax + norm.ppf(0.83, loc=0, scale=twosigma_hindcast/2)
p17_hindcast_DM = gwt_hindcast_dailymax + norm.ppf(0.17, loc=0, scale=twosigma_hindcast/2)

p50_projection_low_DM = gwt_low_dailymax.median(axis=1)
p83_projection_low_DM = gwt_low_dailymax.median(axis=1) + p83_ensemble_low_DM + p83_slr_low_DM
p17_projection_low_DM = gwt_low_dailymax.median(axis=1) -p83_ensemble_low_DM - p17_slr_low_DM

p50_projection_intlow_DM = gwt_intlow_dailymax.median(axis=1)
p83_projection_intlow_DM = gwt_intlow_dailymax.median(axis=1) + p83_ensemble_intlow_DM + p83_slr_intlow_DM
p17_projection_intlow_DM = gwt_intlow_dailymax.median(axis=1) -p83_ensemble_intlow_DM - p17_slr_intlow_DM

p50_projection_int_DM = gwt_int_dailymax.median(axis=1)
p83_projection_int_DM = gwt_int_dailymax.median(axis=1) + p83_ensemble_int_DM + p83_slr_int_DM
p17_projection_int_DM = gwt_int_dailymax.median(axis=1) -p83_ensemble_int_DM - p17_slr_int_DM

p50_projection_inthigh_DM = gwt_inthigh_dailymax.median(axis=1)
p83_projection_inthigh_DM = gwt_inthigh_dailymax.median(axis=1) + p83_ensemble_inthigh_DM + p83_slr_inthigh_DM
p17_projection_inthigh_DM = gwt_inthigh_dailymax.median(axis=1) -p83_ensemble_inthigh_DM - p17_slr_inthigh_DM

p50_projection_high_DM = gwt_high_dailymax.median(axis=1)
p83_projection_high_DM = gwt_high_dailymax.median(axis=1) + p83_ensemble_high_DM + p83_slr_high_DM
p17_projection_hig_DM = gwt_high_dailymax.median(axis=1) -p83_ensemble_high_DM - p17_slr_high_DM

#%% Plot the Intermediate scenario of the projected groundwater table - Daily Maximum
fontsize = 20
%matplotlib qt
fig, ax = plt.subplots(1,1,figsize=(18,18),sharex=True)

## Hindcast: daily maxima with 83rd and 17th percentiles
ax.plot(gwt_hindcast_dailymax, color='black', label='Hindcast', linewidth=0.3)
ax.fill_between(gwt_hindcast_dailymax.index, p17_hindcast_DM, p83_hindcast_DM, color='blue', alpha=0.4, label='17th-83rd Percentiles')

## Projections: daily maxima with 83rd and 17th percentiles
ax.plot(p50_projection_int_DM, color='purple', linestyle="-", linewidth=0.3, label='50th Percentile Projection')
ax.fill_between(p50_projection_int_DM.index, p17_projection_int_DM, p83_projection_int_DM, color='purple', alpha=0.2, label='17th-83rd Percentiles')

## Add horizontal lines at road elevations
ax.axhline(y=roadelevation_sseacoast, color='red', linestyle='--', label='Intersection Elevations')
ax.text(pd.to_datetime('2002-01-01'), roadelevation_sseacoast, 'S Seacoast', verticalalignment='bottom', horizontalalignment='left', color='red', fontsize=fontsize-2)
ax.axhline(y=roadelevation_descanso, color = 'seagreen', linestyle='--', label='D0041 Road Elevation')
ax.text(pd.to_datetime('2002-01-01'), roadelevation_descanso, 'Descanso', verticalalignment='bottom', horizontalalignment='left', color='seagreen', fontsize=fontsize-2)
ax.axhline(y=roadelevation_encanto, color = 'darkorange', linestyle='--', label='D0043 Road Elevation')
ax.text(pd.to_datetime('2002-01-01'), roadelevation_encanto-0.01, 'Encanto', verticalalignment='bottom', horizontalalignment='left', color='darkorange', fontsize=fontsize-2)
ax.axhline(y=roadelevation_cortez, color = 'darkgreen', linestyle='--', label='D0045 Road Elevation')
ax.text(pd.to_datetime('2002-01-01'), roadelevation_cortez, 'Cortez', verticalalignment='bottom', horizontalalignment='left', color='darkgreen', fontsize=fontsize-2)
ax.axhline(y=roadelevation_palm, color = 'tomato', linestyle='--', label='D0057 Road Elevation')
ax.text(pd.to_datetime('2002-01-01'), roadelevation_palm, 'Palm', verticalalignment='bottom', horizontalalignment='left', color='tomato', fontsize=fontsize-2)

order = [0, 2, 3, 4]
ax.set_title(f'Seacoast Groundwater Table\nIntermediate SLR Scenario\nDaily Maximum', fontsize=fontsize)

handles, labels = ax.get_legend_handles_labels()
legend = ax.legend([handles[idx] for idx in order], [labels[idx] for idx in order], loc='lower right', fontsize=fontsize-2)
for line in legend.get_lines():
    line.set_linewidth(3)

ax.set_ylabel('Groundwater Table Elevation\n(m, NAVD88)', fontsize=fontsize)
ax.tick_params(axis='both', which='major', labelsize=fontsize)
ax.set_xlim([pd.Timestamp('2000-01-01'), pd.Timestamp('2100-01-01')])
ax.set_xlabel('Year', fontsize=fontsize)
ax.grid(True)

plt.tight_layout()
plt.show()
#%% Plot the 5 scenarios of the projected groundwater table - Annual Mean of Daily Maximum
fontsize = 20
%matplotlib qt
fig, ax = plt.subplots(1,1,figsize=(18,18),sharex=True)

## Hindcast: annual mean daily maxima with 83rd and 17th percentiles
ax.plot(gwt_hindcast_dailymax_annualmean, color='black', linewidth=3, label='Hindcast')
ax.fill_between(gwt_hindcast_dailymax_annualmean.index, p17_hindcast, p83_hindcast, color='black', alpha=0.2, label='17th-83rd Percentiles')

## Projections: annual mean daily maxima with 83rd and 17th percentiles
ax.plot(p50_projection_low, color='darkgreen', linestyle=':', linewidth=3, label='Low Scenario')
ax.fill_between(gwt_low_dailymax_annualmean.index, p17_projection_low, p83_projection_low, color='darkgreen', alpha=0.1)
ax.plot(p50_projection_intlow, color='mediumblue', linestyle=':', linewidth=3, label='Intermediate Low Scenario')
ax.fill_between(gwt_intlow_dailymax_annualmean.index, p17_projection_intlow, p83_projection_intlow, color='mediumblue', alpha=0.1)
ax.plot(p50_projection_int, color='purple', linestyle=':', linewidth=3, label='Intermediate Scenario')
ax.fill_between(gwt_int_dailymax_annualmean.index, p17_projection_int, p83_projection_int, color='purple', alpha=0.1)
ax.plot(p50_projection_inthigh, color='darkorange', linestyle=':', linewidth=3, label='Intermediate High Scenario')
ax.fill_between(gwt_inthigh_dailymax_annualmean.index, p17_projection_inthigh, p83_projection_inthigh, color='darkorange', alpha=0.1)
ax.plot(p50_projection_high, color='#d62728', linestyle=':', linewidth=3, label='High Scenario')
ax.fill_between(gwt_high_dailymax_annualmean.index, p17_projection_high, p83_projection_high, color='#d62728', alpha=0.1)

## Add horizontal lines at road elevations
ax.axhline(y=roadelevation_sseacoast, color='red', linestyle='--', label='Road Elevation')
ax.text(pd.to_datetime('2002-01-01'), roadelevation_sseacoast, 'S Seacoast', verticalalignment='bottom', horizontalalignment='left', color='red', fontsize=fontsize)

ax.set_ylim([1.15, 3.5])
ax.set_yticks(np.arange(1.2, 3.51, 0.2))
ax.set_ylabel('Groundwater Table Elevation\n(m, NAVD88)', fontsize=fontsize)
handles, labels = ax.get_legend_handles_labels()
order = [0, 1, 6, 5, 4, 3, 2, 7]
ax.legend([handles[idx] for idx in order], [labels[idx] for idx in order], loc='upper left', fontsize=fontsize)
ax.set_title(f'Seacoast Groundwater Table Projections\nAnnual Mean of Daily Maximum', fontsize=fontsize)
ax.tick_params(axis='both', which='major', labelsize=fontsize)

ax.set_xlim([pd.Timestamp('2000-01-01'), pd.Timestamp('2100-01-01')])
ax.set_xlabel('Year', fontsize=fontsize)

ax.grid(True)

plt.tight_layout()
plt.show()
#%% SUPPLEMENTAL: Plot the hourly groundwater table for the Intermediate Scenario for 2082
### (the year that the annual mean of daily maxima exceeds S Seacoast road elevation)
fontsize = 20
%matplotlib qt
fig, ax = plt.subplots(1,1,figsize=(18,18),sharex=True)

ax.plot(gwt_int['2082-01-01':'2082-12-31'].mean(axis = 1), color='purple', linewidth=0.5, label='Ensemble Average')

## Add horizontal lines at road elevations
ax.axhline(y=roadelevation_sseacoast, color='red', linestyle='--', label='S Seacoast Road Elevation')
# ax.text(pd.to_datetime('2002-01-01'), roadelevation_sseacoast, 'D0038', verticalalignment='bottom', horizontalalignment='left', color='red', fontsize=fontsize-2)
# ax.text(pd.to_datetime('2084-01-02'), roadelevation_sseacoast, 'S Seacoast', verticalalignment='bottom', horizontalalignment='left', color='red', fontsize=fontsize)
# ax.axhline(y=roadelevation_descanso, color = 'C1', linestyle='--', label='Descanso, Road Elevation')
# ax.axhline(y=roadelevation_encanto, color = 'C2', linestyle='--', label='Encanto, Road Elevation')
# ax.axhline(y=roadelevation_cortez, color = 'C3', linestyle='--', label='Cortez, Road Elevation')
# ax.axhline(y=roadelevation_palm, color = 'C4', linestyle='--', label='Palm, Road Elevation')

# ax.plot(ljtide_2100,alpha=0.2,linewidth=0.5)

ax.set_ylim([0.95, 2.3])
# ax.set_ylim([0.9, 3.55])
# ax.set_yticks(np.arange(1.2, 3.51, 0.2))
ax.set_ylabel('Groundwater Table Elevation\n(m, NAVD88)', fontsize=fontsize)
handles, labels = ax.get_legend_handles_labels()
order = [0, 1]
ax.legend([handles[idx] for idx in order], [labels[idx] for idx in order], loc='upper center', fontsize=fontsize)
ax.set_title(f'Hourly Seacoast Groundwater Table Projections\nIntermediate SLR Scenario', fontsize=fontsize)
ax.tick_params(axis='both', which='major', labelsize=fontsize)

ax.set_xlim([pd.Timestamp('2082-01-01'), pd.Timestamp('2083-01-01')])
ax.set_xlabel('Date', fontsize=fontsize)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

ax.grid(True)

# plt.gcf().autofmt_xdate()
plt.tight_layout()
plt.show()

#%% Individual SLR scenario plots: Full Hourly Time Series (LOW)
fontsize=20
%matplotlib qt
fig, ax = plt.subplots(1,1,figsize=(18,18),sharex=True)

## Plot the hindcast
ax.plot(hindcast_2003, label='Hindcast', color='black',linewidth=0.3,alpha=1.0)

### LOW ###
ax.plot(gwt_low[0]//100, label = 'Projection Ensemble', color='grey', alpha=0.7)
for i in range(0, ensemble_size_tot):
    ax.plot(gwt_low[i], color='grey', linewidth=0.3, alpha=0.2)
ax.plot(gwt_low.mean(axis=1), label='Ensemble Mean', color='darkgreen', linewidth=0.3, alpha=0.4)
ax.set_title('Hourly Seacoast Groundwater Table Projections\n50$^{th}$ Percentile Low SLR Scenario', fontsize=fontsize)
ax.set_ylim([0.24, 2.9])
ax.set_yticks(np.arange(0.25, 2.76, 0.25))

## Add horizontal lines at road elevations
ax.axhline(y=roadelevation_sseacoast, color='red', linestyle='--', label='Intersection Elevations')
ax.text(pd.to_datetime('2002-01-01'), roadelevation_sseacoast, 'S Seacoast', verticalalignment='bottom', horizontalalignment='left', color='red', fontsize=fontsize-2)
ax.axhline(y=roadelevation_descanso, color = 'seagreen', linestyle='--')
ax.text(pd.to_datetime('2002-01-01'), roadelevation_descanso, 'Descanso', verticalalignment='bottom', horizontalalignment='left', color='darkgreen', fontsize=fontsize-2)
ax.axhline(y=roadelevation_encanto, color = 'darkorange', linestyle='--')
ax.text(pd.to_datetime('2002-01-01'), roadelevation_encanto, 'Encanto', verticalalignment='bottom', horizontalalignment='left', color='darkorange', fontsize=fontsize-7)
ax.axhline(y=roadelevation_cortez, color = 'darkgreen', linestyle='--')
ax.text(pd.to_datetime('2002-01-01'), roadelevation_cortez, 'Cortez', verticalalignment='bottom', horizontalalignment='left', color='seagreen', fontsize=fontsize-2)
ax.axhline(y=roadelevation_palm, color = 'tomato', linestyle='--')
ax.text(pd.to_datetime('2002-01-01'), roadelevation_palm, 'Palm', verticalalignment='bottom', horizontalalignment='left', color='tomato', fontsize=fontsize-3)

order = [0, 2, 1, 3]

handles, labels = ax.get_legend_handles_labels()
legend = ax.legend([handles[idx] for idx in order], [labels[idx] for idx in order], loc='upper left', fontsize=fontsize-2)
for line in legend.get_lines():
    line.set_linewidth(3)
ax.set_ylabel('Groundwater Table Elevation\n(m, NAVD88)', fontsize=fontsize)

ax.tick_params(axis='both', which='major', labelsize=fontsize)

ax.set_xlim([pd.Timestamp('2000-01-01'), pd.Timestamp('2100-01-01')])
ax.set_xlabel('Year', fontsize=fontsize)

ax.grid(True)

plt.tight_layout()
plt.show()
#%% Individual SLR scenario plots: Full Hourly Time Series (INT LOW)
fontsize=20
%matplotlib qt
fig, ax = plt.subplots(1,1,figsize=(18,18),sharex=True)

## Plot the hindcast
ax.plot(hindcast_2003, label='Hindcast', color='black',linewidth=0.3,alpha=1.0)

### INT LOW ###
ax.plot(gwt_intlow[0]//100, label = 'Projection Ensemble', color='grey', alpha=0.7)
for i in range(0, ensemble_size_tot):
    ax.plot(gwt_intlow[i], color='grey', linewidth=0.3, alpha=0.2)
ax.plot(gwt_intlow.mean(axis=1), label='Ensemble Mean', color='mediumblue', linewidth=0.3, alpha=0.4)
ax.set_title('Hourly Seacoast Groundwater Table Projections\n50$^{th}$ Percentile Intermediate Low SLR Scenario', fontsize=fontsize)
ax.set_ylim([0.24, 2.9])
ax.set_yticks(np.arange(0.25, 2.76, 0.25))

## Add horizontal lines at road elevations
ax.axhline(y=roadelevation_sseacoast, color='red', linestyle='--', label='Intersection Elevations')
ax.text(pd.to_datetime('2002-01-01'), roadelevation_sseacoast, 'S Seacoast', verticalalignment='bottom', horizontalalignment='left', color='red', fontsize=fontsize-2)
ax.axhline(y=roadelevation_descanso, color = 'seagreen', linestyle='--')
ax.text(pd.to_datetime('2002-01-01'), roadelevation_descanso, 'Descanso', verticalalignment='bottom', horizontalalignment='left', color='darkgreen', fontsize=fontsize-2)
ax.axhline(y=roadelevation_encanto, color = 'darkorange', linestyle='--')
ax.text(pd.to_datetime('2002-01-01'), roadelevation_encanto, 'Encanto', verticalalignment='bottom', horizontalalignment='left', color='darkorange', fontsize=fontsize-7)
ax.axhline(y=roadelevation_cortez, color = 'darkgreen', linestyle='--')
ax.text(pd.to_datetime('2002-01-01'), roadelevation_cortez, 'Cortez', verticalalignment='bottom', horizontalalignment='left', color='seagreen', fontsize=fontsize-2)
ax.axhline(y=roadelevation_palm, color = 'tomato', linestyle='--')
ax.text(pd.to_datetime('2002-01-01'), roadelevation_palm, 'Palm', verticalalignment='bottom', horizontalalignment='left', color='tomato', fontsize=fontsize-3)

order = [0, 2, 1, 3]

handles, labels = ax.get_legend_handles_labels()
legend = ax.legend([handles[idx] for idx in order], [labels[idx] for idx in order], loc='upper left', fontsize=fontsize-2)
for line in legend.get_lines():
    line.set_linewidth(3)
ax.set_ylabel('Groundwater Table Elevation\n(m, NAVD88)', fontsize=fontsize)

ax.tick_params(axis='both', which='major', labelsize=fontsize)

ax.set_xlim([pd.Timestamp('2000-01-01'), pd.Timestamp('2100-01-01')])
ax.set_xlabel('Year', fontsize=fontsize)

ax.grid(True)

plt.tight_layout()
plt.show()
#%% Individual SLR scenario plots: Full Hourly Time Series (INT)
fontsize=20
%matplotlib qt
fig, ax = plt.subplots(1,1,figsize=(18,18),sharex=True)

## Plot the hindcast
ax.plot(hindcast_2003, label='Hindcast', color='black',linewidth=0.3,alpha=1.0)

### INT ###
ax.plot(gwt_int[0]//100, label = 'Projection Ensemble', color='grey', alpha=0.7)
for i in range(0, ensemble_size_tot):
    ax.plot(gwt_int[i], color='grey', linewidth=0.3, alpha=0.2)
ax.plot(gwt_int.mean(axis=1), label='Ensemble Mean', color='purple', linewidth=0.3, alpha=0.6)
ax.set_title('Hourly Seacoast Groundwater Table Projections\n50$^{th}$ Percentile Intermediate SLR Scenario', fontsize=fontsize)
ax.set_ylim([0.24, 3.0])
ax.set_yticks(np.arange(0.25, 3.01, 0.25))

## Add horizontal lines at road elevations
ax.axhline(y=roadelevation_sseacoast, color='red', linestyle='--', label='Intersection Elevations')
ax.text(pd.to_datetime('2002-01-01'), roadelevation_sseacoast, 'S Seacoast', verticalalignment='bottom', horizontalalignment='left', color='red', fontsize=fontsize-2)
ax.axhline(y=roadelevation_descanso, color = 'seagreen', linestyle='--')
ax.text(pd.to_datetime('2002-01-01'), roadelevation_descanso, 'Descanso', verticalalignment='bottom', horizontalalignment='left', color='darkgreen', fontsize=fontsize-2)
ax.axhline(y=roadelevation_encanto, color = 'darkorange', linestyle='--')
ax.text(pd.to_datetime('2002-01-01'), roadelevation_encanto, 'Encanto', verticalalignment='bottom', horizontalalignment='left', color='darkorange', fontsize=fontsize-7)
ax.axhline(y=roadelevation_cortez, color = 'darkgreen', linestyle='--')
ax.text(pd.to_datetime('2002-01-01'), roadelevation_cortez, 'Cortez', verticalalignment='bottom', horizontalalignment='left', color='seagreen', fontsize=fontsize-2)
ax.axhline(y=roadelevation_palm, color = 'tomato', linestyle='--')
ax.text(pd.to_datetime('2002-01-01'), roadelevation_palm, 'Palm', verticalalignment='bottom', horizontalalignment='left', color='tomato', fontsize=fontsize-3)

order = [0, 2, 1, 3]

handles, labels = ax.get_legend_handles_labels()
legend = ax.legend([handles[idx] for idx in order], [labels[idx] for idx in order], loc='upper left', fontsize=fontsize-2)
for line in legend.get_lines():
    line.set_linewidth(3)
ax.set_ylabel('Groundwater Table Elevation\n(m, NAVD88)', fontsize=fontsize)

ax.tick_params(axis='both', which='major', labelsize=fontsize)

ax.set_xlim([pd.Timestamp('2000-01-01'), pd.Timestamp('2100-01-01')])
ax.set_xlabel('Year', fontsize=fontsize)

ax.grid(True)

plt.tight_layout()
plt.show()
#%% Individual SLR scenario plots: Full Hourly Time Series (INT HIGH)
fontsize=20
%matplotlib qt
fig, ax = plt.subplots(1,1,figsize=(18,18),sharex=True)

## Plot the hindcast
ax.plot(hindcast_2003, label='Hindcast', color='black',linewidth=0.3,alpha=1.0)

### INT HIGH ###
ax.plot(gwt_inthigh[0]//100, label = 'Projection Ensemble', color='grey', alpha=0.7)
for i in range(0, ensemble_size_tot):
    ax.plot(gwt_inthigh[i], color='grey', linewidth=0.3, alpha=0.2)
ax.plot(gwt_inthigh.mean(axis=1), label='Ensemble Mean', color='darkorange', linewidth=0.3, alpha=0.6)
ax.set_title('Hourly Seacoast Groundwater Table Projections\n50$^{th}$ Percentile Intermediate High SLR Scenario', fontsize=fontsize)
ax.set_ylim([0.24, 3.5])
ax.set_yticks(np.arange(0.25, 3.51, 0.25))

## Add horizontal lines at road elevations
ax.axhline(y=roadelevation_sseacoast, color='red', linestyle='--', label='Intersection Elevations')
ax.text(pd.to_datetime('2002-01-01'), roadelevation_sseacoast, 'S Seacoast', verticalalignment='bottom', horizontalalignment='left', color='red', fontsize=fontsize-2)
ax.axhline(y=roadelevation_descanso, color = 'seagreen', linestyle='--')
ax.text(pd.to_datetime('2002-01-01'), roadelevation_descanso, 'Descanso', verticalalignment='bottom', horizontalalignment='left', color='darkgreen', fontsize=fontsize-2)
ax.axhline(y=roadelevation_encanto, color = 'darkorange', linestyle='--')
ax.text(pd.to_datetime('2002-01-01'), roadelevation_encanto, 'Encanto', verticalalignment='bottom', horizontalalignment='left', color='darkorange', fontsize=fontsize-9)
ax.axhline(y=roadelevation_cortez, color = 'darkgreen', linestyle='--')
ax.text(pd.to_datetime('2002-01-01'), roadelevation_cortez, 'Cortez', verticalalignment='bottom', horizontalalignment='left', color='seagreen', fontsize=fontsize-2)
ax.axhline(y=roadelevation_palm, color = 'tomato', linestyle='--')
ax.text(pd.to_datetime('2002-01-01'), roadelevation_palm, 'Palm', verticalalignment='bottom', horizontalalignment='left', color='tomato', fontsize=fontsize-5)

order = [0, 2, 1, 3]

handles, labels = ax.get_legend_handles_labels()
legend = ax.legend([handles[idx] for idx in order], [labels[idx] for idx in order], loc='upper left', fontsize=fontsize-2)
for line in legend.get_lines():
    line.set_linewidth(3)
ax.set_ylabel('Groundwater Table Elevation\n(m, NAVD88)', fontsize=fontsize)

ax.tick_params(axis='both', which='major', labelsize=fontsize)

ax.set_xlim([pd.Timestamp('2000-01-01'), pd.Timestamp('2100-01-01')])
ax.set_xlabel('Year', fontsize=fontsize)

ax.grid(True)

plt.tight_layout()
plt.show()
#%% Individual SLR scenario plots: Full Hourly Time Series (HIGH)
fontsize=20
%matplotlib qt
fig, ax = plt.subplots(1,1,figsize=(18,18),sharex=True)

## Plot the hindcast
ax.plot(hindcast_2003, label='Hindcast', color='black',linewidth=0.3,alpha=1.0)

### HIGH ###
ax.plot(gwt_high[0]//100, label = 'Projection Ensemble', color='grey', alpha=0.7)
for i in range(0, ensemble_size_tot):
    ax.plot(gwt_high[i], color='grey', linewidth=0.3, alpha=0.2)
ax.plot(gwt_high.mean(axis=1), label='Ensemble Mean', color='#d62728', linewidth=0.3, alpha=0.6)
ax.set_title('Hourly Seacoast Groundwater Table Projections\n50$^{th}$ Percentile High SLR Scenario', fontsize=fontsize)
ax.set_ylim([0.24, 4.0])
ax.set_yticks(np.arange(0.25, 4.01, 0.25))

## Add horizontal lines at road elevations
ax.axhline(y=roadelevation_sseacoast, color='red', linestyle='--', label='Intersection Elevations')
ax.text(pd.to_datetime('2002-01-01'), roadelevation_sseacoast, 'S Seacoast', verticalalignment='bottom', horizontalalignment='left', color='red', fontsize=fontsize-2)
ax.axhline(y=roadelevation_descanso, color = 'seagreen', linestyle='--')
ax.text(pd.to_datetime('2002-01-01'), roadelevation_descanso, 'Descanso', verticalalignment='bottom', horizontalalignment='left', color='darkgreen', fontsize=fontsize-2)
ax.axhline(y=roadelevation_encanto, color = 'darkorange', linestyle='--')
ax.text(pd.to_datetime('2002-01-01'), roadelevation_encanto, 'Encanto', verticalalignment='bottom', horizontalalignment='left', color='darkorange', fontsize=fontsize-11)
ax.axhline(y=roadelevation_cortez, color = 'darkgreen', linestyle='--')
ax.text(pd.to_datetime('2002-01-01'), roadelevation_cortez, 'Cortez', verticalalignment='bottom', horizontalalignment='left', color='seagreen', fontsize=fontsize-2)
ax.axhline(y=roadelevation_palm, color = 'tomato', linestyle='--')
ax.text(pd.to_datetime('2002-01-01'), roadelevation_palm, 'Palm', verticalalignment='bottom', horizontalalignment='left', color='tomato', fontsize=fontsize-5)

order = [0, 2, 1, 3]

handles, labels = ax.get_legend_handles_labels()
legend = ax.legend([handles[idx] for idx in order], [labels[idx] for idx in order], loc='upper left', fontsize=fontsize-2)
for line in legend.get_lines():
    line.set_linewidth(3)
ax.set_ylabel('Groundwater Table Elevation\n(m, NAVD88)', fontsize=fontsize)

ax.tick_params(axis='both', which='major', labelsize=fontsize)

ax.set_xlim([pd.Timestamp('2000-01-01'), pd.Timestamp('2100-01-01')])
ax.set_xlabel('Year', fontsize=fontsize)

ax.grid(True)

plt.tight_layout()
plt.show()
#%% Flood metrics: S Seacoast - Intermediate Scenario

# Initialize results storage
full_results_int = []

seasonal_results = []

## Intermediate Scenario
for realization in gwt_int.columns:
    gwt_int_flooding = pd.DataFrame(gwt_int[realization])
    gwt_int_flooding.columns = ['gwt']
    gwt_int_flooding['year'] = gwt_int_flooding.index.year
    gwt_int_flooding['above_threshold'] = gwt_int_flooding['gwt'] > roadelevation_sseacoast
    gwt_int_flooding['flood_depth'] = gwt_int_flooding['gwt'] - roadelevation_sseacoast
    gwt_int_flooding['flood_depth'] = gwt_int_flooding['flood_depth'].apply(lambda x: x if x > 0 else np.NaN)
    ## Drop nans
    gwt_int_flooding = gwt_int_flooding.dropna()

    realization_results = []
    realization_seasonal_results = []

    for year, group in gwt_int_flooding.groupby('year'):
        group['event_start'] = (group.index.to_series().diff() != pd.Timedelta('1H'))
        # group['event_start'] = (group.index.to_series().diff().shift(-1) == pd.Timedelta('1H'))
        num_events = group['event_start'].sum()

        ## Metric 3a: Number of days emergent
        days_with_flooding = group.groupby(group.index.date)['above_threshold'].any().sum()

        ## Metric 4a: Total time emergent
        hours_above_threshold = group['above_threshold'].sum()

        ## Metric 2a: Duration of Events and Ponding Depths
        event_durations = []
        ponding_depths = []

        event_start_indices = group.index[group['event_start']].tolist()
        event_start_indices.append(group.index[-1] + pd.Timedelta('1H'))  # Add an end marker for the last event

        for i in range(len(event_start_indices) - 1):
            event_period = group.loc[event_start_indices[i]:event_start_indices[i + 1] - pd.Timedelta('1H')]
            event_durations.append(len(event_period))
            ponding_depths.append(event_period['flood_depth'].max())

        ## Seasonal metrics initialization
        winter_flooding = 0
        winter_day_flood = 0
        winter_night_flood = 0
        winter_both_flood = 0
        summer_flooding = 0
        summer_day_flood = 0
        summer_night_flood = 0
        summer_both_flood = 0

        # Calculate winter flooding
        winter_period = group[(group.index.month >= 10) | (group.index.month <= 3)]
        if not winter_period.empty:
            winter_flood_days = winter_period.groupby(winter_period.index.date)['above_threshold'].any()
            winter_flooding = winter_flood_days.sum()

            # Determine day/night/both flooding for winter period
            for day, has_flooding in winter_flood_days.items():
                if has_flooding:
                    day_data = winter_period.loc[str(day)]
                    day_flood = day_data.loc[(day_data.index.hour >= 12) & (day_data.index.hour < 24), 'above_threshold'].any()
                    night_flood = day_data.loc[(day_data.index.hour >= 0) & (day_data.index.hour < 12), 'above_threshold'].any()
                    if day_flood and night_flood:
                        winter_both_flood += 1
                    elif day_flood:
                        winter_day_flood += 1
                    elif night_flood:
                        winter_night_flood += 1

        # Calculate summer flooding
        summer_period = group[(group.index.month >= 4) & (group.index.month <= 9)]
        if not summer_period.empty:
            summer_flood_days = summer_period.groupby(summer_period.index.date)['above_threshold'].any()
            summer_flooding = summer_flood_days.sum()

            # Determine day/night/both flooding for summer period
            for day, has_flooding in summer_flood_days.items():
                if has_flooding:
                    day_data = summer_period.loc[str(day)]
                    day_flood = day_data.loc[(day_data.index.hour >= 12) & (day_data.index.hour < 24), 'above_threshold'].any()
                    night_flood = day_data.loc[(day_data.index.hour >= 0) & (day_data.index.hour < 12), 'above_threshold'].any()
                    if day_flood and night_flood:
                        summer_both_flood += 1
                    elif day_flood:
                        summer_day_flood += 1
                    elif night_flood:
                        summer_night_flood += 1

        ## Save Realization Results
        realization_results.append({
            'year': year,
            'ponding_depths': ponding_depths,
            'days_with_flooding': days_with_flooding,
            'hours_above_threshold': hours_above_threshold,
            'num_events': num_events,
            'event_durations': event_durations,
        })

        realization_seasonal_results.append({
            'year': year,
            'winter_flooding': winter_flooding,
            'winter_day_flood': winter_day_flood,
            'winter_night_flood': winter_night_flood,
            'winter_both_flood': winter_both_flood,
            'summer_flooding': summer_flooding,
            'summer_day_flood': summer_day_flood,
            'summer_night_flood': summer_night_flood,
            'summer_both_flood': summer_both_flood
        })

    full_results_int.append(pd.DataFrame(realization_results))
    seasonal_results.append(pd.DataFrame(realization_seasonal_results))

#%% Compute metrics from the full results by year S Seacoast - Intermediate Scenario
## Compute the minimum, maximum, 25th, 50th, and 75th percentiles

# Ensure all final results are numeric
for df in full_results_int:
    df.apply(pd.to_numeric, errors='coerce')

def safe_percentile(x, q):
    if len(x) > 0:
        return np.percentile(x, q) 
    else:
        return np.nan

## Concatenate all realizations into a single DataFrame
data_int = pd.concat(full_results_int)

## Compute ponding depth metrics
ponding_depths_int = data_int.groupby('year')['ponding_depths'].apply(list).reset_index()
ponding_depths_int.columns = ['year', 'ponding_depths_list']
ponding_depths_int['combined_ponding_depths'] = ponding_depths_int['ponding_depths_list'].apply(lambda x: list(np.concatenate(x)))
ponding_depths_int['percentile_1'] = ponding_depths_int['combined_ponding_depths'].apply(lambda x: safe_percentile(x, 1))
ponding_depths_int['percentile_25'] = ponding_depths_int['combined_ponding_depths'].apply(lambda x: safe_percentile(x, 25))
ponding_depths_int['percentile_50'] = ponding_depths_int['combined_ponding_depths'].apply(lambda x: safe_percentile(x, 50))
ponding_depths_int['percentile_75'] = ponding_depths_int['combined_ponding_depths'].apply(lambda x: safe_percentile(x, 75))
ponding_depths_int['percentile_99'] = ponding_depths_int['combined_ponding_depths'].apply(lambda x: safe_percentile(x, 99))
ponding_depths_int.drop(ponding_depths_int.tail(1).index, inplace=True)

## Compute event duration metrics
event_durations_int = data_int.groupby('year')['event_durations'].apply(list).reset_index()
event_durations_int.columns = ['year', 'event_durations_list']
event_durations_int['combined_event_durations'] = event_durations_int['event_durations_list'].apply(lambda x: list(np.concatenate(x)))
event_durations_int['percentile_1'] = event_durations_int['combined_event_durations'].apply(lambda x: safe_percentile(x, 1))
event_durations_int['percentile_25'] = event_durations_int['combined_event_durations'].apply(lambda x: safe_percentile(x, 25))
event_durations_int['percentile_50'] = event_durations_int['combined_event_durations'].apply(lambda x: safe_percentile(x, 50))
event_durations_int['percentile_75'] = event_durations_int['combined_event_durations'].apply(lambda x: safe_percentile(x, 75))
event_durations_int['percentile_99'] = event_durations_int['combined_event_durations'].apply(lambda x: safe_percentile(x, 99))
event_durations_int.drop(event_durations_int.tail(1).index, inplace=True)

## Compute number of days emergent metrics
days_emergent_int = data_int.groupby('year')['days_with_flooding'].apply(list).reset_index()
days_emergent_int['mean'] = days_emergent_int['days_with_flooding'].apply(np.mean)
days_emergent_int['std'] = days_emergent_int['days_with_flooding'].apply(np.std)
## If mean is < 1, set mean and std to 0
days_emergent_int.loc[days_emergent_int['mean'] < 1, 'mean'] = 0
days_emergent_int.loc[days_emergent_int['mean'] < 1, 'std'] = 0
## Cap std at the mean
days_emergent_int.loc[days_emergent_int['std'] > days_emergent_int['mean'], 'std'] = days_emergent_int['mean']
days_emergent_int.drop(days_emergent_int.tail(1).index, inplace=True)

## Compute total hours emergent metrics
hours_emergent_int = data_int.groupby('year')['hours_above_threshold'].apply(list).reset_index()
hours_emergent_int['mean'] = hours_emergent_int['hours_above_threshold'].apply(np.mean)
hours_emergent_int['std'] = hours_emergent_int['hours_above_threshold'].apply(np.std)
## If mean is < 1, set mean and std to 0
hours_emergent_int.loc[hours_emergent_int['mean'] < 1, 'mean'] = 0
hours_emergent_int.loc[hours_emergent_int['mean'] < 1, 'std'] = 0
## Cap std at the mean
hours_emergent_int.loc[hours_emergent_int['std'] > hours_emergent_int['mean'], 'std'] = hours_emergent_int['mean']
hours_emergent_int.drop(hours_emergent_int.tail(1).index, inplace=True)

combined_seasonal_results = pd.concat(seasonal_results).groupby('year').agg(
    mean_winter_flooding=('winter_flooding', 'mean'),
    std_winter_flooding=('winter_flooding', 'std'),
    mean_winter_day_flood=('winter_day_flood', 'mean'),
    std_winter_day_flood=('winter_day_flood', 'std'),
    mean_winter_night_flood=('winter_night_flood', 'mean'),
    std_winter_night_flood=('winter_night_flood', 'std'),
    mean_winter_both_flood=('winter_both_flood', 'mean'),
    std_winter_both_flood=('winter_both_flood', 'std'),
    mean_summer_flooding=('summer_flooding', 'mean'),
    std_summer_flooding=('summer_flooding', 'std'),
    mean_summer_day_flood=('summer_day_flood', 'mean'),
    std_summer_day_flood=('summer_day_flood', 'std'),
    mean_summer_night_flood=('summer_night_flood', 'mean'),
    std_summer_night_flood=('summer_night_flood', 'std'),
    mean_summer_both_flood=('summer_both_flood', 'mean'),
    std_summer_both_flood=('summer_both_flood', 'std')
).reset_index()

# Drop the last row
combined_seasonal_results.drop(combined_seasonal_results.tail(1).index, inplace=True)

# Ensure that all columns are numeric
combined_seasonal_results = combined_seasonal_results.apply(pd.to_numeric, errors='coerce')

## Define capped_std to ensure standard deviation bars do not extend below 0
def capped_seasonal_std(std_column, mean_column):
    # Calculate capped values where std deviation does not make the bar fall below zero
    capped_seasonal_values = np.where(
        (combined_seasonal_results[mean_column] - combined_seasonal_results[std_column]) < 0,
        combined_seasonal_results[mean_column],
        combined_seasonal_results[std_column]
    )
    return capped_seasonal_values

# Cap standard deviations where needed
for column in ['std_winter_flooding', 'std_winter_day_flood', 
               'std_winter_night_flood', 'std_winter_both_flood', 'std_summer_flooding', 
               'std_summer_day_flood', 'std_summer_night_flood', 'std_summer_both_flood']:
    combined_seasonal_results[column] = capped_seasonal_std(column, column.replace('std', 'mean'))

#%% Plot Flood Metrics including Seasonal Metrics S Seacoast - Intermediate SLR Scenario

boxplot_depth = [
    [
        ponding_depths_int['percentile_1'][i],
        ponding_depths_int['percentile_25'][i],
        ponding_depths_int['percentile_50'][i],
        ponding_depths_int['percentile_75'][i],
        ponding_depths_int['percentile_99'][i]
    ]
    for i in range(len(ponding_depths_int['year']))
]

boxplot_duration = [
    [
        event_durations_int['percentile_1'][i],
        event_durations_int['percentile_25'][i],
        event_durations_int['percentile_50'][i],
        event_durations_int['percentile_75'][i],
        event_durations_int['percentile_99'][i]
    ]
    for i in range(len(event_durations_int['year']))
]

# Adjust font size and figure size for clarity
fontsize = 14
fig, ax = plt.subplots(5, 1, figsize=(12, 14), sharex=True)

# ## OPTION 1: Plot Mean Flood Depth
# ax[0].bar(ponding_depths_int['year'], ponding_depths_int['mean_flood_depth'], yerr=ponding_depths_int['std_flood_depth'], color='blue', capsize=3)
# ax[0].set_ylabel('Mean Flood\nDepth (m)', fontsize=fontsize)

# ## OPTION 2: Box & Whisker plot for ponding_depths_int['flood_depth_min', 'flood_depth_25th', 'flood_depth_50th', 'flood_depth_75th', 'flood_depth_max']
# Create custom boxplots using the summary stats
for i, (min_val, q1, median, q3, max_val) in enumerate(boxplot_depth):
    ax[0].plot([ponding_depths_int['year'][i], ponding_depths_int['year'][i]], [min_val, max_val], color='black')  # Whiskers
    ax[0].plot([ponding_depths_int['year'][i], ponding_depths_int['year'][i]], [q1, q3], color='black', linewidth=7)  # Interquartile range (box)
    ax[0].plot([ponding_depths_int['year'][i], ponding_depths_int['year'][i]], [q1, q3], color='white', linewidth=5)  # Interquartile range (fill)
    ax[0].plot(ponding_depths_int['year'][i], median, 'k_', markersize=7)  # Median line
ax[0].set_yticks(np.arange(0, 0.71, 0.1))
ax[0].set_ylabel('Ponding\nDepth (m)', fontsize=fontsize)
## Plot horizontal line at 0.25m and put text on left side, top of line plot 'Safe Limit for Emergency Vehicles'
ax[0].axhline(y=0.25, color='r', linestyle='--')
ax[0].text(2020.5, 0.27, 'Safe Limit for\nEmergency Vehicles', color='k', fontsize=fontsize-2)

# ax[0].set_title('D0038\nIntermediate Sea Level Rise Scenario\nGroundwater Emergence Events', fontsize=fontsize+2)
ax[0].set_title('S Seacoast\nIntermediate Sea Level Rise Scenario\nGroundwater Emergence Events', fontsize=fontsize+2)

# ## OPTION 1: Plot the average hours per event
# [ax[1].bar(event_durations_int['year'], event_durations_int['mean_avg_hours_per_event'], yerr=event_durations_int['std_avg_hours_per_event'], color='blue', capsize=3)
# [ax[1].set_ylabel('Avg Flood\nDuration (hr)', fontsize=fontsize)

## OPTION 2: Box & Whisker plot for event_durations_int['duration_min', 'duration_25th', 'duration_50th', 'duration_75th', 'duration_max']
# Create custom boxplots using the summary stats
for i, (min_val, q1, median, q3, max_val) in enumerate(boxplot_duration):
    ax[1].plot([event_durations_int['year'][i], event_durations_int['year'][i]], [min_val, max_val], color='black')  # Whiskers
    ax[1].plot([event_durations_int['year'][i], event_durations_int['year'][i]], [q1, q3], color='black', linewidth=7)  # Interquartile range (box)
    ax[1].plot([event_durations_int['year'][i], event_durations_int['year'][i]], [q1, q3], color='white', linewidth=5)  # Interquartile range (fill)
    ax[1].plot(event_durations_int['year'][i], median, 'k_', markersize=7)  # Median line
ax[1].set_ylabel('Duration of\nEvents (hr)', fontsize=fontsize)
## OPTION 2b: boxplot using boxplot matplotlib function (need all data for boxplot to compute stats on)
# bp = ax[1].boxplot(boxplot_data, positions=event_durations_int['year'], widths=0.6)
# Customize the boxplot appearance
# for element in ['boxes', 'whiskers','means', 'medians', 'caps']:
#     plt.setp(bp[element], color='black')
# plt.setp(bp['fliers'], marker='o', markersize=3, markerfacecolor='red')
ax[1].set_yticks(np.arange(0, 26, 5))

# Plot the number of days with flooding
ax[2].bar(days_emergent_int['year'], days_emergent_int['mean'], yerr=days_emergent_int['std'], color='mediumseagreen', capsize=3)
ax[2].set_ylabel('# Days\nEmergent', fontsize=fontsize)
ax[2].text(2016.5, 354, 'Daily', color='k', fontsize=fontsize-2)
ax[2].set_ylim([0, 365])

# Plot the total hours above threshold
ax[3].bar(hours_emergent_int['year'], hours_emergent_int['mean'], yerr=hours_emergent_int['std'], color='mediumseagreen', capsize=3)
ax[3].set_ylabel('Total Time\nEmergent (hr)', fontsize=fontsize)
## Make y-axis plot ticks and labels every 1000
ax[3].set_yticks(np.arange(0, 6001, 1000))

# Stacked bar plot for seasonal day/night/both flooding
years = combined_seasonal_results['year']
summer_day = combined_seasonal_results['mean_summer_day_flood']
summer_night = combined_seasonal_results['mean_summer_night_flood']
summer_both = combined_seasonal_results['mean_summer_both_flood']
winter_day = combined_seasonal_results['mean_winter_day_flood']
winter_night = combined_seasonal_results['mean_winter_night_flood']
winter_both = combined_seasonal_results['mean_winter_both_flood']

# ## OPTION 1: Plot Day/Night/Both (units = # days)
# ax[4].bar(years, summer_day, label='Summer Day', color='lightblue')
# ax[4].bar(years, summer_night, bottom=summer_day, label='Summer Night', color='skyblue')
# ax[4].bar(years, summer_both, bottom=summer_day + summer_night, label='Summer Both', color='deepskyblue')
# ax[4].bar(years, winter_day, bottom=summer_day + summer_night + summer_both, label='Winter Day', color='lightcoral')
# ax[4].bar(years, winter_night, bottom=summer_day + summer_night + summer_both + winter_day, label='Winter Night', color='indianred')
# ax[4].bar(years, winter_both, bottom=summer_day + summer_night + summer_both + winter_day + winter_night, label='Winter Both', color='firebrick')

## OPTION 2: Plot Day/Night (units = # events)
# ax[4].bar(years, winter_night + winter_both, label='Winter Night', color='indigo')
# ax[4].bar(years, winter_day + winter_both, bottom=winter_night + winter_both, label='Winter Day', color='plum')
# ax[4].bar(years, summer_night + summer_both, bottom=winter_night + winter_day + 2*winter_both, label='Summer Night', color='royalblue')
# ax[4].bar(years, summer_day + summer_both, bottom=winter_night + winter_day + 2*winter_both + summer_night + summer_both, label='Summer Day', color='deepskyblue')
ax[4].bar(years, winter_night + winter_both, label='Winter Night', color='indigo')
ax[4].bar(years, winter_day + winter_both, bottom=winter_night + winter_both, label='Winter Day', color='mediumorchid')
ax[4].bar(years, summer_night + summer_both, bottom=winter_night + winter_day + 2*winter_both, label='Summer Night', color='mediumblue')
ax[4].bar(years, summer_day + summer_both, bottom=winter_night + winter_day + 2*winter_both + summer_night + summer_both, label='Summer Day', color='deepskyblue')
# Add error bars for standard deviation
# ax[4].errorbar(years, winter_night + winter_both, 
#                yerr=combined_seasonal_results['std_winter_night_flood']+combined_seasonal_results['std_winter_both_flood'], fmt='none', ecolor='plum', capsize=2)
# ax[4].errorbar(years, winter_night + winter_day + 2*winter_both, 
#                yerr=combined_seasonal_results['std_winter_day_flood']+combined_seasonal_results['std_winter_both_flood'], fmt='none', ecolor='plum', capsize=2)
# ax[4].errorbar(years, winter_night + winter_day + 2*winter_both + summer_night + summer_both, 
#                yerr=combined_seasonal_results['std_summer_night_flood']+combined_seasonal_results['std_summer_both_flood'], fmt='none', ecolor='lightcyan', capsize=2)
# ax[4].errorbar(years, winter_night + winter_day + 2*winter_both + summer_night + summer_day + 2*summer_both, 
#                yerr=combined_seasonal_results['std_summer_day_flood']+combined_seasonal_results['std_summer_both_flood'], fmt='none', ecolor='blue', capsize=2)

# Set labels and title for stacked bar plot
ax[4].set_ylabel('Emergence Events\nBy Season & TOD', fontsize=fontsize)
handles, labels = ax[4].get_legend_handles_labels()
ax[4].legend(handles[::-1], labels[::-1], loc='upper left', ncol=2, fontsize=fontsize-2)

# Set common x-axis label
ax[4].set_xlabel('Year', fontsize=fontsize)

# Set tick label font size
for a in ax:
    a.tick_params(axis='both', which='major', labelsize=fontsize-2)
    a.xaxis.grid(True, which='major')

## Restrict the x-axis
ax[0].set_xlim([2020, 2100])

plt.tight_layout()
plt.show()
#%% Flood metrics: S Seacoast - Low Scenario

# Initialize results storage
full_results_low = []

seasonal_results = []

## Intermediate Scenario
for realization in gwt_low.columns:
    gwt_low_flooding = pd.DataFrame(gwt_low[realization])
    gwt_low_flooding.columns = ['gwt']
    gwt_low_flooding['year'] = gwt_low_flooding.index.year
    gwt_low_flooding['above_threshold'] = gwt_low_flooding['gwt'] > roadelevation_sseacoast
    gwt_low_flooding['flood_depth'] = gwt_low_flooding['gwt'] - roadelevation_sseacoast
    gwt_low_flooding['flood_depth'] = gwt_low_flooding['flood_depth'].apply(lambda x: x if x > 0 else np.NaN)
    ## Drop nans
    gwt_low_flooding = gwt_low_flooding.dropna()

    realization_results = []
    realization_seasonal_results = []

    for year, group in gwt_low_flooding.groupby('year'):
        group['event_start'] = (group.index.to_series().diff() != pd.Timedelta('1H'))
        # group['event_start'] = (group.index.to_series().diff().shift(-1) == pd.Timedelta('1H'))
        num_events = group['event_start'].sum()

        ## Metric 3a: Number of days emergent
        days_with_flooding = group.groupby(group.index.date)['above_threshold'].any().sum()

        ## Metric 4a: Total time emergent
        hours_above_threshold = group['above_threshold'].sum()

        ## Metric 2a: Duration of Events and Ponding Depths
        event_durations = []
        ponding_depths = []

        event_start_indices = group.index[group['event_start']].tolist()
        event_start_indices.append(group.index[-1] + pd.Timedelta('1H'))  # Add an end marker for the last event

        for i in range(len(event_start_indices) - 1):
            event_period = group.loc[event_start_indices[i]:event_start_indices[i + 1] - pd.Timedelta('1H')]
            event_durations.append(len(event_period))
            ponding_depths.append(event_period['flood_depth'].max())

        ## Seasonal metrics initialization
        winter_flooding = 0
        winter_day_flood = 0
        winter_night_flood = 0
        winter_both_flood = 0
        summer_flooding = 0
        summer_day_flood = 0
        summer_night_flood = 0
        summer_both_flood = 0

        # Calculate winter flooding
        winter_period = group[(group.index.month >= 10) | (group.index.month <= 3)]
        if not winter_period.empty:
            winter_flood_days = winter_period.groupby(winter_period.index.date)['above_threshold'].any()
            winter_flooding = winter_flood_days.sum()

            # Determine day/night/both flooding for winter period
            for day, has_flooding in winter_flood_days.items():
                if has_flooding:
                    day_data = winter_period.loc[str(day)]
                    day_flood = day_data.loc[(day_data.index.hour >= 12) & (day_data.index.hour < 24), 'above_threshold'].any()
                    night_flood = day_data.loc[(day_data.index.hour >= 0) & (day_data.index.hour < 12), 'above_threshold'].any()
                    if day_flood and night_flood:
                        winter_both_flood += 1
                    elif day_flood:
                        winter_day_flood += 1
                    elif night_flood:
                        winter_night_flood += 1

        # Calculate summer flooding
        summer_period = group[(group.index.month >= 4) & (group.index.month <= 9)]
        if not summer_period.empty:
            summer_flood_days = summer_period.groupby(summer_period.index.date)['above_threshold'].any()
            summer_flooding = summer_flood_days.sum()

            # Determine day/night/both flooding for summer period
            for day, has_flooding in summer_flood_days.items():
                if has_flooding:
                    day_data = summer_period.loc[str(day)]
                    day_flood = day_data.loc[(day_data.index.hour >= 12) & (day_data.index.hour < 24), 'above_threshold'].any()
                    night_flood = day_data.loc[(day_data.index.hour >= 0) & (day_data.index.hour < 12), 'above_threshold'].any()
                    if day_flood and night_flood:
                        summer_both_flood += 1
                    elif day_flood:
                        summer_day_flood += 1
                    elif night_flood:
                        summer_night_flood += 1

        ## Save Realization Results
        realization_results.append({
            'year': year,
            'ponding_depths': ponding_depths,
            'days_with_flooding': days_with_flooding,
            'hours_above_threshold': hours_above_threshold,
            'num_events': num_events,
            'event_durations': event_durations,
        })

        realization_seasonal_results.append({
            'year': year,
            'winter_flooding': winter_flooding,
            'winter_day_flood': winter_day_flood,
            'winter_night_flood': winter_night_flood,
            'winter_both_flood': winter_both_flood,
            'summer_flooding': summer_flooding,
            'summer_day_flood': summer_day_flood,
            'summer_night_flood': summer_night_flood,
            'summer_both_flood': summer_both_flood
        })

    full_results_low.append(pd.DataFrame(realization_results))
    seasonal_results.append(pd.DataFrame(realization_seasonal_results))

## Compute metrics from the full results by year S Seacoast - Intermediate Scenario
## Compute the minimum, maximum, 25th, 50th, and 75th percentiles

# Ensure all final results are numeric
for df in full_results_low:
    df.apply(pd.to_numeric, errors='coerce')

def safe_percentile(x, q):
    if len(x) > 0:
        return np.percentile(x, q) 
    else:
        return np.nan

## Concatenate all realizations into a single DataFrame
data_low = pd.concat(full_results_low)

## Compute ponding depth metrics
ponding_depths_low = data_low.groupby('year')['ponding_depths'].apply(list).reset_index()
ponding_depths_low.columns = ['year', 'ponding_depths_list']
ponding_depths_low['combined_ponding_depths'] = ponding_depths_low['ponding_depths_list'].apply(lambda x: list(np.concatenate(x)))
ponding_depths_low['percentile_1'] = ponding_depths_low['combined_ponding_depths'].apply(lambda x: safe_percentile(x, 1))
ponding_depths_low['percentile_25'] = ponding_depths_low['combined_ponding_depths'].apply(lambda x: safe_percentile(x, 25))
ponding_depths_low['percentile_50'] = ponding_depths_low['combined_ponding_depths'].apply(lambda x: safe_percentile(x, 50))
ponding_depths_low['percentile_75'] = ponding_depths_low['combined_ponding_depths'].apply(lambda x: safe_percentile(x, 75))
ponding_depths_low['percentile_99'] = ponding_depths_low['combined_ponding_depths'].apply(lambda x: safe_percentile(x, 99))
ponding_depths_low.drop(ponding_depths_low.tail(1).index, inplace=True)

## Compute event duration metrics
event_durations_low = data_low.groupby('year')['event_durations'].apply(list).reset_index()
event_durations_low.columns = ['year', 'event_durations_list']
event_durations_low['combined_event_durations'] = event_durations_low['event_durations_list'].apply(lambda x: list(np.concatenate(x)))
event_durations_low['percentile_1'] = event_durations_low['combined_event_durations'].apply(lambda x: safe_percentile(x, 1))
event_durations_low['percentile_25'] = event_durations_low['combined_event_durations'].apply(lambda x: safe_percentile(x, 25))
event_durations_low['percentile_50'] = event_durations_low['combined_event_durations'].apply(lambda x: safe_percentile(x, 50))
event_durations_low['percentile_75'] = event_durations_low['combined_event_durations'].apply(lambda x: safe_percentile(x, 75))
event_durations_low['percentile_99'] = event_durations_low['combined_event_durations'].apply(lambda x: safe_percentile(x, 99))
event_durations_low.drop(event_durations_low.tail(1).index, inplace=True)

## Compute number of days emergent metrics
days_emergent_low = data_low.groupby('year')['days_with_flooding'].apply(list).reset_index()
days_emergent_low['mean'] = days_emergent_low['days_with_flooding'].apply(np.mean)
days_emergent_low['std'] = days_emergent_low['days_with_flooding'].apply(np.std)
## If mean is < 1, set mean and std to 0
days_emergent_low.loc[days_emergent_low['mean'] < 1, 'mean'] = 0
days_emergent_low.loc[days_emergent_low['mean'] < 1, 'std'] = 0
## Cap std at the mean
days_emergent_low.loc[days_emergent_low['std'] > days_emergent_low['mean'], 'std'] = days_emergent_low['mean']
days_emergent_low.drop(days_emergent_low.tail(1).index, inplace=True)

## Compute total hours emergent metrics
hours_emergent_low = data_low.groupby('year')['hours_above_threshold'].apply(list).reset_index()
hours_emergent_low['mean'] = hours_emergent_low['hours_above_threshold'].apply(np.mean)
hours_emergent_low['std'] = hours_emergent_low['hours_above_threshold'].apply(np.std)
## If mean is < 1, set mean and std to 0
hours_emergent_low.loc[hours_emergent_low['mean'] < 1, 'mean'] = 0
hours_emergent_low.loc[hours_emergent_low['mean'] < 1, 'std'] = 0
## Cap std at the mean
hours_emergent_low.loc[hours_emergent_low['std'] > hours_emergent_low['mean'], 'std'] = hours_emergent_low['mean']
hours_emergent_low.drop(hours_emergent_low.tail(1).index, inplace=True)

combined_seasonal_results = pd.concat(seasonal_results).groupby('year').agg(
    mean_winter_flooding=('winter_flooding', 'mean'),
    std_winter_flooding=('winter_flooding', 'std'),
    mean_winter_day_flood=('winter_day_flood', 'mean'),
    std_winter_day_flood=('winter_day_flood', 'std'),
    mean_winter_night_flood=('winter_night_flood', 'mean'),
    std_winter_night_flood=('winter_night_flood', 'std'),
    mean_winter_both_flood=('winter_both_flood', 'mean'),
    std_winter_both_flood=('winter_both_flood', 'std'),
    mean_summer_flooding=('summer_flooding', 'mean'),
    std_summer_flooding=('summer_flooding', 'std'),
    mean_summer_day_flood=('summer_day_flood', 'mean'),
    std_summer_day_flood=('summer_day_flood', 'std'),
    mean_summer_night_flood=('summer_night_flood', 'mean'),
    std_summer_night_flood=('summer_night_flood', 'std'),
    mean_summer_both_flood=('summer_both_flood', 'mean'),
    std_summer_both_flood=('summer_both_flood', 'std')
).reset_index()

# Drop the last row
combined_seasonal_results.drop(combined_seasonal_results.tail(1).index, inplace=True)

# Ensure that all columns are numeric
combined_seasonal_results = combined_seasonal_results.apply(pd.to_numeric, errors='coerce')

## Define capped_std to ensure standard deviation bars do not extend below 0
def capped_seasonal_std(std_column, mean_column):
    # Calculate capped values where std deviation does not make the bar fall below zero
    capped_seasonal_values = np.where(
        (combined_seasonal_results[mean_column] - combined_seasonal_results[std_column]) < 0,
        combined_seasonal_results[mean_column],
        combined_seasonal_results[std_column]
    )
    return capped_seasonal_values

# Cap standard deviations where needed
for column in ['std_winter_flooding', 'std_winter_day_flood', 
               'std_winter_night_flood', 'std_winter_both_flood', 'std_summer_flooding', 
               'std_summer_day_flood', 'std_summer_night_flood', 'std_summer_both_flood']:
    combined_seasonal_results[column] = capped_seasonal_std(column, column.replace('std', 'mean'))

## Plot Flood Metrics including Seasonal Metrics S Seacoast - Intermediate SLR Scenario

boxplot_depth = [
    [
        ponding_depths_low['percentile_1'][i],
        ponding_depths_low['percentile_25'][i],
        ponding_depths_low['percentile_50'][i],
        ponding_depths_low['percentile_75'][i],
        ponding_depths_low['percentile_99'][i]
    ]
    for i in range(len(ponding_depths_low['year']))
]

boxplot_duration = [
    [
        event_durations_low['percentile_1'][i],
        event_durations_low['percentile_25'][i],
        event_durations_low['percentile_50'][i],
        event_durations_low['percentile_75'][i],
        event_durations_low['percentile_99'][i]
    ]
    for i in range(len(event_durations_low['year']))
]

## PLOT
fontsize = 14
fig, ax = plt.subplots(5, 1, figsize=(12, 14), sharex=True)

for i, (min_val, q1, median, q3, max_val) in enumerate(boxplot_depth):
    ax[0].plot([ponding_depths_low['year'][i], ponding_depths_low['year'][i]], [min_val, max_val], color='black')  # Whiskers
    ax[0].plot([ponding_depths_low['year'][i], ponding_depths_low['year'][i]], [q1, q3], color='black', linewidth=7)  # Interquartile range (box)
    ax[0].plot([ponding_depths_low['year'][i], ponding_depths_low['year'][i]], [q1, q3], color='white', linewidth=5)  # Interquartile range (fill)
    ax[0].plot(ponding_depths_low['year'][i], median, 'k_', markersize=7)  # Median line
ax[0].set_yticks(np.arange(0, 0.71, 0.1))
ax[0].set_ylabel('Ponding\nDepth (m)', fontsize=fontsize)
## Plot horizontal line at 0.25m and put text on left side, top of line plot 'Safe Limit for Emergency Vehicles'
ax[0].axhline(y=0.25, color='r', linestyle='--')
ax[0].text(2020.5, 0.27, 'Safe Limit for\nEmergency Vehicles', color='k', fontsize=fontsize-2)

ax[0].set_title('S Seacoast\nLow Sea Level Rise Scenario\nGroundwater Emergence Events', fontsize=fontsize+2)

for i, (min_val, q1, median, q3, max_val) in enumerate(boxplot_duration):
    ax[1].plot([event_durations_low['year'][i], event_durations_low['year'][i]], [min_val, max_val], color='black')  # Whiskers
    ax[1].plot([event_durations_low['year'][i], event_durations_low['year'][i]], [q1, q3], color='black', linewidth=7)  # Interquartile range (box)
    ax[1].plot([event_durations_low['year'][i], event_durations_low['year'][i]], [q1, q3], color='white', linewidth=5)  # Interquartile range (fill)
    ax[1].plot(event_durations_low['year'][i], median, 'k_', markersize=7)  # Median line
ax[1].set_ylabel('Duration of\nEvents (hr)', fontsize=fontsize)
ax[1].set_yticks(np.arange(0, 16, 5))

# Plot the number of days with flooding
ax[2].bar(days_emergent_low['year'], days_emergent_low['mean'], yerr=days_emergent_low['std'], color='mediumseagreen', capsize=3)
ax[2].set_ylabel('# Days\nEmergent', fontsize=fontsize)
ax[2].text(2016.5, 354, 'Daily', color='k', fontsize=fontsize-2)
ax[2].set_yticks(np.arange(0, 16, 5))

# Plot the total hours above threshold
ax[3].bar(hours_emergent_low['year'], hours_emergent_low['mean'], yerr=hours_emergent_low['std'], color='mediumseagreen', capsize=3)
ax[3].set_ylabel('Total Time\nEmergent (hr)', fontsize=fontsize)
ax[3].set_yticks(np.arange(0, 51, 10))

# Stacked bar plot for seasonal day/night/both flooding
years = combined_seasonal_results['year']
summer_day = combined_seasonal_results['mean_summer_day_flood']
summer_night = combined_seasonal_results['mean_summer_night_flood']
summer_both = combined_seasonal_results['mean_summer_both_flood']
winter_day = combined_seasonal_results['mean_winter_day_flood']
winter_night = combined_seasonal_results['mean_winter_night_flood']
winter_both = combined_seasonal_results['mean_winter_both_flood']

ax[4].bar(years, winter_night + winter_both, label='Winter Night', color='indigo')
ax[4].bar(years, winter_day + winter_both, bottom=winter_night + winter_both, label='Winter Day', color='mediumorchid')
ax[4].bar(years, summer_night + summer_both, bottom=winter_night + winter_day + 2*winter_both, label='Summer Night', color='mediumblue')
ax[4].bar(years, summer_day + summer_both, bottom=winter_night + winter_day + 2*winter_both + summer_night + summer_both, label='Summer Day', color='deepskyblue')

ax[4].set_ylim([0, 10])

# Set labels and title for stacked bar plot
ax[4].set_ylabel('Emergence Events\nBy Season & TOD', fontsize=fontsize)
handles, labels = ax[4].get_legend_handles_labels()
ax[4].legend(handles[::-1], labels[::-1], loc='upper left', ncol = 2,fontsize=fontsize-2)

# Set common x-axis label
ax[4].set_xlabel('Year', fontsize=fontsize)

# Set tick label font size
for a in ax:
    a.tick_params(axis='both', which='major', labelsize=fontsize-2)
    a.xaxis.grid(True, which='major')

## Restrict the x-axis
ax[0].set_xlim([2020, 2100])

plt.tight_layout()
plt.show()
#%% Flood metrics: S Seacoast - Int Low Scenario

# Initialize results storage
full_results_intlow = []

seasonal_results = []

## Intermediate Scenario
for realization in gwt_intlow.columns:
    gwt_intlow_flooding = pd.DataFrame(gwt_intlow[realization])
    gwt_intlow_flooding.columns = ['gwt']
    gwt_intlow_flooding['year'] = gwt_intlow_flooding.index.year
    gwt_intlow_flooding['above_threshold'] = gwt_intlow_flooding['gwt'] > roadelevation_sseacoast
    gwt_intlow_flooding['flood_depth'] = gwt_intlow_flooding['gwt'] - roadelevation_sseacoast
    gwt_intlow_flooding['flood_depth'] = gwt_intlow_flooding['flood_depth'].apply(lambda x: x if x > 0 else np.NaN)
    ## Drop nans
    gwt_intlow_flooding = gwt_intlow_flooding.dropna()

    realization_results = []
    realization_seasonal_results = []

    for year, group in gwt_intlow_flooding.groupby('year'):
        group['event_start'] = (group.index.to_series().diff() != pd.Timedelta('1H'))
        # group['event_start'] = (group.index.to_series().diff().shift(-1) == pd.Timedelta('1H'))
        num_events = group['event_start'].sum()

        ## Metric 3a: Number of days emergent
        days_with_flooding = group.groupby(group.index.date)['above_threshold'].any().sum()

        ## Metric 4a: Total time emergent
        hours_above_threshold = group['above_threshold'].sum()

        ## Metric 2a: Duration of Events and Ponding Depths
        event_durations = []
        ponding_depths = []

        event_start_indices = group.index[group['event_start']].tolist()
        event_start_indices.append(group.index[-1] + pd.Timedelta('1H'))  # Add an end marker for the last event

        for i in range(len(event_start_indices) - 1):
            event_period = group.loc[event_start_indices[i]:event_start_indices[i + 1] - pd.Timedelta('1H')]
            event_durations.append(len(event_period))
            ponding_depths.append(event_period['flood_depth'].max())

        ## Seasonal metrics initialization
        winter_flooding = 0
        winter_day_flood = 0
        winter_night_flood = 0
        winter_both_flood = 0
        summer_flooding = 0
        summer_day_flood = 0
        summer_night_flood = 0
        summer_both_flood = 0

        # Calculate winter flooding
        winter_period = group[(group.index.month >= 10) | (group.index.month <= 3)]
        if not winter_period.empty:
            winter_flood_days = winter_period.groupby(winter_period.index.date)['above_threshold'].any()
            winter_flooding = winter_flood_days.sum()

            # Determine day/night/both flooding for winter period
            for day, has_flooding in winter_flood_days.items():
                if has_flooding:
                    day_data = winter_period.loc[str(day)]
                    day_flood = day_data.loc[(day_data.index.hour >= 12) & (day_data.index.hour < 24), 'above_threshold'].any()
                    night_flood = day_data.loc[(day_data.index.hour >= 0) & (day_data.index.hour < 12), 'above_threshold'].any()
                    if day_flood and night_flood:
                        winter_both_flood += 1
                    elif day_flood:
                        winter_day_flood += 1
                    elif night_flood:
                        winter_night_flood += 1

        # Calculate summer flooding
        summer_period = group[(group.index.month >= 4) & (group.index.month <= 9)]
        if not summer_period.empty:
            summer_flood_days = summer_period.groupby(summer_period.index.date)['above_threshold'].any()
            summer_flooding = summer_flood_days.sum()

            # Determine day/night/both flooding for summer period
            for day, has_flooding in summer_flood_days.items():
                if has_flooding:
                    day_data = summer_period.loc[str(day)]
                    day_flood = day_data.loc[(day_data.index.hour >= 12) & (day_data.index.hour < 24), 'above_threshold'].any()
                    night_flood = day_data.loc[(day_data.index.hour >= 0) & (day_data.index.hour < 12), 'above_threshold'].any()
                    if day_flood and night_flood:
                        summer_both_flood += 1
                    elif day_flood:
                        summer_day_flood += 1
                    elif night_flood:
                        summer_night_flood += 1

        ## Save Realization Results
        realization_results.append({
            'year': year,
            'ponding_depths': ponding_depths,
            'days_with_flooding': days_with_flooding,
            'hours_above_threshold': hours_above_threshold,
            'num_events': num_events,
            'event_durations': event_durations,
        })

        realization_seasonal_results.append({
            'year': year,
            'winter_flooding': winter_flooding,
            'winter_day_flood': winter_day_flood,
            'winter_night_flood': winter_night_flood,
            'winter_both_flood': winter_both_flood,
            'summer_flooding': summer_flooding,
            'summer_day_flood': summer_day_flood,
            'summer_night_flood': summer_night_flood,
            'summer_both_flood': summer_both_flood
        })

    full_results_intlow.append(pd.DataFrame(realization_results))
    seasonal_results.append(pd.DataFrame(realization_seasonal_results))

## Compute metrics from the full results by year S Seacoast - Intermediate Scenario
## Compute the minimum, maximum, 25th, 50th, and 75th percentiles

# Ensure all final results are numeric
for df in full_results_intlow:
    df.apply(pd.to_numeric, errors='coerce')

def safe_percentile(x, q):
    if len(x) > 0:
        return np.percentile(x, q) 
    else:
        return np.nan

## Concatenate all realizations into a single DataFrame
data_intlow = pd.concat(full_results_intlow)

## Compute ponding depth metrics
ponding_depths_intlow = data_intlow.groupby('year')['ponding_depths'].apply(list).reset_index()
ponding_depths_intlow.columns = ['year', 'ponding_depths_list']
ponding_depths_intlow['combined_ponding_depths'] = ponding_depths_intlow['ponding_depths_list'].apply(lambda x: list(np.concatenate(x)))
ponding_depths_intlow['percentile_1'] = ponding_depths_intlow['combined_ponding_depths'].apply(lambda x: safe_percentile(x, 1))
ponding_depths_intlow['percentile_25'] = ponding_depths_intlow['combined_ponding_depths'].apply(lambda x: safe_percentile(x, 25))
ponding_depths_intlow['percentile_50'] = ponding_depths_intlow['combined_ponding_depths'].apply(lambda x: safe_percentile(x, 50))
ponding_depths_intlow['percentile_75'] = ponding_depths_intlow['combined_ponding_depths'].apply(lambda x: safe_percentile(x, 75))
ponding_depths_intlow['percentile_99'] = ponding_depths_intlow['combined_ponding_depths'].apply(lambda x: safe_percentile(x, 99))
ponding_depths_intlow.drop(ponding_depths_intlow.tail(1).index, inplace=True)

## Compute event duration metrics
event_durations_intlow = data_intlow.groupby('year')['event_durations'].apply(list).reset_index()
event_durations_intlow.columns = ['year', 'event_durations_list']
event_durations_intlow['combined_event_durations'] = event_durations_intlow['event_durations_list'].apply(lambda x: list(np.concatenate(x)))
event_durations_intlow['percentile_1'] = event_durations_intlow['combined_event_durations'].apply(lambda x: safe_percentile(x, 1))
event_durations_intlow['percentile_25'] = event_durations_intlow['combined_event_durations'].apply(lambda x: safe_percentile(x, 25))
event_durations_intlow['percentile_50'] = event_durations_intlow['combined_event_durations'].apply(lambda x: safe_percentile(x, 50))
event_durations_intlow['percentile_75'] = event_durations_intlow['combined_event_durations'].apply(lambda x: safe_percentile(x, 75))
event_durations_intlow['percentile_99'] = event_durations_intlow['combined_event_durations'].apply(lambda x: safe_percentile(x, 99))
event_durations_intlow.drop(event_durations_intlow.tail(1).index, inplace=True)

## Compute number of days emergent metrics
days_emergent_intlow = data_intlow.groupby('year')['days_with_flooding'].apply(list).reset_index()
days_emergent_intlow['mean'] = days_emergent_intlow['days_with_flooding'].apply(np.mean)
days_emergent_intlow['std'] = days_emergent_intlow['days_with_flooding'].apply(np.std)
## If mean is < 1, set mean and std to 0
days_emergent_intlow.loc[days_emergent_intlow['mean'] < 1, 'mean'] = 0
days_emergent_intlow.loc[days_emergent_intlow['mean'] < 1, 'std'] = 0
## Cap std at the mean
days_emergent_intlow.loc[days_emergent_intlow['std'] > days_emergent_intlow['mean'], 'std'] = days_emergent_intlow['mean']
days_emergent_intlow.drop(days_emergent_intlow.tail(1).index, inplace=True)

## Compute total hours emergent metrics
hours_emergent_intlow = data_intlow.groupby('year')['hours_above_threshold'].apply(list).reset_index()
hours_emergent_intlow['mean'] = hours_emergent_intlow['hours_above_threshold'].apply(np.mean)
hours_emergent_intlow['std'] = hours_emergent_intlow['hours_above_threshold'].apply(np.std)
## If mean is < 1, set mean and std to 0
hours_emergent_intlow.loc[hours_emergent_intlow['mean'] < 1, 'mean'] = 0
hours_emergent_intlow.loc[hours_emergent_intlow['mean'] < 1, 'std'] = 0
## Cap std at the mean
hours_emergent_intlow.loc[hours_emergent_intlow['std'] > hours_emergent_intlow['mean'], 'std'] = hours_emergent_intlow['mean']
hours_emergent_intlow.drop(hours_emergent_intlow.tail(1).index, inplace=True)

combined_seasonal_results = pd.concat(seasonal_results).groupby('year').agg(
    mean_winter_flooding=('winter_flooding', 'mean'),
    std_winter_flooding=('winter_flooding', 'std'),
    mean_winter_day_flood=('winter_day_flood', 'mean'),
    std_winter_day_flood=('winter_day_flood', 'std'),
    mean_winter_night_flood=('winter_night_flood', 'mean'),
    std_winter_night_flood=('winter_night_flood', 'std'),
    mean_winter_both_flood=('winter_both_flood', 'mean'),
    std_winter_both_flood=('winter_both_flood', 'std'),
    mean_summer_flooding=('summer_flooding', 'mean'),
    std_summer_flooding=('summer_flooding', 'std'),
    mean_summer_day_flood=('summer_day_flood', 'mean'),
    std_summer_day_flood=('summer_day_flood', 'std'),
    mean_summer_night_flood=('summer_night_flood', 'mean'),
    std_summer_night_flood=('summer_night_flood', 'std'),
    mean_summer_both_flood=('summer_both_flood', 'mean'),
    std_summer_both_flood=('summer_both_flood', 'std')
).reset_index()

# Drop the last row
combined_seasonal_results.drop(combined_seasonal_results.tail(1).index, inplace=True)

# Ensure that all columns are numeric
combined_seasonal_results = combined_seasonal_results.apply(pd.to_numeric, errors='coerce')

## Define capped_std to ensure standard deviation bars do not extend below 0
def capped_seasonal_std(std_column, mean_column):
    # Calculate capped values where std deviation does not make the bar fall below zero
    capped_seasonal_values = np.where(
        (combined_seasonal_results[mean_column] - combined_seasonal_results[std_column]) < 0,
        combined_seasonal_results[mean_column],
        combined_seasonal_results[std_column]
    )
    return capped_seasonal_values

# Cap standard deviations where needed
for column in ['std_winter_flooding', 'std_winter_day_flood', 
               'std_winter_night_flood', 'std_winter_both_flood', 'std_summer_flooding', 
               'std_summer_day_flood', 'std_summer_night_flood', 'std_summer_both_flood']:
    combined_seasonal_results[column] = capped_seasonal_std(column, column.replace('std', 'mean'))

## Plot Flood Metrics including Seasonal Metrics S Seacoast - Intermediate SLR Scenario

boxplot_depth = [
    [
        ponding_depths_intlow['percentile_1'][i],
        ponding_depths_intlow['percentile_25'][i],
        ponding_depths_intlow['percentile_50'][i],
        ponding_depths_intlow['percentile_75'][i],
        ponding_depths_intlow['percentile_99'][i]
    ]
    for i in range(len(ponding_depths_intlow['year']))
]

boxplot_duration = [
    [
        event_durations_intlow['percentile_1'][i],
        event_durations_intlow['percentile_25'][i],
        event_durations_intlow['percentile_50'][i],
        event_durations_intlow['percentile_75'][i],
        event_durations_intlow['percentile_99'][i]
    ]
    for i in range(len(event_durations_intlow['year']))
]

## PLOT
fontsize = 14
fig, ax = plt.subplots(5, 1, figsize=(12, 14), sharex=True)

for i, (min_val, q1, median, q3, max_val) in enumerate(boxplot_depth):
    ax[0].plot([ponding_depths_intlow['year'][i], ponding_depths_intlow['year'][i]], [min_val, max_val], color='black')  # Whiskers
    ax[0].plot([ponding_depths_intlow['year'][i], ponding_depths_intlow['year'][i]], [q1, q3], color='black', linewidth=7)  # Interquartile range (box)
    ax[0].plot([ponding_depths_intlow['year'][i], ponding_depths_intlow['year'][i]], [q1, q3], color='white', linewidth=5)  # Interquartile range (fill)
    ax[0].plot(ponding_depths_intlow['year'][i], median, 'k_', markersize=7)  # Median line
ax[0].set_yticks(np.arange(0, 0.51, 0.1))
ax[0].set_ylabel('Ponding\nDepth (m)', fontsize=fontsize)
## Plot horizontal line at 0.25m and put text on left side, top of line plot 'Safe Limit for Emergency Vehicles'
ax[0].axhline(y=0.25, color='r', linestyle='--')
ax[0].text(2020.5, 0.27, 'Safe Limit for\nEmergency Vehicles', color='k', fontsize=fontsize-2)

ax[0].set_title('S Seacoast\nIntermediate Low Sea Level Rise Scenario\nGroundwater Emergence Events', fontsize=fontsize+2)

for i, (min_val, q1, median, q3, max_val) in enumerate(boxplot_duration):
    ax[1].plot([event_durations_intlow['year'][i], event_durations_intlow['year'][i]], [min_val, max_val], color='black')  # Whiskers
    ax[1].plot([event_durations_intlow['year'][i], event_durations_intlow['year'][i]], [q1, q3], color='black', linewidth=7)  # Interquartile range (box)
    ax[1].plot([event_durations_intlow['year'][i], event_durations_intlow['year'][i]], [q1, q3], color='white', linewidth=5)  # Interquartile range (fill)
    ax[1].plot(event_durations_intlow['year'][i], median, 'k_', markersize=7)  # Median line
ax[1].set_ylabel('Duration of\nEvents (hr)', fontsize=fontsize)
ax[1].set_yticks(np.arange(0, 11, 2))

# Plot the number of days with flooding
ax[2].bar(days_emergent_intlow['year'], days_emergent_intlow['mean'], yerr=days_emergent_intlow['std'], color='mediumseagreen', capsize=3)
ax[2].set_ylabel('# Days\nEmergent', fontsize=fontsize)
# ax[2].text(2016.5, 354, 'Daily', color='k', fontsize=fontsize-2)
ax[2].set_yticks(np.arange(0, 101, 20))

# Plot the total hours above threshold
ax[3].bar(hours_emergent_intlow['year'], hours_emergent_intlow['mean'], yerr=hours_emergent_intlow['std'], color='mediumseagreen', capsize=3)
ax[3].set_ylabel('Total Time\nEmergent (hr)', fontsize=fontsize)
ax[3].set_yticks(np.arange(0, 251, 50))

# Stacked bar plot for seasonal day/night/both flooding
years = combined_seasonal_results['year']
summer_day = combined_seasonal_results['mean_summer_day_flood']
summer_night = combined_seasonal_results['mean_summer_night_flood']
summer_both = combined_seasonal_results['mean_summer_both_flood']
winter_day = combined_seasonal_results['mean_winter_day_flood']
winter_night = combined_seasonal_results['mean_winter_night_flood']
winter_both = combined_seasonal_results['mean_winter_both_flood']

ax[4].bar(years, winter_night + winter_both, label='Winter Night', color='indigo')
ax[4].bar(years, winter_day + winter_both, bottom=winter_night + winter_both, label='Winter Day', color='mediumorchid')
ax[4].bar(years, summer_night + summer_both, bottom=winter_night + winter_day + 2*winter_both, label='Summer Night', color='mediumblue')
ax[4].bar(years, summer_day + summer_both, bottom=winter_night + winter_day + 2*winter_both + summer_night + summer_both, label='Summer Day', color='deepskyblue')

# ax[4].set_ylim([0, 15])

# Set labels and title for stacked bar plot
ax[4].set_ylabel('Emergence Events\nBy Season & TOD', fontsize=fontsize)
handles, labels = ax[4].get_legend_handles_labels()
ax[4].legend(handles[::-1], labels[::-1], loc='upper left', ncol = 2,fontsize=fontsize-2)

# Set common x-axis label
ax[4].set_xlabel('Year', fontsize=fontsize)

# Set tick label font size
for a in ax:
    a.tick_params(axis='both', which='major', labelsize=fontsize-2)
    a.xaxis.grid(True, which='major')

## Restrict the x-axis
ax[0].set_xlim([2020, 2100])

plt.tight_layout()
plt.show()
#%% Flood metrics: S Seacoast - Int High Scenario

# Initialize results storage
full_results_inthigh = []

seasonal_results = []

## Intermediate Scenario
for realization in gwt_inthigh.columns:
    gwt_inthigh_flooding = pd.DataFrame(gwt_inthigh[realization])
    gwt_inthigh_flooding.columns = ['gwt']
    gwt_inthigh_flooding['year'] = gwt_inthigh_flooding.index.year
    gwt_inthigh_flooding['above_threshold'] = gwt_inthigh_flooding['gwt'] > roadelevation_sseacoast
    gwt_inthigh_flooding['flood_depth'] = gwt_inthigh_flooding['gwt'] - roadelevation_sseacoast
    gwt_inthigh_flooding['flood_depth'] = gwt_inthigh_flooding['flood_depth'].apply(lambda x: x if x > 0 else np.NaN)
    ## Drop nans
    gwt_inthigh_flooding = gwt_inthigh_flooding.dropna()

    realization_results = []
    realization_seasonal_results = []

    for year, group in gwt_inthigh_flooding.groupby('year'):
        group['event_start'] = (group.index.to_series().diff() != pd.Timedelta('1H'))
        # group['event_start'] = (group.index.to_series().diff().shift(-1) == pd.Timedelta('1H'))
        num_events = group['event_start'].sum()

        ## Metric 3a: Number of days emergent
        days_with_flooding = group.groupby(group.index.date)['above_threshold'].any().sum()

        ## Metric 4a: Total time emergent
        hours_above_threshold = group['above_threshold'].sum()

        ## Metric 2a: Duration of Events and Ponding Depths
        event_durations = []
        ponding_depths = []

        event_start_indices = group.index[group['event_start']].tolist()
        event_start_indices.append(group.index[-1] + pd.Timedelta('1H'))  # Add an end marker for the last event

        for i in range(len(event_start_indices) - 1):
            event_period = group.loc[event_start_indices[i]:event_start_indices[i + 1] - pd.Timedelta('1H')]
            event_durations.append(len(event_period))
            ponding_depths.append(event_period['flood_depth'].max())

        ## Seasonal metrics initialization
        winter_flooding = 0
        winter_day_flood = 0
        winter_night_flood = 0
        winter_both_flood = 0
        summer_flooding = 0
        summer_day_flood = 0
        summer_night_flood = 0
        summer_both_flood = 0

        # Calculate winter flooding
        winter_period = group[(group.index.month >= 10) | (group.index.month <= 3)]
        if not winter_period.empty:
            winter_flood_days = winter_period.groupby(winter_period.index.date)['above_threshold'].any()
            winter_flooding = winter_flood_days.sum()

            # Determine day/night/both flooding for winter period
            for day, has_flooding in winter_flood_days.items():
                if has_flooding:
                    day_data = winter_period.loc[str(day)]
                    day_flood = day_data.loc[(day_data.index.hour >= 12) & (day_data.index.hour < 24), 'above_threshold'].any()
                    night_flood = day_data.loc[(day_data.index.hour >= 0) & (day_data.index.hour < 12), 'above_threshold'].any()
                    if day_flood and night_flood:
                        winter_both_flood += 1
                    elif day_flood:
                        winter_day_flood += 1
                    elif night_flood:
                        winter_night_flood += 1

        # Calculate summer flooding
        summer_period = group[(group.index.month >= 4) & (group.index.month <= 9)]
        if not summer_period.empty:
            summer_flood_days = summer_period.groupby(summer_period.index.date)['above_threshold'].any()
            summer_flooding = summer_flood_days.sum()

            # Determine day/night/both flooding for summer period
            for day, has_flooding in summer_flood_days.items():
                if has_flooding:
                    day_data = summer_period.loc[str(day)]
                    day_flood = day_data.loc[(day_data.index.hour >= 12) & (day_data.index.hour < 24), 'above_threshold'].any()
                    night_flood = day_data.loc[(day_data.index.hour >= 0) & (day_data.index.hour < 12), 'above_threshold'].any()
                    if day_flood and night_flood:
                        summer_both_flood += 1
                    elif day_flood:
                        summer_day_flood += 1
                    elif night_flood:
                        summer_night_flood += 1

        ## Save Realization Results
        realization_results.append({
            'year': year,
            'ponding_depths': ponding_depths,
            'days_with_flooding': days_with_flooding,
            'hours_above_threshold': hours_above_threshold,
            'num_events': num_events,
            'event_durations': event_durations,
        })

        realization_seasonal_results.append({
            'year': year,
            'winter_flooding': winter_flooding,
            'winter_day_flood': winter_day_flood,
            'winter_night_flood': winter_night_flood,
            'winter_both_flood': winter_both_flood,
            'summer_flooding': summer_flooding,
            'summer_day_flood': summer_day_flood,
            'summer_night_flood': summer_night_flood,
            'summer_both_flood': summer_both_flood
        })

    full_results_inthigh.append(pd.DataFrame(realization_results))
    seasonal_results.append(pd.DataFrame(realization_seasonal_results))

## Compute metrics from the full results by year S Seacoast - Intermediate Scenario
## Compute the minimum, maximum, 25th, 50th, and 75th percentiles

# Ensure all final results are numeric
for df in full_results_inthigh:
    df.apply(pd.to_numeric, errors='coerce')

def safe_percentile(x, q):
    if len(x) > 0:
        return np.percentile(x, q) 
    else:
        return np.nan

## Concatenate all realizations into a single DataFrame
data_inthigh = pd.concat(full_results_inthigh)

## Compute ponding depth metrics
ponding_depths_inthigh = data_inthigh.groupby('year')['ponding_depths'].apply(list).reset_index()
ponding_depths_inthigh.columns = ['year', 'ponding_depths_list']
ponding_depths_inthigh['combined_ponding_depths'] = ponding_depths_inthigh['ponding_depths_list'].apply(lambda x: list(np.concatenate(x)))
ponding_depths_inthigh['percentile_1'] = ponding_depths_inthigh['combined_ponding_depths'].apply(lambda x: safe_percentile(x, 1))
ponding_depths_inthigh['percentile_25'] = ponding_depths_inthigh['combined_ponding_depths'].apply(lambda x: safe_percentile(x, 25))
ponding_depths_inthigh['percentile_50'] = ponding_depths_inthigh['combined_ponding_depths'].apply(lambda x: safe_percentile(x, 50))
ponding_depths_inthigh['percentile_75'] = ponding_depths_inthigh['combined_ponding_depths'].apply(lambda x: safe_percentile(x, 75))
ponding_depths_inthigh['percentile_99'] = ponding_depths_inthigh['combined_ponding_depths'].apply(lambda x: safe_percentile(x, 99))
ponding_depths_inthigh.drop(ponding_depths_inthigh.tail(1).index, inplace=True)

## Compute event duration metrics
event_durations_inthigh = data_inthigh.groupby('year')['event_durations'].apply(list).reset_index()
event_durations_inthigh.columns = ['year', 'event_durations_list']
event_durations_inthigh['combined_event_durations'] = event_durations_inthigh['event_durations_list'].apply(lambda x: list(np.concatenate(x)))
event_durations_inthigh['percentile_1'] = event_durations_inthigh['combined_event_durations'].apply(lambda x: safe_percentile(x, 1))
event_durations_inthigh['percentile_25'] = event_durations_inthigh['combined_event_durations'].apply(lambda x: safe_percentile(x, 25))
event_durations_inthigh['percentile_50'] = event_durations_inthigh['combined_event_durations'].apply(lambda x: safe_percentile(x, 50))
event_durations_inthigh['percentile_75'] = event_durations_inthigh['combined_event_durations'].apply(lambda x: safe_percentile(x, 75))
event_durations_inthigh['percentile_99'] = event_durations_inthigh['combined_event_durations'].apply(lambda x: safe_percentile(x, 99))
event_durations_inthigh.drop(event_durations_inthigh.tail(1).index, inplace=True)

## Compute number of days emergent metrics
days_emergent_inthigh = data_inthigh.groupby('year')['days_with_flooding'].apply(list).reset_index()
days_emergent_inthigh['mean'] = days_emergent_inthigh['days_with_flooding'].apply(np.mean)
days_emergent_inthigh['std'] = days_emergent_inthigh['days_with_flooding'].apply(np.std)
## If mean is < 1, set mean and std to 0
days_emergent_inthigh.loc[days_emergent_inthigh['mean'] < 1, 'mean'] = 0
days_emergent_inthigh.loc[days_emergent_inthigh['mean'] < 1, 'std'] = 0
## Cap std at the mean
days_emergent_inthigh.loc[days_emergent_inthigh['std'] > days_emergent_inthigh['mean'], 'std'] = days_emergent_inthigh['mean']
days_emergent_inthigh.drop(days_emergent_inthigh.tail(1).index, inplace=True)

## Compute total hours emergent metrics
hours_emergent_inthigh = data_inthigh.groupby('year')['hours_above_threshold'].apply(list).reset_index()
hours_emergent_inthigh['mean'] = hours_emergent_inthigh['hours_above_threshold'].apply(np.mean)
hours_emergent_inthigh['std'] = hours_emergent_inthigh['hours_above_threshold'].apply(np.std)
## If mean is < 1, set mean and std to 0
hours_emergent_inthigh.loc[hours_emergent_inthigh['mean'] < 1, 'mean'] = 0
hours_emergent_inthigh.loc[hours_emergent_inthigh['mean'] < 1, 'std'] = 0
## Cap std at the mean
hours_emergent_inthigh.loc[hours_emergent_inthigh['std'] > hours_emergent_inthigh['mean'], 'std'] = hours_emergent_inthigh['mean']
hours_emergent_inthigh.drop(hours_emergent_inthigh.tail(1).index, inplace=True)

combined_seasonal_results = pd.concat(seasonal_results).groupby('year').agg(
    mean_winter_flooding=('winter_flooding', 'mean'),
    std_winter_flooding=('winter_flooding', 'std'),
    mean_winter_day_flood=('winter_day_flood', 'mean'),
    std_winter_day_flood=('winter_day_flood', 'std'),
    mean_winter_night_flood=('winter_night_flood', 'mean'),
    std_winter_night_flood=('winter_night_flood', 'std'),
    mean_winter_both_flood=('winter_both_flood', 'mean'),
    std_winter_both_flood=('winter_both_flood', 'std'),
    mean_summer_flooding=('summer_flooding', 'mean'),
    std_summer_flooding=('summer_flooding', 'std'),
    mean_summer_day_flood=('summer_day_flood', 'mean'),
    std_summer_day_flood=('summer_day_flood', 'std'),
    mean_summer_night_flood=('summer_night_flood', 'mean'),
    std_summer_night_flood=('summer_night_flood', 'std'),
    mean_summer_both_flood=('summer_both_flood', 'mean'),
    std_summer_both_flood=('summer_both_flood', 'std')
).reset_index()

# Drop the last row
combined_seasonal_results.drop(combined_seasonal_results.tail(1).index, inplace=True)

# Ensure that all columns are numeric
combined_seasonal_results = combined_seasonal_results.apply(pd.to_numeric, errors='coerce')

## Define capped_std to ensure standard deviation bars do not extend below 0
def capped_seasonal_std(std_column, mean_column):
    # Calculate capped values where std deviation does not make the bar fall below zero
    capped_seasonal_values = np.where(
        (combined_seasonal_results[mean_column] - combined_seasonal_results[std_column]) < 0,
        combined_seasonal_results[mean_column],
        combined_seasonal_results[std_column]
    )
    return capped_seasonal_values

# Cap standard deviations where needed
for column in ['std_winter_flooding', 'std_winter_day_flood', 
               'std_winter_night_flood', 'std_winter_both_flood', 'std_summer_flooding', 
               'std_summer_day_flood', 'std_summer_night_flood', 'std_summer_both_flood']:
    combined_seasonal_results[column] = capped_seasonal_std(column, column.replace('std', 'mean'))

## Plot Flood Metrics including Seasonal Metrics S Seacoast - Intermediate SLR Scenario

boxplot_depth = [
    [
        ponding_depths_inthigh['percentile_1'][i],
        ponding_depths_inthigh['percentile_25'][i],
        ponding_depths_inthigh['percentile_50'][i],
        ponding_depths_inthigh['percentile_75'][i],
        ponding_depths_inthigh['percentile_99'][i]
    ]
    for i in range(len(ponding_depths_inthigh['year']))
]

boxplot_duration = [
    [
        event_durations_inthigh['percentile_1'][i],
        event_durations_inthigh['percentile_25'][i],
        event_durations_inthigh['percentile_50'][i],
        event_durations_inthigh['percentile_75'][i],
        event_durations_inthigh['percentile_99'][i]
    ]
    for i in range(len(event_durations_inthigh['year']))
]

## PLOT
fontsize = 14
fig, ax = plt.subplots(5, 1, figsize=(12, 14), sharex=True)

for i, (min_val, q1, median, q3, max_val) in enumerate(boxplot_depth):
    ax[0].plot([ponding_depths_inthigh['year'][i], ponding_depths_inthigh['year'][i]], [min_val, max_val], color='black')  # Whiskers
    ax[0].plot([ponding_depths_inthigh['year'][i], ponding_depths_inthigh['year'][i]], [q1, q3], color='black', linewidth=7)  # Interquartile range (box)
    ax[0].plot([ponding_depths_inthigh['year'][i], ponding_depths_inthigh['year'][i]], [q1, q3], color='white', linewidth=5)  # Interquartile range (fill)
    ax[0].plot(ponding_depths_inthigh['year'][i], median, 'k_', markersize=7)  # Median line
ax[0].set_yticks(np.arange(0, 1.21, 0.2))
ax[0].set_ylabel('Ponding\nDepth (m)', fontsize=fontsize)
## Plot horizontal line at 0.25m and put text on left side, top of line plot 'Safe Limit for Emergency Vehicles'
ax[0].axhline(y=0.25, color='r', linestyle='--')
ax[0].text(2020.5, 0.27, 'Safe Limit for\nEmergency Vehicles', color='k', fontsize=fontsize-2)

ax[0].set_title('S Seacoast\nIntermediate High Sea Level Rise Scenario\nGroundwater Emergence Events', fontsize=fontsize+2)

for i, (min_val, q1, median, q3, max_val) in enumerate(boxplot_duration):
    ax[1].plot([event_durations_inthigh['year'][i], event_durations_inthigh['year'][i]], [min_val, max_val], color='black')  # Whiskers
    ax[1].plot([event_durations_inthigh['year'][i], event_durations_inthigh['year'][i]], [q1, q3], color='black', linewidth=7)  # Interquartile range (box)
    ax[1].plot([event_durations_inthigh['year'][i], event_durations_inthigh['year'][i]], [q1, q3], color='white', linewidth=5)  # Interquartile range (fill)
    ax[1].plot(event_durations_inthigh['year'][i], median, 'k_', markersize=7)  # Median line
ax[1].set_ylabel('Duration of\nEvents (hr)', fontsize=fontsize)
# ax[1].set_yticks(np.arange(0, 11, 2))
ax[1].set_ylim([0, 40])

# Plot the number of days with flooding
ax[2].bar(days_emergent_inthigh['year'], days_emergent_inthigh['mean'], yerr=days_emergent_inthigh['std'], color='mediumseagreen', capsize=3)
ax[2].set_ylabel('# Days\nEmergent', fontsize=fontsize)
ax[2].text(2016.5, 354, 'Daily', color='k', fontsize=fontsize-2)
ax[2].set_ylim([0, 365])

# Plot the total hours above threshold
ax[3].bar(hours_emergent_inthigh['year'], hours_emergent_inthigh['mean'], yerr=hours_emergent_inthigh['std'], color='mediumseagreen', capsize=3)
ax[3].set_ylabel('Total Time\nEmergent (hr)', fontsize=fontsize)
ax[3].set_yticks(np.arange(0, 9001, 1000))

# Stacked bar plot for seasonal day/night/both flooding
years = combined_seasonal_results['year']
summer_day = combined_seasonal_results['mean_summer_day_flood']
summer_night = combined_seasonal_results['mean_summer_night_flood']
summer_both = combined_seasonal_results['mean_summer_both_flood']
winter_day = combined_seasonal_results['mean_winter_day_flood']
winter_night = combined_seasonal_results['mean_winter_night_flood']
winter_both = combined_seasonal_results['mean_winter_both_flood']

ax[4].bar(years, winter_night + winter_both, label='Winter Night', color='indigo')
ax[4].bar(years, winter_day + winter_both, bottom=winter_night + winter_both, label='Winter Day', color='mediumorchid')
ax[4].bar(years, summer_night + summer_both, bottom=winter_night + winter_day + 2*winter_both, label='Summer Night', color='mediumblue')
ax[4].bar(years, summer_day + summer_both, bottom=winter_night + winter_day + 2*winter_both + summer_night + summer_both, label='Summer Day', color='deepskyblue')

# ax[4].set_ylim([0, 15])

# Set labels and title for stacked bar plot
ax[4].set_ylabel('Emergence Events\nBy Season & TOD', fontsize=fontsize)
handles, labels = ax[4].get_legend_handles_labels()
ax[4].legend(handles[::-1], labels[::-1], loc='upper left', ncol = 2,fontsize=fontsize-2)

# Set common x-axis label
ax[4].set_xlabel('Year', fontsize=fontsize)

# Set tick label font size
for a in ax:
    a.tick_params(axis='both', which='major', labelsize=fontsize-2)
    a.xaxis.grid(True, which='major')

## Restrict the x-axis
ax[0].set_xlim([2020, 2100])

plt.tight_layout()
plt.show()
#%% Flood metrics: S Seacoast - High Scenario

# Initialize results storage
full_results_high = []

seasonal_results = []

## Intermediate Scenario
for realization in gwt_high.columns:
    gwt_high_flooding = pd.DataFrame(gwt_high[realization])
    gwt_high_flooding.columns = ['gwt']
    gwt_high_flooding['year'] = gwt_high_flooding.index.year
    gwt_high_flooding['above_threshold'] = gwt_high_flooding['gwt'] > roadelevation_sseacoast
    gwt_high_flooding['flood_depth'] = gwt_high_flooding['gwt'] - roadelevation_sseacoast
    gwt_high_flooding['flood_depth'] = gwt_high_flooding['flood_depth'].apply(lambda x: x if x > 0 else np.NaN)
    ## Drop nans
    gwt_high_flooding = gwt_high_flooding.dropna()

    realization_results = []
    realization_seasonal_results = []

    for year, group in gwt_high_flooding.groupby('year'):
        group['event_start'] = (group.index.to_series().diff() != pd.Timedelta('1H'))
        # group['event_start'] = (group.index.to_series().diff().shift(-1) == pd.Timedelta('1H'))
        num_events = group['event_start'].sum()

        ## Metric 3a: Number of days emergent
        days_with_flooding = group.groupby(group.index.date)['above_threshold'].any().sum()

        ## Metric 4a: Total time emergent
        hours_above_threshold = group['above_threshold'].sum()

        ## Metric 2a: Duration of Events and Ponding Depths
        event_durations = []
        ponding_depths = []

        event_start_indices = group.index[group['event_start']].tolist()
        event_start_indices.append(group.index[-1] + pd.Timedelta('1H'))  # Add an end marker for the last event

        for i in range(len(event_start_indices) - 1):
            event_period = group.loc[event_start_indices[i]:event_start_indices[i + 1] - pd.Timedelta('1H')]
            event_durations.append(len(event_period))
            ponding_depths.append(event_period['flood_depth'].max())

        ## Seasonal metrics initialization
        winter_flooding = 0
        winter_day_flood = 0
        winter_night_flood = 0
        winter_both_flood = 0
        summer_flooding = 0
        summer_day_flood = 0
        summer_night_flood = 0
        summer_both_flood = 0

        # Calculate winter flooding
        winter_period = group[(group.index.month >= 10) | (group.index.month <= 3)]
        if not winter_period.empty:
            winter_flood_days = winter_period.groupby(winter_period.index.date)['above_threshold'].any()
            winter_flooding = winter_flood_days.sum()

            # Determine day/night/both flooding for winter period
            for day, has_flooding in winter_flood_days.items():
                if has_flooding:
                    day_data = winter_period.loc[str(day)]
                    day_flood = day_data.loc[(day_data.index.hour >= 12) & (day_data.index.hour < 24), 'above_threshold'].any()
                    night_flood = day_data.loc[(day_data.index.hour >= 0) & (day_data.index.hour < 12), 'above_threshold'].any()
                    if day_flood and night_flood:
                        winter_both_flood += 1
                    elif day_flood:
                        winter_day_flood += 1
                    elif night_flood:
                        winter_night_flood += 1

        # Calculate summer flooding
        summer_period = group[(group.index.month >= 4) & (group.index.month <= 9)]
        if not summer_period.empty:
            summer_flood_days = summer_period.groupby(summer_period.index.date)['above_threshold'].any()
            summer_flooding = summer_flood_days.sum()

            # Determine day/night/both flooding for summer period
            for day, has_flooding in summer_flood_days.items():
                if has_flooding:
                    day_data = summer_period.loc[str(day)]
                    day_flood = day_data.loc[(day_data.index.hour >= 12) & (day_data.index.hour < 24), 'above_threshold'].any()
                    night_flood = day_data.loc[(day_data.index.hour >= 0) & (day_data.index.hour < 12), 'above_threshold'].any()
                    if day_flood and night_flood:
                        summer_both_flood += 1
                    elif day_flood:
                        summer_day_flood += 1
                    elif night_flood:
                        summer_night_flood += 1

        ## Save Realization Results
        realization_results.append({
            'year': year,
            'ponding_depths': ponding_depths,
            'days_with_flooding': days_with_flooding,
            'hours_above_threshold': hours_above_threshold,
            'num_events': num_events,
            'event_durations': event_durations,
        })

        realization_seasonal_results.append({
            'year': year,
            'winter_flooding': winter_flooding,
            'winter_day_flood': winter_day_flood,
            'winter_night_flood': winter_night_flood,
            'winter_both_flood': winter_both_flood,
            'summer_flooding': summer_flooding,
            'summer_day_flood': summer_day_flood,
            'summer_night_flood': summer_night_flood,
            'summer_both_flood': summer_both_flood
        })

    full_results_high.append(pd.DataFrame(realization_results))
    seasonal_results.append(pd.DataFrame(realization_seasonal_results))

## Compute metrics from the full results by year S Seacoast - Intermediate Scenario
## Compute the minimum, maximum, 25th, 50th, and 75th percentiles

# Ensure all final results are numeric
for df in full_results_high:
    df.apply(pd.to_numeric, errors='coerce')

def safe_percentile(x, q):
    if len(x) > 0:
        return np.percentile(x, q) 
    else:
        return np.nan

## Concatenate all realizations into a single DataFrame
data_high = pd.concat(full_results_high)

## Compute ponding depth metrics
ponding_depths_high = data_high.groupby('year')['ponding_depths'].apply(list).reset_index()
ponding_depths_high.columns = ['year', 'ponding_depths_list']
ponding_depths_high['combined_ponding_depths'] = ponding_depths_high['ponding_depths_list'].apply(lambda x: list(np.concatenate(x)))
ponding_depths_high['percentile_1'] = ponding_depths_high['combined_ponding_depths'].apply(lambda x: safe_percentile(x, 1))
ponding_depths_high['percentile_25'] = ponding_depths_high['combined_ponding_depths'].apply(lambda x: safe_percentile(x, 25))
ponding_depths_high['percentile_50'] = ponding_depths_high['combined_ponding_depths'].apply(lambda x: safe_percentile(x, 50))
ponding_depths_high['percentile_75'] = ponding_depths_high['combined_ponding_depths'].apply(lambda x: safe_percentile(x, 75))
ponding_depths_high['percentile_99'] = ponding_depths_high['combined_ponding_depths'].apply(lambda x: safe_percentile(x, 99))
ponding_depths_high.drop(ponding_depths_high.tail(1).index, inplace=True)

## Compute event duration metrics
event_durations_high = data_high.groupby('year')['event_durations'].apply(list).reset_index()
event_durations_high.columns = ['year', 'event_durations_list']
event_durations_high['combined_event_durations'] = event_durations_high['event_durations_list'].apply(lambda x: list(np.concatenate(x)))
event_durations_high['percentile_1'] = event_durations_high['combined_event_durations'].apply(lambda x: safe_percentile(x, 1))
event_durations_high['percentile_25'] = event_durations_high['combined_event_durations'].apply(lambda x: safe_percentile(x, 25))
event_durations_high['percentile_50'] = event_durations_high['combined_event_durations'].apply(lambda x: safe_percentile(x, 50))
event_durations_high['percentile_75'] = event_durations_high['combined_event_durations'].apply(lambda x: safe_percentile(x, 75))
event_durations_high['percentile_99'] = event_durations_high['combined_event_durations'].apply(lambda x: safe_percentile(x, 99))
event_durations_high.drop(event_durations_high.tail(1).index, inplace=True)

## Compute number of days emergent metrics
days_emergent_high = data_high.groupby('year')['days_with_flooding'].apply(list).reset_index()
days_emergent_high['mean'] = days_emergent_high['days_with_flooding'].apply(np.mean)
days_emergent_high['std'] = days_emergent_high['days_with_flooding'].apply(np.std)
## If mean is < 1, set mean and std to 0
days_emergent_high.loc[days_emergent_high['mean'] < 1, 'mean'] = 0
days_emergent_high.loc[days_emergent_high['mean'] < 1, 'std'] = 0
## Cap std at the mean
days_emergent_high.loc[days_emergent_high['std'] > days_emergent_high['mean'], 'std'] = days_emergent_high['mean']
days_emergent_high.drop(days_emergent_high.tail(1).index, inplace=True)

## Compute total hours emergent metrics
hours_emergent_high = data_high.groupby('year')['hours_above_threshold'].apply(list).reset_index()
hours_emergent_high['mean'] = hours_emergent_high['hours_above_threshold'].apply(np.mean)
hours_emergent_high['std'] = hours_emergent_high['hours_above_threshold'].apply(np.std)
## If mean is < 1, set mean and std to 0
hours_emergent_high.loc[hours_emergent_high['mean'] < 1, 'mean'] = 0
hours_emergent_high.loc[hours_emergent_high['mean'] < 1, 'std'] = 0
## Cap std at the mean
hours_emergent_high.loc[hours_emergent_high['std'] > hours_emergent_high['mean'], 'std'] = hours_emergent_high['mean']
hours_emergent_high.drop(hours_emergent_high.tail(1).index, inplace=True)

combined_seasonal_results = pd.concat(seasonal_results).groupby('year').agg(
    mean_winter_flooding=('winter_flooding', 'mean'),
    std_winter_flooding=('winter_flooding', 'std'),
    mean_winter_day_flood=('winter_day_flood', 'mean'),
    std_winter_day_flood=('winter_day_flood', 'std'),
    mean_winter_night_flood=('winter_night_flood', 'mean'),
    std_winter_night_flood=('winter_night_flood', 'std'),
    mean_winter_both_flood=('winter_both_flood', 'mean'),
    std_winter_both_flood=('winter_both_flood', 'std'),
    mean_summer_flooding=('summer_flooding', 'mean'),
    std_summer_flooding=('summer_flooding', 'std'),
    mean_summer_day_flood=('summer_day_flood', 'mean'),
    std_summer_day_flood=('summer_day_flood', 'std'),
    mean_summer_night_flood=('summer_night_flood', 'mean'),
    std_summer_night_flood=('summer_night_flood', 'std'),
    mean_summer_both_flood=('summer_both_flood', 'mean'),
    std_summer_both_flood=('summer_both_flood', 'std')
).reset_index()

# Drop the last row
combined_seasonal_results.drop(combined_seasonal_results.tail(1).index, inplace=True)

# Ensure that all columns are numeric
combined_seasonal_results = combined_seasonal_results.apply(pd.to_numeric, errors='coerce')

## Define capped_std to ensure standard deviation bars do not extend below 0
def capped_seasonal_std(std_column, mean_column):
    # Calculate capped values where std deviation does not make the bar fall below zero
    capped_seasonal_values = np.where(
        (combined_seasonal_results[mean_column] - combined_seasonal_results[std_column]) < 0,
        combined_seasonal_results[mean_column],
        combined_seasonal_results[std_column]
    )
    return capped_seasonal_values

# Cap standard deviations where needed
for column in ['std_winter_flooding', 'std_winter_day_flood', 
               'std_winter_night_flood', 'std_winter_both_flood', 'std_summer_flooding', 
               'std_summer_day_flood', 'std_summer_night_flood', 'std_summer_both_flood']:
    combined_seasonal_results[column] = capped_seasonal_std(column, column.replace('std', 'mean'))

## Plot Flood Metrics including Seasonal Metrics S Seacoast - Intermediate SLR Scenario

boxplot_depth = [
    [
        ponding_depths_high['percentile_1'][i],
        ponding_depths_high['percentile_25'][i],
        ponding_depths_high['percentile_50'][i],
        ponding_depths_high['percentile_75'][i],
        ponding_depths_high['percentile_99'][i]
    ]
    for i in range(len(ponding_depths_high['year']))
]

boxplot_duration = [
    [
        event_durations_high['percentile_1'][i],
        event_durations_high['percentile_25'][i],
        event_durations_high['percentile_50'][i],
        event_durations_high['percentile_75'][i],
        event_durations_high['percentile_99'][i]
    ]
    for i in range(len(event_durations_high['year']))
]

## PLOT
fontsize = 14
fig, ax = plt.subplots(5, 1, figsize=(12, 14), sharex=True)

for i, (min_val, q1, median, q3, max_val) in enumerate(boxplot_depth):
    ax[0].plot([ponding_depths_high['year'][i], ponding_depths_high['year'][i]], [min_val, max_val], color='black')  # Whiskers
    ax[0].plot([ponding_depths_high['year'][i], ponding_depths_high['year'][i]], [q1, q3], color='black', linewidth=7)  # Interquartile range (box)
    ax[0].plot([ponding_depths_high['year'][i], ponding_depths_high['year'][i]], [q1, q3], color='white', linewidth=5)  # Interquartile range (fill)
    ax[0].plot(ponding_depths_high['year'][i], median, 'k_', markersize=7)  # Median line
ax[0].set_yticks(np.arange(0, 2.01, 0.25))
ax[0].set_ylabel('Ponding\nDepth (m)', fontsize=fontsize)
## Plot horizontal line at 0.25m and put text on left side, top of line plot 'Safe Limit for Emergency Vehicles'
ax[0].axhline(y=0.25, color='r', linestyle='--')
ax[0].text(2020.5, 0.27, 'Safe Limit for\nEmergency Vehicles', color='k', fontsize=fontsize-2)

ax[0].set_title('S Seacoast\nHigh Sea Level Rise Scenario\nGroundwater Emergence Events', fontsize=fontsize+2)

for i, (min_val, q1, median, q3, max_val) in enumerate(boxplot_duration):
    ax[1].plot([event_durations_high['year'][i], event_durations_high['year'][i]], [min_val, max_val], color='black')  # Whiskers
    ax[1].plot([event_durations_high['year'][i], event_durations_high['year'][i]], [q1, q3], color='black', linewidth=7)  # Interquartile range (box)
    ax[1].plot([event_durations_high['year'][i], event_durations_high['year'][i]], [q1, q3], color='white', linewidth=5)  # Interquartile range (fill)
    ax[1].plot(event_durations_high['year'][i], median, 'k_', markersize=7)  # Median line
ax[1].set_ylabel('Duration of\nEvents (hr)', fontsize=fontsize)
ax[1].set_yticks(np.arange(0, 51, 10))
ax[1].set_ylim([0, 40])
## Add text "Near Permanent\nInundation" at x = 2092, y = 20, in red
ax[1].text(2092, 10, 'Nearly\nPermanent\nInundation', color='r', fontsize=fontsize-2)

# Plot the number of days with flooding
ax[2].bar(days_emergent_high['year'], days_emergent_high['mean'], yerr=days_emergent_high['std'], color='mediumseagreen', capsize=3)
ax[2].set_ylabel('# Days\nEmergent', fontsize=fontsize)
ax[2].text(2016.5, 354, 'Daily', color='k', fontsize=fontsize-2)
ax[2].set_ylim([0, 365])

# Plot the total hours above threshold
ax[3].bar(hours_emergent_high['year'], hours_emergent_high['mean'], yerr=hours_emergent_high['std'], color='mediumseagreen', capsize=3)
ax[3].set_ylabel('Total Time\nEmergent (hr)', fontsize=fontsize)
ax[3].set_yticks(np.arange(0, 9001, 1000))

# Stacked bar plot for seasonal day/night/both flooding
years = combined_seasonal_results['year']
summer_day = combined_seasonal_results['mean_summer_day_flood']
summer_night = combined_seasonal_results['mean_summer_night_flood']
summer_both = combined_seasonal_results['mean_summer_both_flood']
winter_day = combined_seasonal_results['mean_winter_day_flood']
winter_night = combined_seasonal_results['mean_winter_night_flood']
winter_both = combined_seasonal_results['mean_winter_both_flood']

ax[4].bar(years, winter_night + winter_both, label='Winter Night', color='indigo')
ax[4].bar(years, winter_day + winter_both, bottom=winter_night + winter_both, label='Winter Day', color='mediumorchid')
ax[4].bar(years, summer_night + summer_both, bottom=winter_night + winter_day + 2*winter_both, label='Summer Night', color='mediumblue')
ax[4].bar(years, summer_day + summer_both, bottom=winter_night + winter_day + 2*winter_both + summer_night + summer_both, label='Summer Day', color='deepskyblue')

# ax[4].set_ylim([0, 15])

# Set labels and title for stacked bar plot
ax[4].set_ylabel('Emergence Events\nBy Season & TOD', fontsize=fontsize)
handles, labels = ax[4].get_legend_handles_labels()
ax[4].legend(handles[::-1], labels[::-1], loc='upper left', ncol = 2,fontsize=fontsize-2)

# Set common x-axis label
ax[4].set_xlabel('Year', fontsize=fontsize)

# Set tick label font size
for a in ax:
    a.tick_params(axis='both', which='major', labelsize=fontsize-2)
    a.xaxis.grid(True, which='major')

## Restrict the x-axis
ax[0].set_xlim([2020, 2100])

plt.tight_layout()
plt.show()

#%% Flood metrics: Palm - Intermediate Scenario

# Initialize results storage
full_results_int = []

seasonal_results = []

## Intermediate Scenario
for realization in gwt_int.columns:
    gwt_int_flooding = pd.DataFrame(gwt_int[realization])
    gwt_int_flooding.columns = ['gwt']
    gwt_int_flooding['year'] = gwt_int_flooding.index.year
    gwt_int_flooding['above_threshold'] = gwt_int_flooding['gwt'] > roadelevation_palm
    gwt_int_flooding['flood_depth'] = gwt_int_flooding['gwt'] - roadelevation_palm
    gwt_int_flooding['flood_depth'] = gwt_int_flooding['flood_depth'].apply(lambda x: x if x > 0 else np.NaN)
    ## Drop nans
    gwt_int_flooding = gwt_int_flooding.dropna()

    realization_results = []
    realization_seasonal_results = []

    for year, group in gwt_int_flooding.groupby('year'):
        group['event_start'] = (group.index.to_series().diff() != pd.Timedelta('1H'))
        # group['event_start'] = (group.index.to_series().diff().shift(-1) == pd.Timedelta('1H'))
        num_events = group['event_start'].sum()

        ## Metric 3a: Number of days emergent
        days_with_flooding = group.groupby(group.index.date)['above_threshold'].any().sum()

        ## Metric 4a: Total time emergent
        hours_above_threshold = group['above_threshold'].sum()

        ## Metric 2a: Duration of Events and Ponding Depths
        event_durations = []
        ponding_depths = []

        event_start_indices = group.index[group['event_start']].tolist()
        event_start_indices.append(group.index[-1] + pd.Timedelta('1H'))  # Add an end marker for the last event

        for i in range(len(event_start_indices) - 1):
            event_period = group.loc[event_start_indices[i]:event_start_indices[i + 1] - pd.Timedelta('1H')]
            event_durations.append(len(event_period))
            ponding_depths.append(event_period['flood_depth'].max())

        ## Seasonal metrics initialization
        winter_flooding = 0
        winter_day_flood = 0
        winter_night_flood = 0
        winter_both_flood = 0
        summer_flooding = 0
        summer_day_flood = 0
        summer_night_flood = 0
        summer_both_flood = 0

        # Calculate winter flooding
        winter_period = group[(group.index.month >= 10) | (group.index.month <= 3)]
        if not winter_period.empty:
            winter_flood_days = winter_period.groupby(winter_period.index.date)['above_threshold'].any()
            winter_flooding = winter_flood_days.sum()

            # Determine day/night/both flooding for winter period
            for day, has_flooding in winter_flood_days.items():
                if has_flooding:
                    day_data = winter_period.loc[str(day)]
                    day_flood = day_data.loc[(day_data.index.hour >= 12) & (day_data.index.hour < 24), 'above_threshold'].any()
                    night_flood = day_data.loc[(day_data.index.hour >= 0) & (day_data.index.hour < 12), 'above_threshold'].any()
                    if day_flood and night_flood:
                        winter_both_flood += 1
                    elif day_flood:
                        winter_day_flood += 1
                    elif night_flood:
                        winter_night_flood += 1

        # Calculate summer flooding
        summer_period = group[(group.index.month >= 4) & (group.index.month <= 9)]
        if not summer_period.empty:
            summer_flood_days = summer_period.groupby(summer_period.index.date)['above_threshold'].any()
            summer_flooding = summer_flood_days.sum()

            # Determine day/night/both flooding for summer period
            for day, has_flooding in summer_flood_days.items():
                if has_flooding:
                    day_data = summer_period.loc[str(day)]
                    day_flood = day_data.loc[(day_data.index.hour >= 12) & (day_data.index.hour < 24), 'above_threshold'].any()
                    night_flood = day_data.loc[(day_data.index.hour >= 0) & (day_data.index.hour < 12), 'above_threshold'].any()
                    if day_flood and night_flood:
                        summer_both_flood += 1
                    elif day_flood:
                        summer_day_flood += 1
                    elif night_flood:
                        summer_night_flood += 1

        ## Save Realization Results
        realization_results.append({
            'year': year,
            'ponding_depths': ponding_depths,
            'days_with_flooding': days_with_flooding,
            'hours_above_threshold': hours_above_threshold,
            'num_events': num_events,
            'event_durations': event_durations,
        })

        realization_seasonal_results.append({
            'year': year,
            'winter_flooding': winter_flooding,
            'winter_day_flood': winter_day_flood,
            'winter_night_flood': winter_night_flood,
            'winter_both_flood': winter_both_flood,
            'summer_flooding': summer_flooding,
            'summer_day_flood': summer_day_flood,
            'summer_night_flood': summer_night_flood,
            'summer_both_flood': summer_both_flood
        })

    full_results_int.append(pd.DataFrame(realization_results))
    seasonal_results.append(pd.DataFrame(realization_seasonal_results))

## Compute metrics from the full results by year
## Compute the minimum, maximum, 25th, 50th, and 75th percentiles

# Ensure all final results are numeric
for df in full_results_int:
    df.apply(pd.to_numeric, errors='coerce')

def safe_percentile(x, q):
    if len(x) > 0:
        return np.percentile(x, q) 
    else:
        return np.nan

## Concatenate all realizations into a single DataFrame
data_int = pd.concat(full_results_int)

## Compute ponding depth metrics
ponding_depths_int = data_int.groupby('year')['ponding_depths'].apply(list).reset_index()
ponding_depths_int.columns = ['year', 'ponding_depths_list']
ponding_depths_int['combined_ponding_depths'] = ponding_depths_int['ponding_depths_list'].apply(lambda x: list(np.concatenate(x)))
ponding_depths_int['percentile_1'] = ponding_depths_int['combined_ponding_depths'].apply(lambda x: safe_percentile(x, 1))
ponding_depths_int['percentile_25'] = ponding_depths_int['combined_ponding_depths'].apply(lambda x: safe_percentile(x, 25))
ponding_depths_int['percentile_50'] = ponding_depths_int['combined_ponding_depths'].apply(lambda x: safe_percentile(x, 50))
ponding_depths_int['percentile_75'] = ponding_depths_int['combined_ponding_depths'].apply(lambda x: safe_percentile(x, 75))
ponding_depths_int['percentile_99'] = ponding_depths_int['combined_ponding_depths'].apply(lambda x: safe_percentile(x, 99))
ponding_depths_int.drop(ponding_depths_int.tail(1).index, inplace=True)

## Compute event duration metrics
event_durations_int = data_int.groupby('year')['event_durations'].apply(list).reset_index()
event_durations_int.columns = ['year', 'event_durations_list']
event_durations_int['combined_event_durations'] = event_durations_int['event_durations_list'].apply(lambda x: list(np.concatenate(x)))
event_durations_int['percentile_1'] = event_durations_int['combined_event_durations'].apply(lambda x: safe_percentile(x, 1))
event_durations_int['percentile_25'] = event_durations_int['combined_event_durations'].apply(lambda x: safe_percentile(x, 25))
event_durations_int['percentile_50'] = event_durations_int['combined_event_durations'].apply(lambda x: safe_percentile(x, 50))
event_durations_int['percentile_75'] = event_durations_int['combined_event_durations'].apply(lambda x: safe_percentile(x, 75))
event_durations_int['percentile_99'] = event_durations_int['combined_event_durations'].apply(lambda x: safe_percentile(x, 99))
event_durations_int.drop(event_durations_int.tail(1).index, inplace=True)

## Compute number of days emergent metrics
days_emergent_int = data_int.groupby('year')['days_with_flooding'].apply(list).reset_index()
days_emergent_int['mean'] = days_emergent_int['days_with_flooding'].apply(np.mean)
days_emergent_int['std'] = days_emergent_int['days_with_flooding'].apply(np.std)
## If mean is < 1, set mean and std to 0
days_emergent_int.loc[days_emergent_int['mean'] < 1, 'mean'] = 0
days_emergent_int.loc[days_emergent_int['mean'] < 1, 'std'] = 0
## Cap std at the mean
days_emergent_int.loc[days_emergent_int['std'] > days_emergent_int['mean'], 'std'] = days_emergent_int['mean']
days_emergent_int.drop(days_emergent_int.tail(1).index, inplace=True)

## Compute total hours emergent metrics
hours_emergent_int = data_int.groupby('year')['hours_above_threshold'].apply(list).reset_index()
hours_emergent_int['mean'] = hours_emergent_int['hours_above_threshold'].apply(np.mean)
hours_emergent_int['std'] = hours_emergent_int['hours_above_threshold'].apply(np.std)
## If mean is < 1, set mean and std to 0
hours_emergent_int.loc[hours_emergent_int['mean'] < 1, 'mean'] = 0
hours_emergent_int.loc[hours_emergent_int['mean'] < 1, 'std'] = 0
## Cap std at the mean
hours_emergent_int.loc[hours_emergent_int['std'] > hours_emergent_int['mean'], 'std'] = hours_emergent_int['mean']
hours_emergent_int.drop(hours_emergent_int.tail(1).index, inplace=True)

combined_seasonal_results = pd.concat(seasonal_results).groupby('year').agg(
    mean_winter_flooding=('winter_flooding', 'mean'),
    std_winter_flooding=('winter_flooding', 'std'),
    mean_winter_day_flood=('winter_day_flood', 'mean'),
    std_winter_day_flood=('winter_day_flood', 'std'),
    mean_winter_night_flood=('winter_night_flood', 'mean'),
    std_winter_night_flood=('winter_night_flood', 'std'),
    mean_winter_both_flood=('winter_both_flood', 'mean'),
    std_winter_both_flood=('winter_both_flood', 'std'),
    mean_summer_flooding=('summer_flooding', 'mean'),
    std_summer_flooding=('summer_flooding', 'std'),
    mean_summer_day_flood=('summer_day_flood', 'mean'),
    std_summer_day_flood=('summer_day_flood', 'std'),
    mean_summer_night_flood=('summer_night_flood', 'mean'),
    std_summer_night_flood=('summer_night_flood', 'std'),
    mean_summer_both_flood=('summer_both_flood', 'mean'),
    std_summer_both_flood=('summer_both_flood', 'std')
).reset_index()

# Drop the last row
combined_seasonal_results.drop(combined_seasonal_results.tail(1).index, inplace=True)

# Ensure that all columns are numeric
combined_seasonal_results = combined_seasonal_results.apply(pd.to_numeric, errors='coerce')

## Define capped_std to ensure standard deviation bars do not extend below 0
def capped_seasonal_std(std_column, mean_column):
    # Calculate capped values where std deviation does not make the bar fall below zero
    capped_seasonal_values = np.where(
        (combined_seasonal_results[mean_column] - combined_seasonal_results[std_column]) < 0,
        combined_seasonal_results[mean_column],
        combined_seasonal_results[std_column]
    )
    return capped_seasonal_values

# Cap standard deviations where needed
for column in ['std_winter_flooding', 'std_winter_day_flood', 
               'std_winter_night_flood', 'std_winter_both_flood', 'std_summer_flooding', 
               'std_summer_day_flood', 'std_summer_night_flood', 'std_summer_both_flood']:
    combined_seasonal_results[column] = capped_seasonal_std(column, column.replace('std', 'mean'))

## Plot Flood Metrics including Seasonal Metrics - Intermediate SLR Scenario

boxplot_depth = [
    [
        ponding_depths_int['percentile_1'][i],
        ponding_depths_int['percentile_25'][i],
        ponding_depths_int['percentile_50'][i],
        ponding_depths_int['percentile_75'][i],
        ponding_depths_int['percentile_99'][i]
    ]
    for i in range(len(ponding_depths_int['year']))
]

boxplot_duration = [
    [
        event_durations_int['percentile_1'][i],
        event_durations_int['percentile_25'][i],
        event_durations_int['percentile_50'][i],
        event_durations_int['percentile_75'][i],
        event_durations_int['percentile_99'][i]
    ]
    for i in range(len(event_durations_int['year']))
]

fontsize = 14
fig, ax = plt.subplots(5, 1, figsize=(12, 14), sharex=True)

# ## OPTION 1: Plot Mean Flood Depth
# ax[0].bar(ponding_depths_int['year'], ponding_depths_int['mean_flood_depth'], yerr=ponding_depths_int['std_flood_depth'], color='blue', capsize=3)
# ax[0].set_ylabel('Mean Flood\nDepth (m)', fontsize=fontsize)

# ## OPTION 2: Box & Whisker plot for ponding_depths_int['flood_depth_min', 'flood_depth_25th', 'flood_depth_50th', 'flood_depth_75th', 'flood_depth_max']
# Create custom boxplots using the summary stats
for i, (min_val, q1, median, q3, max_val) in enumerate(boxplot_depth):
    ax[0].plot([ponding_depths_int['year'][i], ponding_depths_int['year'][i]], [min_val, max_val], color='black')  # Whiskers
    ax[0].plot([ponding_depths_int['year'][i], ponding_depths_int['year'][i]], [q1, q3], color='black', linewidth=7)  # Interquartile range (box)
    ax[0].plot([ponding_depths_int['year'][i], ponding_depths_int['year'][i]], [q1, q3], color='white', linewidth=5)  # Interquartile range (fill)
    ax[0].plot(ponding_depths_int['year'][i], median, 'k_', markersize=7)  # Median line
ax[0].set_yticks(np.arange(0, 0.61, 0.1))
ax[0].set_ylabel('Ponding\nDepth (m)', fontsize=fontsize)
## Plot horizontal line at 0.25m and put text on left side, top of line plot 'Safe Limit for Emergency Vehicles'
ax[0].axhline(y=0.25, color='r', linestyle='--')
ax[0].text(2020.5, 0.27, 'Safe Limit for\nEmergency Vehicles', color='k', fontsize=fontsize-2)

# ax[0].set_title('D0038\nIntermediate Sea Level Rise Scenario\nGroundwater Emergence Events', fontsize=fontsize+2)
ax[0].set_title('Palm Ave\nIntermediate Sea Level Rise Scenario\nGroundwater Emergence Events', fontsize=fontsize+2)

# ## OPTION 1: Plot the average hours per event
# [ax[1].bar(event_durations_int['year'], event_durations_int['mean_avg_hours_per_event'], yerr=event_durations_int['std_avg_hours_per_event'], color='blue', capsize=3)
# [ax[1].set_ylabel('Avg Flood\nDuration (hr)', fontsize=fontsize)

## OPTION 2: Box & Whisker plot for event_durations_int['duration_min', 'duration_25th', 'duration_50th', 'duration_75th', 'duration_max']
# Create custom boxplots using the summary stats
for i, (min_val, q1, median, q3, max_val) in enumerate(boxplot_duration):
    ax[1].plot([event_durations_int['year'][i], event_durations_int['year'][i]], [min_val, max_val], color='black')  # Whiskers
    ax[1].plot([event_durations_int['year'][i], event_durations_int['year'][i]], [q1, q3], color='black', linewidth=7)  # Interquartile range (box)
    ax[1].plot([event_durations_int['year'][i], event_durations_int['year'][i]], [q1, q3], color='white', linewidth=5)  # Interquartile range (fill)
    ax[1].plot(event_durations_int['year'][i], median, 'k_', markersize=7)  # Median line
ax[1].set_ylabel('Duration of\nEvents (hr)', fontsize=fontsize)
## OPTION 2b: boxplot using boxplot matplotlib function (need all data for boxplot to compute stats on)
# bp = ax[1].boxplot(boxplot_data, positions=event_durations_int['year'], widths=0.6)
# Customize the boxplot appearance
# for element in ['boxes', 'whiskers','means', 'medians', 'caps']:
#     plt.setp(bp[element], color='black')
# plt.setp(bp['fliers'], marker='o', markersize=3, markerfacecolor='red')
ax[1].set_yticks(np.arange(0, 21, 5))

# Plot the number of days with flooding
ax[2].bar(days_emergent_int['year'], days_emergent_int['mean'], yerr=days_emergent_int['std'], color='mediumseagreen', capsize=3)
ax[2].set_ylabel('# Days\nEmergent', fontsize=fontsize)
ax[2].text(2016.5, 354, 'Daily', color='k', fontsize=fontsize-2)
ax[2].set_ylim([0, 365])

# Plot the total hours above threshold
ax[3].bar(hours_emergent_int['year'], hours_emergent_int['mean'], yerr=hours_emergent_int['std'], color='mediumseagreen', capsize=3)
ax[3].set_ylabel('Total Time\nEmergent (hr)', fontsize=fontsize)
## Make y-axis plot ticks and labels every 1000
ax[3].set_yticks(np.arange(0, 4001, 1000))

# Stacked bar plot for seasonal day/night/both flooding
years = combined_seasonal_results['year']
summer_day = combined_seasonal_results['mean_summer_day_flood']
summer_night = combined_seasonal_results['mean_summer_night_flood']
summer_both = combined_seasonal_results['mean_summer_both_flood']
winter_day = combined_seasonal_results['mean_winter_day_flood']
winter_night = combined_seasonal_results['mean_winter_night_flood']
winter_both = combined_seasonal_results['mean_winter_both_flood']

# ## OPTION 1: Plot Day/Night/Both (units = # days)
# ax[4].bar(years, summer_day, label='Summer Day', color='lightblue')
# ax[4].bar(years, summer_night, bottom=summer_day, label='Summer Night', color='skyblue')
# ax[4].bar(years, summer_both, bottom=summer_day + summer_night, label='Summer Both', color='deepskyblue')
# ax[4].bar(years, winter_day, bottom=summer_day + summer_night + summer_both, label='Winter Day', color='lightcoral')
# ax[4].bar(years, winter_night, bottom=summer_day + summer_night + summer_both + winter_day, label='Winter Night', color='indianred')
# ax[4].bar(years, winter_both, bottom=summer_day + summer_night + summer_both + winter_day + winter_night, label='Winter Both', color='firebrick')

## OPTION 2: Plot Day/Night (units = # events)
# ax[4].bar(years, winter_night + winter_both, label='Winter Night', color='indigo')
# ax[4].bar(years, winter_day + winter_both, bottom=winter_night + winter_both, label='Winter Day', color='plum')
# ax[4].bar(years, summer_night + summer_both, bottom=winter_night + winter_day + 2*winter_both, label='Summer Night', color='royalblue')
# ax[4].bar(years, summer_day + summer_both, bottom=winter_night + winter_day + 2*winter_both + summer_night + summer_both, label='Summer Day', color='deepskyblue')
ax[4].bar(years, winter_night + winter_both, label='Winter Night', color='indigo')
ax[4].bar(years, winter_day + winter_both, bottom=winter_night + winter_both, label='Winter Day', color='mediumorchid')
ax[4].bar(years, summer_night + summer_both, bottom=winter_night + winter_day + 2*winter_both, label='Summer Night', color='mediumblue')
ax[4].bar(years, summer_day + summer_both, bottom=winter_night + winter_day + 2*winter_both + summer_night + summer_both, label='Summer Day', color='deepskyblue')
# Add error bars for standard deviation
# ax[4].errorbar(years, winter_night + winter_both, 
#                yerr=combined_seasonal_results['std_winter_night_flood']+combined_seasonal_results['std_winter_both_flood'], fmt='none', ecolor='plum', capsize=2)
# ax[4].errorbar(years, winter_night + winter_day + 2*winter_both, 
#                yerr=combined_seasonal_results['std_winter_day_flood']+combined_seasonal_results['std_winter_both_flood'], fmt='none', ecolor='plum', capsize=2)
# ax[4].errorbar(years, winter_night + winter_day + 2*winter_both + summer_night + summer_both, 
#                yerr=combined_seasonal_results['std_summer_night_flood']+combined_seasonal_results['std_summer_both_flood'], fmt='none', ecolor='lightcyan', capsize=2)
# ax[4].errorbar(years, winter_night + winter_day + 2*winter_both + summer_night + summer_day + 2*summer_both, 
#                yerr=combined_seasonal_results['std_summer_day_flood']+combined_seasonal_results['std_summer_both_flood'], fmt='none', ecolor='blue', capsize=2)

# Set labels and title for stacked bar plot
ax[4].set_ylabel('Emergence Events\nBy Season & TOD', fontsize=fontsize)
handles, labels = ax[4].get_legend_handles_labels()
ax[4].legend(handles[::-1], labels[::-1], loc='upper left', ncol=2, fontsize=fontsize-2)

# Set common x-axis label
ax[4].set_xlabel('Year', fontsize=fontsize)

# Set tick label font size
for a in ax:
    a.tick_params(axis='both', which='major', labelsize=fontsize-2)
    a.xaxis.grid(True, which='major')

## Restrict the x-axis
ax[0].set_xlim([2020, 2100])

plt.tight_layout()
plt.show()

#%% Flood metrics: Encanto - Intermediate Scenario

# Initialize results storage
full_results_int = []

seasonal_results = []

## Intermediate Scenario
for realization in gwt_int.columns:
    gwt_int_flooding = pd.DataFrame(gwt_int[realization])
    gwt_int_flooding.columns = ['gwt']
    gwt_int_flooding['year'] = gwt_int_flooding.index.year
    gwt_int_flooding['above_threshold'] = gwt_int_flooding['gwt'] > roadelevation_encanto
    gwt_int_flooding['flood_depth'] = gwt_int_flooding['gwt'] - roadelevation_encanto
    gwt_int_flooding['flood_depth'] = gwt_int_flooding['flood_depth'].apply(lambda x: x if x > 0 else np.NaN)
    ## Drop nans
    gwt_int_flooding = gwt_int_flooding.dropna()

    realization_results = []
    realization_seasonal_results = []

    for year, group in gwt_int_flooding.groupby('year'):
        group['event_start'] = (group.index.to_series().diff() != pd.Timedelta('1H'))
        # group['event_start'] = (group.index.to_series().diff().shift(-1) == pd.Timedelta('1H'))
        num_events = group['event_start'].sum()

        ## Metric 3a: Number of days emergent
        days_with_flooding = group.groupby(group.index.date)['above_threshold'].any().sum()

        ## Metric 4a: Total time emergent
        hours_above_threshold = group['above_threshold'].sum()

        ## Metric 2a: Duration of Events and Ponding Depths
        event_durations = []
        ponding_depths = []

        event_start_indices = group.index[group['event_start']].tolist()
        event_start_indices.append(group.index[-1] + pd.Timedelta('1H'))  # Add an end marker for the last event

        for i in range(len(event_start_indices) - 1):
            event_period = group.loc[event_start_indices[i]:event_start_indices[i + 1] - pd.Timedelta('1H')]
            event_durations.append(len(event_period))
            ponding_depths.append(event_period['flood_depth'].max())

        ## Seasonal metrics initialization
        winter_flooding = 0
        winter_day_flood = 0
        winter_night_flood = 0
        winter_both_flood = 0
        summer_flooding = 0
        summer_day_flood = 0
        summer_night_flood = 0
        summer_both_flood = 0

        # Calculate winter flooding
        winter_period = group[(group.index.month >= 10) | (group.index.month <= 3)]
        if not winter_period.empty:
            winter_flood_days = winter_period.groupby(winter_period.index.date)['above_threshold'].any()
            winter_flooding = winter_flood_days.sum()

            # Determine day/night/both flooding for winter period
            for day, has_flooding in winter_flood_days.items():
                if has_flooding:
                    day_data = winter_period.loc[str(day)]
                    day_flood = day_data.loc[(day_data.index.hour >= 12) & (day_data.index.hour < 24), 'above_threshold'].any()
                    night_flood = day_data.loc[(day_data.index.hour >= 0) & (day_data.index.hour < 12), 'above_threshold'].any()
                    if day_flood and night_flood:
                        winter_both_flood += 1
                    elif day_flood:
                        winter_day_flood += 1
                    elif night_flood:
                        winter_night_flood += 1

        # Calculate summer flooding
        summer_period = group[(group.index.month >= 4) & (group.index.month <= 9)]
        if not summer_period.empty:
            summer_flood_days = summer_period.groupby(summer_period.index.date)['above_threshold'].any()
            summer_flooding = summer_flood_days.sum()

            # Determine day/night/both flooding for summer period
            for day, has_flooding in summer_flood_days.items():
                if has_flooding:
                    day_data = summer_period.loc[str(day)]
                    day_flood = day_data.loc[(day_data.index.hour >= 12) & (day_data.index.hour < 24), 'above_threshold'].any()
                    night_flood = day_data.loc[(day_data.index.hour >= 0) & (day_data.index.hour < 12), 'above_threshold'].any()
                    if day_flood and night_flood:
                        summer_both_flood += 1
                    elif day_flood:
                        summer_day_flood += 1
                    elif night_flood:
                        summer_night_flood += 1

        ## Save Realization Results
        realization_results.append({
            'year': year,
            'ponding_depths': ponding_depths,
            'days_with_flooding': days_with_flooding,
            'hours_above_threshold': hours_above_threshold,
            'num_events': num_events,
            'event_durations': event_durations,
        })

        realization_seasonal_results.append({
            'year': year,
            'winter_flooding': winter_flooding,
            'winter_day_flood': winter_day_flood,
            'winter_night_flood': winter_night_flood,
            'winter_both_flood': winter_both_flood,
            'summer_flooding': summer_flooding,
            'summer_day_flood': summer_day_flood,
            'summer_night_flood': summer_night_flood,
            'summer_both_flood': summer_both_flood
        })

    full_results_int.append(pd.DataFrame(realization_results))
    seasonal_results.append(pd.DataFrame(realization_seasonal_results))

## Compute metrics from the full results by year
## Compute the minimum, maximum, 25th, 50th, and 75th percentiles

# Ensure all final results are numeric
for df in full_results_int:
    df.apply(pd.to_numeric, errors='coerce')

def safe_percentile(x, q):
    if len(x) > 0:
        return np.percentile(x, q) 
    else:
        return np.nan

## Concatenate all realizations into a single DataFrame
data_int = pd.concat(full_results_int)

## Compute ponding depth metrics
ponding_depths_int = data_int.groupby('year')['ponding_depths'].apply(list).reset_index()
ponding_depths_int.columns = ['year', 'ponding_depths_list']
ponding_depths_int['combined_ponding_depths'] = ponding_depths_int['ponding_depths_list'].apply(lambda x: list(np.concatenate(x)))
ponding_depths_int['percentile_1'] = ponding_depths_int['combined_ponding_depths'].apply(lambda x: safe_percentile(x, 1))
ponding_depths_int['percentile_25'] = ponding_depths_int['combined_ponding_depths'].apply(lambda x: safe_percentile(x, 25))
ponding_depths_int['percentile_50'] = ponding_depths_int['combined_ponding_depths'].apply(lambda x: safe_percentile(x, 50))
ponding_depths_int['percentile_75'] = ponding_depths_int['combined_ponding_depths'].apply(lambda x: safe_percentile(x, 75))
ponding_depths_int['percentile_99'] = ponding_depths_int['combined_ponding_depths'].apply(lambda x: safe_percentile(x, 99))
ponding_depths_int.drop(ponding_depths_int.tail(1).index, inplace=True)

## Compute event duration metrics
event_durations_int = data_int.groupby('year')['event_durations'].apply(list).reset_index()
event_durations_int.columns = ['year', 'event_durations_list']
event_durations_int['combined_event_durations'] = event_durations_int['event_durations_list'].apply(lambda x: list(np.concatenate(x)))
event_durations_int['percentile_1'] = event_durations_int['combined_event_durations'].apply(lambda x: safe_percentile(x, 1))
event_durations_int['percentile_25'] = event_durations_int['combined_event_durations'].apply(lambda x: safe_percentile(x, 25))
event_durations_int['percentile_50'] = event_durations_int['combined_event_durations'].apply(lambda x: safe_percentile(x, 50))
event_durations_int['percentile_75'] = event_durations_int['combined_event_durations'].apply(lambda x: safe_percentile(x, 75))
event_durations_int['percentile_99'] = event_durations_int['combined_event_durations'].apply(lambda x: safe_percentile(x, 99))
event_durations_int.drop(event_durations_int.tail(1).index, inplace=True)

## Compute number of days emergent metrics
days_emergent_int = data_int.groupby('year')['days_with_flooding'].apply(list).reset_index()
days_emergent_int['mean'] = days_emergent_int['days_with_flooding'].apply(np.mean)
days_emergent_int['std'] = days_emergent_int['days_with_flooding'].apply(np.std)
## If mean is < 1, set mean and std to 0
days_emergent_int.loc[days_emergent_int['mean'] < 1, 'mean'] = 0
days_emergent_int.loc[days_emergent_int['mean'] < 1, 'std'] = 0
## Cap std at the mean
days_emergent_int.loc[days_emergent_int['std'] > days_emergent_int['mean'], 'std'] = days_emergent_int['mean']
days_emergent_int.drop(days_emergent_int.tail(1).index, inplace=True)

## Compute total hours emergent metrics
hours_emergent_int = data_int.groupby('year')['hours_above_threshold'].apply(list).reset_index()
hours_emergent_int['mean'] = hours_emergent_int['hours_above_threshold'].apply(np.mean)
hours_emergent_int['std'] = hours_emergent_int['hours_above_threshold'].apply(np.std)
## If mean is < 1, set mean and std to 0
hours_emergent_int.loc[hours_emergent_int['mean'] < 1, 'mean'] = 0
hours_emergent_int.loc[hours_emergent_int['mean'] < 1, 'std'] = 0
## Cap std at the mean
hours_emergent_int.loc[hours_emergent_int['std'] > hours_emergent_int['mean'], 'std'] = hours_emergent_int['mean']
hours_emergent_int.drop(hours_emergent_int.tail(1).index, inplace=True)

combined_seasonal_results = pd.concat(seasonal_results).groupby('year').agg(
    mean_winter_flooding=('winter_flooding', 'mean'),
    std_winter_flooding=('winter_flooding', 'std'),
    mean_winter_day_flood=('winter_day_flood', 'mean'),
    std_winter_day_flood=('winter_day_flood', 'std'),
    mean_winter_night_flood=('winter_night_flood', 'mean'),
    std_winter_night_flood=('winter_night_flood', 'std'),
    mean_winter_both_flood=('winter_both_flood', 'mean'),
    std_winter_both_flood=('winter_both_flood', 'std'),
    mean_summer_flooding=('summer_flooding', 'mean'),
    std_summer_flooding=('summer_flooding', 'std'),
    mean_summer_day_flood=('summer_day_flood', 'mean'),
    std_summer_day_flood=('summer_day_flood', 'std'),
    mean_summer_night_flood=('summer_night_flood', 'mean'),
    std_summer_night_flood=('summer_night_flood', 'std'),
    mean_summer_both_flood=('summer_both_flood', 'mean'),
    std_summer_both_flood=('summer_both_flood', 'std')
).reset_index()

# Drop the last row
combined_seasonal_results.drop(combined_seasonal_results.tail(1).index, inplace=True)

# Ensure that all columns are numeric
combined_seasonal_results = combined_seasonal_results.apply(pd.to_numeric, errors='coerce')

## Define capped_std to ensure standard deviation bars do not extend below 0
def capped_seasonal_std(std_column, mean_column):
    # Calculate capped values where std deviation does not make the bar fall below zero
    capped_seasonal_values = np.where(
        (combined_seasonal_results[mean_column] - combined_seasonal_results[std_column]) < 0,
        combined_seasonal_results[mean_column],
        combined_seasonal_results[std_column]
    )
    return capped_seasonal_values

# Cap standard deviations where needed
for column in ['std_winter_flooding', 'std_winter_day_flood', 
               'std_winter_night_flood', 'std_winter_both_flood', 'std_summer_flooding', 
               'std_summer_day_flood', 'std_summer_night_flood', 'std_summer_both_flood']:
    combined_seasonal_results[column] = capped_seasonal_std(column, column.replace('std', 'mean'))

## Plot Flood Metrics including Seasonal Metrics - Intermediate SLR Scenario

boxplot_depth = [
    [
        ponding_depths_int['percentile_1'][i],
        ponding_depths_int['percentile_25'][i],
        ponding_depths_int['percentile_50'][i],
        ponding_depths_int['percentile_75'][i],
        ponding_depths_int['percentile_99'][i]
    ]
    for i in range(len(ponding_depths_int['year']))
]

boxplot_duration = [
    [
        event_durations_int['percentile_1'][i],
        event_durations_int['percentile_25'][i],
        event_durations_int['percentile_50'][i],
        event_durations_int['percentile_75'][i],
        event_durations_int['percentile_99'][i]
    ]
    for i in range(len(event_durations_int['year']))
]

fontsize = 14
fig, ax = plt.subplots(5, 1, figsize=(12, 14), sharex=True)

# ## OPTION 1: Plot Mean Flood Depth
# ax[0].bar(ponding_depths_int['year'], ponding_depths_int['mean_flood_depth'], yerr=ponding_depths_int['std_flood_depth'], color='blue', capsize=3)
# ax[0].set_ylabel('Mean Flood\nDepth (m)', fontsize=fontsize)

# ## OPTION 2: Box & Whisker plot for ponding_depths_int['flood_depth_min', 'flood_depth_25th', 'flood_depth_50th', 'flood_depth_75th', 'flood_depth_max']
# Create custom boxplots using the summary stats
for i, (min_val, q1, median, q3, max_val) in enumerate(boxplot_depth):
    ax[0].plot([ponding_depths_int['year'][i], ponding_depths_int['year'][i]], [min_val, max_val], color='black')  # Whiskers
    ax[0].plot([ponding_depths_int['year'][i], ponding_depths_int['year'][i]], [q1, q3], color='black', linewidth=7)  # Interquartile range (box)
    ax[0].plot([ponding_depths_int['year'][i], ponding_depths_int['year'][i]], [q1, q3], color='white', linewidth=5)  # Interquartile range (fill)
    ax[0].plot(ponding_depths_int['year'][i], median, 'k_', markersize=7)  # Median line
ax[0].set_yticks(np.arange(0, 0.51, 0.1))
ax[0].set_ylabel('Ponding\nDepth (m)', fontsize=fontsize)
## Plot horizontal line at 0.25m and put text on left side, top of line plot 'Safe Limit for Emergency Vehicles'
ax[0].axhline(y=0.25, color='r', linestyle='--')
ax[0].text(2020.5, 0.27, 'Safe Limit for\nEmergency Vehicles', color='k', fontsize=fontsize-2)

# ax[0].set_title('D0038\nIntermediate Sea Level Rise Scenario\nGroundwater Emergence Events', fontsize=fontsize+2)
ax[0].set_title('Encanto Ave\nIntermediate Sea Level Rise Scenario\nGroundwater Emergence Events', fontsize=fontsize+2)

# ## OPTION 1: Plot the average hours per event
# [ax[1].bar(event_durations_int['year'], event_durations_int['mean_avg_hours_per_event'], yerr=event_durations_int['std_avg_hours_per_event'], color='blue', capsize=3)
# [ax[1].set_ylabel('Avg Flood\nDuration (hr)', fontsize=fontsize)

## OPTION 2: Box & Whisker plot for event_durations_int['duration_min', 'duration_25th', 'duration_50th', 'duration_75th', 'duration_max']
# Create custom boxplots using the summary stats
for i, (min_val, q1, median, q3, max_val) in enumerate(boxplot_duration):
    ax[1].plot([event_durations_int['year'][i], event_durations_int['year'][i]], [min_val, max_val], color='black')  # Whiskers
    ax[1].plot([event_durations_int['year'][i], event_durations_int['year'][i]], [q1, q3], color='black', linewidth=7)  # Interquartile range (box)
    ax[1].plot([event_durations_int['year'][i], event_durations_int['year'][i]], [q1, q3], color='white', linewidth=5)  # Interquartile range (fill)
    ax[1].plot(event_durations_int['year'][i], median, 'k_', markersize=7)  # Median line
ax[1].set_ylabel('Duration of\nEvents (hr)', fontsize=fontsize)
## OPTION 2b: boxplot using boxplot matplotlib function (need all data for boxplot to compute stats on)
# bp = ax[1].boxplot(boxplot_data, positions=event_durations_int['year'], widths=0.6)
# Customize the boxplot appearance
# for element in ['boxes', 'whiskers','means', 'medians', 'caps']:
#     plt.setp(bp[element], color='black')
# plt.setp(bp['fliers'], marker='o', markersize=3, markerfacecolor='red')
ax[1].set_yticks(np.arange(0, 16, 5))

# Plot the number of days with flooding
ax[2].bar(days_emergent_int['year'], days_emergent_int['mean'], yerr=days_emergent_int['std'], color='mediumseagreen', capsize=3)
ax[2].set_ylabel('# Days\nEmergent', fontsize=fontsize)
ax[2].text(2016.5, 354, 'Daily', color='k', fontsize=fontsize-2)
ax[2].set_ylim([0, 365])

# Plot the total hours above threshold
ax[3].bar(hours_emergent_int['year'], hours_emergent_int['mean'], yerr=hours_emergent_int['std'], color='mediumseagreen', capsize=3)
ax[3].set_ylabel('Total Time\nEmergent (hr)', fontsize=fontsize)
## Make y-axis plot ticks and labels every 1000
ax[3].set_yticks(np.arange(0, 3001, 1000))

# Stacked bar plot for seasonal day/night/both flooding
years = combined_seasonal_results['year']
summer_day = combined_seasonal_results['mean_summer_day_flood']
summer_night = combined_seasonal_results['mean_summer_night_flood']
summer_both = combined_seasonal_results['mean_summer_both_flood']
winter_day = combined_seasonal_results['mean_winter_day_flood']
winter_night = combined_seasonal_results['mean_winter_night_flood']
winter_both = combined_seasonal_results['mean_winter_both_flood']

# ## OPTION 1: Plot Day/Night/Both (units = # days)
# ax[4].bar(years, summer_day, label='Summer Day', color='lightblue')
# ax[4].bar(years, summer_night, bottom=summer_day, label='Summer Night', color='skyblue')
# ax[4].bar(years, summer_both, bottom=summer_day + summer_night, label='Summer Both', color='deepskyblue')
# ax[4].bar(years, winter_day, bottom=summer_day + summer_night + summer_both, label='Winter Day', color='lightcoral')
# ax[4].bar(years, winter_night, bottom=summer_day + summer_night + summer_both + winter_day, label='Winter Night', color='indianred')
# ax[4].bar(years, winter_both, bottom=summer_day + summer_night + summer_both + winter_day + winter_night, label='Winter Both', color='firebrick')

## OPTION 2: Plot Day/Night (units = # events)
# ax[4].bar(years, winter_night + winter_both, label='Winter Night', color='indigo')
# ax[4].bar(years, winter_day + winter_both, bottom=winter_night + winter_both, label='Winter Day', color='plum')
# ax[4].bar(years, summer_night + summer_both, bottom=winter_night + winter_day + 2*winter_both, label='Summer Night', color='royalblue')
# ax[4].bar(years, summer_day + summer_both, bottom=winter_night + winter_day + 2*winter_both + summer_night + summer_both, label='Summer Day', color='deepskyblue')
ax[4].bar(years, winter_night + winter_both, label='Winter Night', color='indigo')
ax[4].bar(years, winter_day + winter_both, bottom=winter_night + winter_both, label='Winter Day', color='mediumorchid')
ax[4].bar(years, summer_night + summer_both, bottom=winter_night + winter_day + 2*winter_both, label='Summer Night', color='mediumblue')
ax[4].bar(years, summer_day + summer_both, bottom=winter_night + winter_day + 2*winter_both + summer_night + summer_both, label='Summer Day', color='deepskyblue')

# Set labels and title for stacked bar plot
ax[4].set_ylabel('Emergence Events\nBy Season & TOD', fontsize=fontsize)
handles, labels = ax[4].get_legend_handles_labels()
ax[4].legend(handles[::-1], labels[::-1], loc='upper left', ncol=2, fontsize=fontsize-2)

# Set common x-axis label
ax[4].set_xlabel('Year', fontsize=fontsize)

# Set tick label font size
for a in ax:
    a.tick_params(axis='both', which='major', labelsize=fontsize-2)
    a.xaxis.grid(True, which='major')

## Restrict the x-axis
ax[0].set_xlim([2020, 2100])

plt.tight_layout()
plt.show()
#%% Flood metrics: Cortez - Intermediate Scenario

# Initialize results storage
full_results_int = []

seasonal_results = []

## Intermediate Scenario
for realization in gwt_int.columns:
    gwt_int_flooding = pd.DataFrame(gwt_int[realization])
    gwt_int_flooding.columns = ['gwt']
    gwt_int_flooding['year'] = gwt_int_flooding.index.year
    gwt_int_flooding['above_threshold'] = gwt_int_flooding['gwt'] > roadelevation_cortez
    gwt_int_flooding['flood_depth'] = gwt_int_flooding['gwt'] - roadelevation_cortez
    gwt_int_flooding['flood_depth'] = gwt_int_flooding['flood_depth'].apply(lambda x: x if x > 0 else np.NaN)
    ## Drop nans
    gwt_int_flooding = gwt_int_flooding.dropna()

    realization_results = []
    realization_seasonal_results = []

    for year, group in gwt_int_flooding.groupby('year'):
        group['event_start'] = (group.index.to_series().diff() != pd.Timedelta('1H'))
        # group['event_start'] = (group.index.to_series().diff().shift(-1) == pd.Timedelta('1H'))
        num_events = group['event_start'].sum()

        ## Metric 3a: Number of days emergent
        days_with_flooding = group.groupby(group.index.date)['above_threshold'].any().sum()

        ## Metric 4a: Total time emergent
        hours_above_threshold = group['above_threshold'].sum()

        ## Metric 2a: Duration of Events and Ponding Depths
        event_durations = []
        ponding_depths = []

        event_start_indices = group.index[group['event_start']].tolist()
        event_start_indices.append(group.index[-1] + pd.Timedelta('1H'))  # Add an end marker for the last event

        for i in range(len(event_start_indices) - 1):
            event_period = group.loc[event_start_indices[i]:event_start_indices[i + 1] - pd.Timedelta('1H')]
            event_durations.append(len(event_period))
            ponding_depths.append(event_period['flood_depth'].max())

        ## Seasonal metrics initialization
        winter_flooding = 0
        winter_day_flood = 0
        winter_night_flood = 0
        winter_both_flood = 0
        summer_flooding = 0
        summer_day_flood = 0
        summer_night_flood = 0
        summer_both_flood = 0

        # Calculate winter flooding
        winter_period = group[(group.index.month >= 10) | (group.index.month <= 3)]
        if not winter_period.empty:
            winter_flood_days = winter_period.groupby(winter_period.index.date)['above_threshold'].any()
            winter_flooding = winter_flood_days.sum()

            # Determine day/night/both flooding for winter period
            for day, has_flooding in winter_flood_days.items():
                if has_flooding:
                    day_data = winter_period.loc[str(day)]
                    day_flood = day_data.loc[(day_data.index.hour >= 12) & (day_data.index.hour < 24), 'above_threshold'].any()
                    night_flood = day_data.loc[(day_data.index.hour >= 0) & (day_data.index.hour < 12), 'above_threshold'].any()
                    if day_flood and night_flood:
                        winter_both_flood += 1
                    elif day_flood:
                        winter_day_flood += 1
                    elif night_flood:
                        winter_night_flood += 1

        # Calculate summer flooding
        summer_period = group[(group.index.month >= 4) & (group.index.month <= 9)]
        if not summer_period.empty:
            summer_flood_days = summer_period.groupby(summer_period.index.date)['above_threshold'].any()
            summer_flooding = summer_flood_days.sum()

            # Determine day/night/both flooding for summer period
            for day, has_flooding in summer_flood_days.items():
                if has_flooding:
                    day_data = summer_period.loc[str(day)]
                    day_flood = day_data.loc[(day_data.index.hour >= 12) & (day_data.index.hour < 24), 'above_threshold'].any()
                    night_flood = day_data.loc[(day_data.index.hour >= 0) & (day_data.index.hour < 12), 'above_threshold'].any()
                    if day_flood and night_flood:
                        summer_both_flood += 1
                    elif day_flood:
                        summer_day_flood += 1
                    elif night_flood:
                        summer_night_flood += 1

        ## Save Realization Results
        realization_results.append({
            'year': year,
            'ponding_depths': ponding_depths,
            'days_with_flooding': days_with_flooding,
            'hours_above_threshold': hours_above_threshold,
            'num_events': num_events,
            'event_durations': event_durations,
        })

        realization_seasonal_results.append({
            'year': year,
            'winter_flooding': winter_flooding,
            'winter_day_flood': winter_day_flood,
            'winter_night_flood': winter_night_flood,
            'winter_both_flood': winter_both_flood,
            'summer_flooding': summer_flooding,
            'summer_day_flood': summer_day_flood,
            'summer_night_flood': summer_night_flood,
            'summer_both_flood': summer_both_flood
        })

    full_results_int.append(pd.DataFrame(realization_results))
    seasonal_results.append(pd.DataFrame(realization_seasonal_results))

# Ensure all final results are numeric
for df in full_results_int:
    df.apply(pd.to_numeric, errors='coerce')

def safe_percentile(x, q):
    if len(x) > 0:
        return np.percentile(x, q) 
    else:
        return np.nan

## Concatenate all realizations into a single DataFrame
data_int = pd.concat(full_results_int)

## Compute ponding depth metrics
ponding_depths_int = data_int.groupby('year')['ponding_depths'].apply(list).reset_index()
ponding_depths_int.columns = ['year', 'ponding_depths_list']
ponding_depths_int['combined_ponding_depths'] = ponding_depths_int['ponding_depths_list'].apply(lambda x: list(np.concatenate(x)))
ponding_depths_int['percentile_1'] = ponding_depths_int['combined_ponding_depths'].apply(lambda x: safe_percentile(x, 1))
ponding_depths_int['percentile_25'] = ponding_depths_int['combined_ponding_depths'].apply(lambda x: safe_percentile(x, 25))
ponding_depths_int['percentile_50'] = ponding_depths_int['combined_ponding_depths'].apply(lambda x: safe_percentile(x, 50))
ponding_depths_int['percentile_75'] = ponding_depths_int['combined_ponding_depths'].apply(lambda x: safe_percentile(x, 75))
ponding_depths_int['percentile_99'] = ponding_depths_int['combined_ponding_depths'].apply(lambda x: safe_percentile(x, 99))
ponding_depths_int.drop(ponding_depths_int.tail(1).index, inplace=True)

## Compute event duration metrics
event_durations_int = data_int.groupby('year')['event_durations'].apply(list).reset_index()
event_durations_int.columns = ['year', 'event_durations_list']
event_durations_int['combined_event_durations'] = event_durations_int['event_durations_list'].apply(lambda x: list(np.concatenate(x)))
event_durations_int['percentile_1'] = event_durations_int['combined_event_durations'].apply(lambda x: safe_percentile(x, 1))
event_durations_int['percentile_25'] = event_durations_int['combined_event_durations'].apply(lambda x: safe_percentile(x, 25))
event_durations_int['percentile_50'] = event_durations_int['combined_event_durations'].apply(lambda x: safe_percentile(x, 50))
event_durations_int['percentile_75'] = event_durations_int['combined_event_durations'].apply(lambda x: safe_percentile(x, 75))
event_durations_int['percentile_99'] = event_durations_int['combined_event_durations'].apply(lambda x: safe_percentile(x, 99))
event_durations_int.drop(event_durations_int.tail(1).index, inplace=True)

## Compute number of days emergent metrics
days_emergent_int = data_int.groupby('year')['days_with_flooding'].apply(list).reset_index()
days_emergent_int['mean'] = days_emergent_int['days_with_flooding'].apply(np.mean)
days_emergent_int['std'] = days_emergent_int['days_with_flooding'].apply(np.std)
## If mean is < 1, set mean and std to 0
days_emergent_int.loc[days_emergent_int['mean'] < 1, 'mean'] = 0
days_emergent_int.loc[days_emergent_int['mean'] < 1, 'std'] = 0
## Cap std at the mean
days_emergent_int.loc[days_emergent_int['std'] > days_emergent_int['mean'], 'std'] = days_emergent_int['mean']
days_emergent_int.drop(days_emergent_int.tail(1).index, inplace=True)

## Compute total hours emergent metrics
hours_emergent_int = data_int.groupby('year')['hours_above_threshold'].apply(list).reset_index()
hours_emergent_int['mean'] = hours_emergent_int['hours_above_threshold'].apply(np.mean)
hours_emergent_int['std'] = hours_emergent_int['hours_above_threshold'].apply(np.std)
## If mean is < 1, set mean and std to 0
hours_emergent_int.loc[hours_emergent_int['mean'] < 1, 'mean'] = 0
hours_emergent_int.loc[hours_emergent_int['mean'] < 1, 'std'] = 0
## Cap std at the mean
hours_emergent_int.loc[hours_emergent_int['std'] > hours_emergent_int['mean'], 'std'] = hours_emergent_int['mean']
hours_emergent_int.drop(hours_emergent_int.tail(1).index, inplace=True)

combined_seasonal_results = pd.concat(seasonal_results).groupby('year').agg(
    mean_winter_flooding=('winter_flooding', 'mean'),
    std_winter_flooding=('winter_flooding', 'std'),
    mean_winter_day_flood=('winter_day_flood', 'mean'),
    std_winter_day_flood=('winter_day_flood', 'std'),
    mean_winter_night_flood=('winter_night_flood', 'mean'),
    std_winter_night_flood=('winter_night_flood', 'std'),
    mean_winter_both_flood=('winter_both_flood', 'mean'),
    std_winter_both_flood=('winter_both_flood', 'std'),
    mean_summer_flooding=('summer_flooding', 'mean'),
    std_summer_flooding=('summer_flooding', 'std'),
    mean_summer_day_flood=('summer_day_flood', 'mean'),
    std_summer_day_flood=('summer_day_flood', 'std'),
    mean_summer_night_flood=('summer_night_flood', 'mean'),
    std_summer_night_flood=('summer_night_flood', 'std'),
    mean_summer_both_flood=('summer_both_flood', 'mean'),
    std_summer_both_flood=('summer_both_flood', 'std')
).reset_index()

# Drop the last row
combined_seasonal_results.drop(combined_seasonal_results.tail(1).index, inplace=True)

# Ensure that all columns are numeric
combined_seasonal_results = combined_seasonal_results.apply(pd.to_numeric, errors='coerce')

## Define capped_std to ensure standard deviation bars do not extend below 0
def capped_seasonal_std(std_column, mean_column):
    # Calculate capped values where std deviation does not make the bar fall below zero
    capped_seasonal_values = np.where(
        (combined_seasonal_results[mean_column] - combined_seasonal_results[std_column]) < 0,
        combined_seasonal_results[mean_column],
        combined_seasonal_results[std_column]
    )
    return capped_seasonal_values

# Cap standard deviations where needed
for column in ['std_winter_flooding', 'std_winter_day_flood', 
               'std_winter_night_flood', 'std_winter_both_flood', 'std_summer_flooding', 
               'std_summer_day_flood', 'std_summer_night_flood', 'std_summer_both_flood']:
    combined_seasonal_results[column] = capped_seasonal_std(column, column.replace('std', 'mean'))

## Plot Flood Metrics including Seasonal Metrics - Intermediate SLR Scenario

boxplot_depth = [
    [
        ponding_depths_int['percentile_1'][i],
        ponding_depths_int['percentile_25'][i],
        ponding_depths_int['percentile_50'][i],
        ponding_depths_int['percentile_75'][i],
        ponding_depths_int['percentile_99'][i]
    ]
    for i in range(len(ponding_depths_int['year']))
]

boxplot_duration = [
    [
        event_durations_int['percentile_1'][i],
        event_durations_int['percentile_25'][i],
        event_durations_int['percentile_50'][i],
        event_durations_int['percentile_75'][i],
        event_durations_int['percentile_99'][i]
    ]
    for i in range(len(event_durations_int['year']))
]

fontsize = 14
fig, ax = plt.subplots(5, 1, figsize=(12, 14), sharex=True)

# Create custom boxplots using the summary stats
for i, (min_val, q1, median, q3, max_val) in enumerate(boxplot_depth):
    ax[0].plot([ponding_depths_int['year'][i], ponding_depths_int['year'][i]], [min_val, max_val], color='black')  # Whiskers
    ax[0].plot([ponding_depths_int['year'][i], ponding_depths_int['year'][i]], [q1, q3], color='black', linewidth=7)  # Interquartile range (box)
    ax[0].plot([ponding_depths_int['year'][i], ponding_depths_int['year'][i]], [q1, q3], color='white', linewidth=5)  # Interquartile range (fill)
    ax[0].plot(ponding_depths_int['year'][i], median, 'k_', markersize=7)  # Median line
ax[0].set_yticks(np.arange(0, 0.41, 0.1))
ax[0].set_ylabel('Ponding\nDepth (m)', fontsize=fontsize)
## Plot horizontal line at 0.25m and put text on left side, top of line plot 'Safe Limit for Emergency Vehicles'
ax[0].axhline(y=0.25, color='r', linestyle='--')
ax[0].text(2020.5, 0.27, 'Safe Limit for\nEmergency Vehicles', color='k', fontsize=fontsize-2)

# ax[0].set_title('D0038\nIntermediate Sea Level Rise Scenario\nGroundwater Emergence Events', fontsize=fontsize+2)
ax[0].set_title('Cortez Ave\nIntermediate Sea Level Rise Scenario\nGroundwater Emergence Events', fontsize=fontsize+2)

# Create custom boxplots using the summary stats
for i, (min_val, q1, median, q3, max_val) in enumerate(boxplot_duration):
    ax[1].plot([event_durations_int['year'][i], event_durations_int['year'][i]], [min_val, max_val], color='black')  # Whiskers
    ax[1].plot([event_durations_int['year'][i], event_durations_int['year'][i]], [q1, q3], color='black', linewidth=7)  # Interquartile range (box)
    ax[1].plot([event_durations_int['year'][i], event_durations_int['year'][i]], [q1, q3], color='white', linewidth=5)  # Interquartile range (fill)
    ax[1].plot(event_durations_int['year'][i], median, 'k_', markersize=7)  # Median line
ax[1].set_ylabel('Duration of\nEvents (hr)', fontsize=fontsize)
ax[1].set_yticks(np.arange(0, 11, 5))

# Plot the number of days with flooding
ax[2].bar(days_emergent_int['year'], days_emergent_int['mean'], yerr=days_emergent_int['std'], color='mediumseagreen', capsize=3)
ax[2].set_ylabel('# Days\nEmergent', fontsize=fontsize)
ax[2].text(2016.5, 354, 'Daily', color='k', fontsize=fontsize-2)
ax[2].set_ylim([0, 365])

# Plot the total hours above threshold
ax[3].bar(hours_emergent_int['year'], hours_emergent_int['mean'], yerr=hours_emergent_int['std'], color='mediumseagreen', capsize=3)
ax[3].set_ylabel('Total Time\nEmergent (hr)', fontsize=fontsize)
## Make y-axis plot ticks and labels every 1000
ax[3].set_yticks(np.arange(0, 2001, 1000))

# Stacked bar plot for seasonal day/night/both flooding
years = combined_seasonal_results['year']
summer_day = combined_seasonal_results['mean_summer_day_flood']
summer_night = combined_seasonal_results['mean_summer_night_flood']
summer_both = combined_seasonal_results['mean_summer_both_flood']
winter_day = combined_seasonal_results['mean_winter_day_flood']
winter_night = combined_seasonal_results['mean_winter_night_flood']
winter_both = combined_seasonal_results['mean_winter_both_flood']


ax[4].bar(years, winter_night + winter_both, label='Winter Night', color='indigo')
ax[4].bar(years, winter_day + winter_both, bottom=winter_night + winter_both, label='Winter Day', color='mediumorchid')
ax[4].bar(years, summer_night + summer_both, bottom=winter_night + winter_day + 2*winter_both, label='Summer Night', color='mediumblue')
ax[4].bar(years, summer_day + summer_both, bottom=winter_night + winter_day + 2*winter_both + summer_night + summer_both, label='Summer Day', color='deepskyblue')

# Set labels and title for stacked bar plot
ax[4].set_ylabel('Emergence Events\nBy Season & TOD', fontsize=fontsize)
handles, labels = ax[4].get_legend_handles_labels()
ax[4].legend(handles[::-1], labels[::-1], loc='upper left', ncol=2, fontsize=fontsize-2)

# Set common x-axis label
ax[4].set_xlabel('Year', fontsize=fontsize)

# Set tick label font size
for a in ax:
    a.tick_params(axis='both', which='major', labelsize=fontsize-2)
    a.xaxis.grid(True, which='major')

## Restrict the x-axis
ax[0].set_xlim([2020, 2100])

plt.tight_layout()
plt.show()
#%% Flood metrics: Descanso - Intermediate Scenario

# Initialize results storage
full_results_int = []

seasonal_results = []

## Intermediate Scenario
for realization in gwt_int.columns:
    gwt_int_flooding = pd.DataFrame(gwt_int[realization])
    gwt_int_flooding.columns = ['gwt']
    gwt_int_flooding['year'] = gwt_int_flooding.index.year
    gwt_int_flooding['above_threshold'] = gwt_int_flooding['gwt'] > roadelevation_descanso
    gwt_int_flooding['flood_depth'] = gwt_int_flooding['gwt'] - roadelevation_descanso
    gwt_int_flooding['flood_depth'] = gwt_int_flooding['flood_depth'].apply(lambda x: x if x > 0 else np.NaN)
    ## Drop nans
    gwt_int_flooding = gwt_int_flooding.dropna()

    realization_results = []
    realization_seasonal_results = []

    for year, group in gwt_int_flooding.groupby('year'):
        group['event_start'] = (group.index.to_series().diff() != pd.Timedelta('1H'))
        # group['event_start'] = (group.index.to_series().diff().shift(-1) == pd.Timedelta('1H'))
        num_events = group['event_start'].sum()

        ## Metric 3a: Number of days emergent
        days_with_flooding = group.groupby(group.index.date)['above_threshold'].any().sum()

        ## Metric 4a: Total time emergent
        hours_above_threshold = group['above_threshold'].sum()

        ## Metric 2a: Duration of Events and Ponding Depths
        event_durations = []
        ponding_depths = []

        event_start_indices = group.index[group['event_start']].tolist()
        event_start_indices.append(group.index[-1] + pd.Timedelta('1H'))  # Add an end marker for the last event

        for i in range(len(event_start_indices) - 1):
            event_period = group.loc[event_start_indices[i]:event_start_indices[i + 1] - pd.Timedelta('1H')]
            event_durations.append(len(event_period))
            ponding_depths.append(event_period['flood_depth'].max())

        ## Seasonal metrics initialization
        winter_flooding = 0
        winter_day_flood = 0
        winter_night_flood = 0
        winter_both_flood = 0
        summer_flooding = 0
        summer_day_flood = 0
        summer_night_flood = 0
        summer_both_flood = 0

        # Calculate winter flooding
        winter_period = group[(group.index.month >= 10) | (group.index.month <= 3)]
        if not winter_period.empty:
            winter_flood_days = winter_period.groupby(winter_period.index.date)['above_threshold'].any()
            winter_flooding = winter_flood_days.sum()

            # Determine day/night/both flooding for winter period
            for day, has_flooding in winter_flood_days.items():
                if has_flooding:
                    day_data = winter_period.loc[str(day)]
                    day_flood = day_data.loc[(day_data.index.hour >= 12) & (day_data.index.hour < 24), 'above_threshold'].any()
                    night_flood = day_data.loc[(day_data.index.hour >= 0) & (day_data.index.hour < 12), 'above_threshold'].any()
                    if day_flood and night_flood:
                        winter_both_flood += 1
                    elif day_flood:
                        winter_day_flood += 1
                    elif night_flood:
                        winter_night_flood += 1

        # Calculate summer flooding
        summer_period = group[(group.index.month >= 4) & (group.index.month <= 9)]
        if not summer_period.empty:
            summer_flood_days = summer_period.groupby(summer_period.index.date)['above_threshold'].any()
            summer_flooding = summer_flood_days.sum()

            # Determine day/night/both flooding for summer period
            for day, has_flooding in summer_flood_days.items():
                if has_flooding:
                    day_data = summer_period.loc[str(day)]
                    day_flood = day_data.loc[(day_data.index.hour >= 12) & (day_data.index.hour < 24), 'above_threshold'].any()
                    night_flood = day_data.loc[(day_data.index.hour >= 0) & (day_data.index.hour < 12), 'above_threshold'].any()
                    if day_flood and night_flood:
                        summer_both_flood += 1
                    elif day_flood:
                        summer_day_flood += 1
                    elif night_flood:
                        summer_night_flood += 1

        ## Save Realization Results
        realization_results.append({
            'year': year,
            'ponding_depths': ponding_depths,
            'days_with_flooding': days_with_flooding,
            'hours_above_threshold': hours_above_threshold,
            'num_events': num_events,
            'event_durations': event_durations,
        })

        realization_seasonal_results.append({
            'year': year,
            'winter_flooding': winter_flooding,
            'winter_day_flood': winter_day_flood,
            'winter_night_flood': winter_night_flood,
            'winter_both_flood': winter_both_flood,
            'summer_flooding': summer_flooding,
            'summer_day_flood': summer_day_flood,
            'summer_night_flood': summer_night_flood,
            'summer_both_flood': summer_both_flood
        })

    full_results_int.append(pd.DataFrame(realization_results))
    seasonal_results.append(pd.DataFrame(realization_seasonal_results))

## Compute metrics from the full results by year
## Compute the minimum, maximum, 25th, 50th, and 75th percentiles

# Ensure all final results are numeric
for df in full_results_int:
    df.apply(pd.to_numeric, errors='coerce')

def safe_percentile(x, q):
    if len(x) > 0:
        return np.percentile(x, q) 
    else:
        return np.nan

## Concatenate all realizations into a single DataFrame
data_int = pd.concat(full_results_int)

## Compute ponding depth metrics
ponding_depths_int = data_int.groupby('year')['ponding_depths'].apply(list).reset_index()
ponding_depths_int.columns = ['year', 'ponding_depths_list']
ponding_depths_int['combined_ponding_depths'] = ponding_depths_int['ponding_depths_list'].apply(lambda x: list(np.concatenate(x)))
ponding_depths_int['percentile_1'] = ponding_depths_int['combined_ponding_depths'].apply(lambda x: safe_percentile(x, 1))
ponding_depths_int['percentile_25'] = ponding_depths_int['combined_ponding_depths'].apply(lambda x: safe_percentile(x, 25))
ponding_depths_int['percentile_50'] = ponding_depths_int['combined_ponding_depths'].apply(lambda x: safe_percentile(x, 50))
ponding_depths_int['percentile_75'] = ponding_depths_int['combined_ponding_depths'].apply(lambda x: safe_percentile(x, 75))
ponding_depths_int['percentile_99'] = ponding_depths_int['combined_ponding_depths'].apply(lambda x: safe_percentile(x, 99))
ponding_depths_int.drop(ponding_depths_int.tail(1).index, inplace=True)

## Compute event duration metrics
event_durations_int = data_int.groupby('year')['event_durations'].apply(list).reset_index()
event_durations_int.columns = ['year', 'event_durations_list']
event_durations_int['combined_event_durations'] = event_durations_int['event_durations_list'].apply(lambda x: list(np.concatenate(x)))
event_durations_int['percentile_1'] = event_durations_int['combined_event_durations'].apply(lambda x: safe_percentile(x, 1))
event_durations_int['percentile_25'] = event_durations_int['combined_event_durations'].apply(lambda x: safe_percentile(x, 25))
event_durations_int['percentile_50'] = event_durations_int['combined_event_durations'].apply(lambda x: safe_percentile(x, 50))
event_durations_int['percentile_75'] = event_durations_int['combined_event_durations'].apply(lambda x: safe_percentile(x, 75))
event_durations_int['percentile_99'] = event_durations_int['combined_event_durations'].apply(lambda x: safe_percentile(x, 99))
event_durations_int.drop(event_durations_int.tail(1).index, inplace=True)

## Compute number of days emergent metrics
days_emergent_int = data_int.groupby('year')['days_with_flooding'].apply(list).reset_index()
days_emergent_int['mean'] = days_emergent_int['days_with_flooding'].apply(np.mean)
days_emergent_int['std'] = days_emergent_int['days_with_flooding'].apply(np.std)
## If mean is < 1, set mean and std to 0
days_emergent_int.loc[days_emergent_int['mean'] < 1, 'mean'] = 0
days_emergent_int.loc[days_emergent_int['mean'] < 1, 'std'] = 0
## Cap std at the mean
days_emergent_int.loc[days_emergent_int['std'] > days_emergent_int['mean'], 'std'] = days_emergent_int['mean']
days_emergent_int.drop(days_emergent_int.tail(1).index, inplace=True)

## Compute total hours emergent metrics
hours_emergent_int = data_int.groupby('year')['hours_above_threshold'].apply(list).reset_index()
hours_emergent_int['mean'] = hours_emergent_int['hours_above_threshold'].apply(np.mean)
hours_emergent_int['std'] = hours_emergent_int['hours_above_threshold'].apply(np.std)
## If mean is < 1, set mean and std to 0
hours_emergent_int.loc[hours_emergent_int['mean'] < 1, 'mean'] = 0
hours_emergent_int.loc[hours_emergent_int['mean'] < 1, 'std'] = 0
## Cap std at the mean
hours_emergent_int.loc[hours_emergent_int['std'] > hours_emergent_int['mean'], 'std'] = hours_emergent_int['mean']
hours_emergent_int.drop(hours_emergent_int.tail(1).index, inplace=True)

combined_seasonal_results = pd.concat(seasonal_results).groupby('year').agg(
    mean_winter_flooding=('winter_flooding', 'mean'),
    std_winter_flooding=('winter_flooding', 'std'),
    mean_winter_day_flood=('winter_day_flood', 'mean'),
    std_winter_day_flood=('winter_day_flood', 'std'),
    mean_winter_night_flood=('winter_night_flood', 'mean'),
    std_winter_night_flood=('winter_night_flood', 'std'),
    mean_winter_both_flood=('winter_both_flood', 'mean'),
    std_winter_both_flood=('winter_both_flood', 'std'),
    mean_summer_flooding=('summer_flooding', 'mean'),
    std_summer_flooding=('summer_flooding', 'std'),
    mean_summer_day_flood=('summer_day_flood', 'mean'),
    std_summer_day_flood=('summer_day_flood', 'std'),
    mean_summer_night_flood=('summer_night_flood', 'mean'),
    std_summer_night_flood=('summer_night_flood', 'std'),
    mean_summer_both_flood=('summer_both_flood', 'mean'),
    std_summer_both_flood=('summer_both_flood', 'std')
).reset_index()

# Drop the last row
combined_seasonal_results.drop(combined_seasonal_results.tail(1).index, inplace=True)

# Ensure that all columns are numeric
combined_seasonal_results = combined_seasonal_results.apply(pd.to_numeric, errors='coerce')

## Define capped_std to ensure standard deviation bars do not extend below 0
def capped_seasonal_std(std_column, mean_column):
    # Calculate capped values where std deviation does not make the bar fall below zero
    capped_seasonal_values = np.where(
        (combined_seasonal_results[mean_column] - combined_seasonal_results[std_column]) < 0,
        combined_seasonal_results[mean_column],
        combined_seasonal_results[std_column]
    )
    return capped_seasonal_values

# Cap standard deviations where needed
for column in ['std_winter_flooding', 'std_winter_day_flood', 
               'std_winter_night_flood', 'std_winter_both_flood', 'std_summer_flooding', 
               'std_summer_day_flood', 'std_summer_night_flood', 'std_summer_both_flood']:
    combined_seasonal_results[column] = capped_seasonal_std(column, column.replace('std', 'mean'))

## Plot Flood Metrics including Seasonal Metrics - Intermediate SLR Scenario

boxplot_depth = [
    [
        ponding_depths_int['percentile_1'][i],
        ponding_depths_int['percentile_25'][i],
        ponding_depths_int['percentile_50'][i],
        ponding_depths_int['percentile_75'][i],
        ponding_depths_int['percentile_99'][i]
    ]
    for i in range(len(ponding_depths_int['year']))
]

boxplot_duration = [
    [
        event_durations_int['percentile_1'][i],
        event_durations_int['percentile_25'][i],
        event_durations_int['percentile_50'][i],
        event_durations_int['percentile_75'][i],
        event_durations_int['percentile_99'][i]
    ]
    for i in range(len(event_durations_int['year']))
]
#%%
fontsize = 14
fig, ax = plt.subplots(5, 1, figsize=(12, 14), sharex=True)

# ## OPTION 1: Plot Mean Flood Depth
# ax[0].bar(ponding_depths_int['year'], ponding_depths_int['mean_flood_depth'], yerr=ponding_depths_int['std_flood_depth'], color='blue', capsize=3)
# ax[0].set_ylabel('Mean Flood\nDepth (m)', fontsize=fontsize)

# ## OPTION 2: Box & Whisker plot for ponding_depths_int['flood_depth_min', 'flood_depth_25th', 'flood_depth_50th', 'flood_depth_75th', 'flood_depth_max']
# Create custom boxplots using the summary stats
for i, (min_val, q1, median, q3, max_val) in enumerate(boxplot_depth):
    ax[0].plot([ponding_depths_int['year'][i], ponding_depths_int['year'][i]], [min_val, max_val], color='black')  # Whiskers
    ax[0].plot([ponding_depths_int['year'][i], ponding_depths_int['year'][i]], [q1, q3], color='black', linewidth=7)  # Interquartile range (box)
    ax[0].plot([ponding_depths_int['year'][i], ponding_depths_int['year'][i]], [q1, q3], color='white', linewidth=5)  # Interquartile range (fill)
    ax[0].plot(ponding_depths_int['year'][i], median, 'k_', markersize=7)  # Median line
ax[0].set_yticks(np.arange(0, 0.51, 0.1))
ax[0].set_ylabel('Ponding\nDepth (m)', fontsize=fontsize)
## Plot horizontal line at 0.25m and put text on left side, top of line plot 'Safe Limit for Emergency Vehicles'
ax[0].axhline(y=0.25, color='r', linestyle='--')
ax[0].text(2020.5, 0.27, 'Safe Limit for\nEmergency Vehicles', color='k', fontsize=fontsize-2)

# ax[0].set_title('D0038\nIntermediate Sea Level Rise Scenario\nGroundwater Emergence Events', fontsize=fontsize+2)
ax[0].set_title('Descanso Ave\nIntermediate Sea Level Rise Scenario\nGroundwater Emergence Events', fontsize=fontsize+2)

# ## OPTION 1: Plot the average hours per event
# [ax[1].bar(event_durations_int['year'], event_durations_int['mean_avg_hours_per_event'], yerr=event_durations_int['std_avg_hours_per_event'], color='blue', capsize=3)
# [ax[1].set_ylabel('Avg Flood\nDuration (hr)', fontsize=fontsize)

## OPTION 2: Box & Whisker plot for event_durations_int['duration_min', 'duration_25th', 'duration_50th', 'duration_75th', 'duration_max']
# Create custom boxplots using the summary stats
for i, (min_val, q1, median, q3, max_val) in enumerate(boxplot_duration):
    ax[1].plot([event_durations_int['year'][i], event_durations_int['year'][i]], [min_val, max_val], color='black')  # Whiskers
    ax[1].plot([event_durations_int['year'][i], event_durations_int['year'][i]], [q1, q3], color='black', linewidth=7)  # Interquartile range (box)
    ax[1].plot([event_durations_int['year'][i], event_durations_int['year'][i]], [q1, q3], color='white', linewidth=5)  # Interquartile range (fill)
    ax[1].plot(event_durations_int['year'][i], median, 'k_', markersize=7)  # Median line
ax[1].set_ylabel('Duration of\nEvents (hr)', fontsize=fontsize)
## OPTION 2b: boxplot using boxplot matplotlib function (need all data for boxplot to compute stats on)
# bp = ax[1].boxplot(boxplot_data, positions=event_durations_int['year'], widths=0.6)
# Customize the boxplot appearance
# for element in ['boxes', 'whiskers','means', 'medians', 'caps']:
#     plt.setp(bp[element], color='black')
# plt.setp(bp['fliers'], marker='o', markersize=3, markerfacecolor='red')
ax[1].set_yticks(np.arange(0, 16, 5))

# Plot the number of days with flooding
ax[2].bar(days_emergent_int['year'], days_emergent_int['mean'], yerr=days_emergent_int['std'], color='mediumseagreen', capsize=3)
ax[2].set_ylabel('# Days\nEmergent', fontsize=fontsize)
# ax[2].text(2016.5, 354, 'Daily', color='k', fontsize=fontsize-2)
ax[2].set_ylim([0, 200])

# Plot the total hours above threshold
ax[3].bar(hours_emergent_int['year'], hours_emergent_int['mean'], yerr=hours_emergent_int['std'], color='mediumseagreen', capsize=3)
ax[3].set_ylabel('Total Time\nEmergent (hr)', fontsize=fontsize)
## Make y-axis plot ticks and labels every 1000
ax[3].set_yticks(np.arange(0, 801, 200))

# Stacked bar plot for seasonal day/night/both flooding
years = combined_seasonal_results['year']
summer_day = combined_seasonal_results['mean_summer_day_flood']
summer_night = combined_seasonal_results['mean_summer_night_flood']
summer_both = combined_seasonal_results['mean_summer_both_flood']
winter_day = combined_seasonal_results['mean_winter_day_flood']
winter_night = combined_seasonal_results['mean_winter_night_flood']
winter_both = combined_seasonal_results['mean_winter_both_flood']

# ## OPTION 1: Plot Day/Night/Both (units = # days)
# ax[4].bar(years, summer_day, label='Summer Day', color='lightblue')
# ax[4].bar(years, summer_night, bottom=summer_day, label='Summer Night', color='skyblue')
# ax[4].bar(years, summer_both, bottom=summer_day + summer_night, label='Summer Both', color='deepskyblue')
# ax[4].bar(years, winter_day, bottom=summer_day + summer_night + summer_both, label='Winter Day', color='lightcoral')
# ax[4].bar(years, winter_night, bottom=summer_day + summer_night + summer_both + winter_day, label='Winter Night', color='indianred')
# ax[4].bar(years, winter_both, bottom=summer_day + summer_night + summer_both + winter_day + winter_night, label='Winter Both', color='firebrick')

## OPTION 2: Plot Day/Night (units = # events)
# ax[4].bar(years, winter_night + winter_both, label='Winter Night', color='indigo')
# ax[4].bar(years, winter_day + winter_both, bottom=winter_night + winter_both, label='Winter Day', color='plum')
# ax[4].bar(years, summer_night + summer_both, bottom=winter_night + winter_day + 2*winter_both, label='Summer Night', color='royalblue')
# ax[4].bar(years, summer_day + summer_both, bottom=winter_night + winter_day + 2*winter_both + summer_night + summer_both, label='Summer Day', color='deepskyblue')
ax[4].bar(years, winter_night + winter_both, label='Winter Night', color='indigo')
ax[4].bar(years, winter_day + winter_both, bottom=winter_night + winter_both, label='Winter Day', color='mediumorchid')
ax[4].bar(years, summer_night + summer_both, bottom=winter_night + winter_day + 2*winter_both, label='Summer Night', color='mediumblue')
ax[4].bar(years, summer_day + summer_both, bottom=winter_night + winter_day + 2*winter_both + summer_night + summer_both, label='Summer Day', color='deepskyblue')
# Add error bars for standard deviation
# ax[4].errorbar(years, winter_night + winter_both, 
#                yerr=combined_seasonal_results['std_winter_night_flood']+combined_seasonal_results['std_winter_both_flood'], fmt='none', ecolor='plum', capsize=2)
# ax[4].errorbar(years, winter_night + winter_day + 2*winter_both, 
#                yerr=combined_seasonal_results['std_winter_day_flood']+combined_seasonal_results['std_winter_both_flood'], fmt='none', ecolor='plum', capsize=2)
# ax[4].errorbar(years, winter_night + winter_day + 2*winter_both + summer_night + summer_both, 
#                yerr=combined_seasonal_results['std_summer_night_flood']+combined_seasonal_results['std_summer_both_flood'], fmt='none', ecolor='lightcyan', capsize=2)
# ax[4].errorbar(years, winter_night + winter_day + 2*winter_both + summer_night + summer_day + 2*summer_both, 
#                yerr=combined_seasonal_results['std_summer_day_flood']+combined_seasonal_results['std_summer_both_flood'], fmt='none', ecolor='blue', capsize=2)

# Set labels and title for stacked bar plot
ax[4].set_ylabel('Emergence Events\nBy Season & TOD', fontsize=fontsize)
handles, labels = ax[4].get_legend_handles_labels()
ax[4].legend(handles[::-1], labels[::-1], loc='upper left', ncol=2, fontsize=fontsize-2)

# Set common x-axis label
ax[4].set_xlabel('Year', fontsize=fontsize)

# Set tick label font size
for a in ax:
    a.tick_params(axis='both', which='major', labelsize=fontsize-2)
    a.xaxis.grid(True, which='major')

## Restrict the x-axis
ax[0].set_xlim([2020, 2100])

plt.tight_layout()
plt.show()
#%% Determine time indices of non-NaN values of gwt_NAVD88 for 25%, 50%, and 75% of the data (model calibration sensitivity testing)
# first_25 = int(len(gwt_NAVD88.dropna()) * 0.25)
# first_25_time = gwt_NAVD88.dropna().index[first_25]

# ## Determine time indices of last 25% of non-NaN values of gwt_NAVD88
# last_25 = int(len(gwt_NAVD88.dropna()) * 0.75)
# last_25_time = gwt_NAVD88.dropna().index[last_25]

# ## Determine time indices of 50% of non-NaN values of gwt_NAVD88
# mid_50 = int(len(gwt_NAVD88.dropna()) * 0.50)
# mid_50_time = gwt_NAVD88.dropna().index[mid_50]

# print(f'First 25% of non-NaN values: {first_25_time}')
# print(f'Last 25% of non-NaN values: {last_25_time}')
# print(f'Mid 50% of non-NaN values: {mid_50_time}')