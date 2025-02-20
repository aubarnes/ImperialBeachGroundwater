"""
Create the Hindcast and Projected Groundwater Table
MOP 38: S Seacoast Drive

Hindcast: 2000-01-01 to 2024-09-30
Projected: 2024-10-01 to 2100-01-01

Hindcast Groundwater Table Elevation = predicted GW tide + Pastas(observed LJ Tide Gauge non-tidal residual + observed precip - observed ETo)
200xProjected Groundwater Table Elevation = annual mean sea level + predicted GW tide + 25xPastas([NTR] + 8x(projected precip - projected ETo)
"""
#%% Imports & Define Directory
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib qt
import pastas as ps
import matplotlib.dates as mdates
import matplotlib.patheffects as pe
from scipy.stats import norm

## Values derived from 2016 DEM
roadelevation_sseacoast = 1.9 # meters (1.846 according to GPS at well location)
roadelevation_encanto = 2.08 # meters (approx low from DEM)
roadelevation_descanso = 2.27 # meters (approx low from DEM)
roadelevation_cortez = 2.13 # meters (approx low from DEM)
roadelevation_palm = 2.0 # meters (approx low from DEM)
roadelevation_carnation = 1.5 # meters (approx low from DEM)

## Observations
path_to_gwt_observations = '/Users/austinbarnes/Documents/UCSD SIO/IB Groundwater/ImperialBeach/data/seacoast_20240514_1124_QC.h5'
path_to_ljtide_full = '/Users/austinbarnes/Documents/UCSD SIO/IB Groundwater/ImperialBeach/data/LJ_tide_data/ljtide_1924.h5'

## Used in Hincast & Projections
path_to_ljtide_2100 = '/Users/austinbarnes/Documents/UCSD SIO/IB Groundwater/ImperialBeach/data/LJ_tide_data/ljtide_2100.h5'
path_to_ETo_2100 = '/Users/austinbarnes/Documents/UCSD SIO/IB Groundwater/ImperialBeach/data/ETo/ETo_spatial/ETo_2100.h5'
path_to_precip = '/Users/austinbarnes/Documents/UCSD SIO/IB Groundwater/ImperialBeach/data/precip_IB/precip_2100.h5'

## Hindcast Only
path_to_ljntr_hindcast = '/Users/austinbarnes/Documents/UCSD SIO/IB Groundwater/ImperialBeach/data/LJ_tide_data/ljntr_hindcast.h5'

## Projections Only
path_to_slr_interp = '/Users/austinbarnes/Documents/UCSD SIO/IB Groundwater/ImperialBeach/data/slr/slr_interp.pkl'
path_to_ljntr_ensemble = '/Users/austinbarnes/Documents/UCSD SIO/IB Groundwater/ImperialBeach/data/LJ_tide_data/ljntr_ensemble.h5'
path_to_cmip6_ensemble = '/Users/austinbarnes/Documents/UCSD SIO/IB Groundwater/ImperialBeach/data/cmip6_ensemble.pkl'

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

## Plot the Full calibration period, hindcast, and residuals
%matplotlib
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True, gridspec_kw={'height_ratios': [2,1]})
ax1.plot(hindcast_2003, label='Hindcast')
ax1.plot(data4comparison, label='Seacoast Data',alpha=0.5)
## Shade in the calibration period
ax1.axvspan(pd.to_datetime(calib_start), pd.to_datetime(calib_end), color='gray', alpha=0.2)
ax1.text(pd.to_datetime(calib_start), 0.45, 'Calibration Period', verticalalignment='top', horizontalalignment='left', color='k', fontsize=12)
ax1.set_ylabel('NAVD88 (m)', fontsize=12)
ax1.set_title('Seacoast Full Model vs. Data')
ax1.legend()
## Second subplot
ax2.scatter((data4comparison - hindcast_2003).index,data4comparison - hindcast_2003,color='k', label='Residual')
# ax3 = ax2.twinx()
# ax3.plot(R2)
ax2.grid(which='both', ls='dotted')
ax2.set_ylabel('(m)', fontsize=12)
ax2.set_title('Model Residuals')
## Restrict x-axis
plt.show()
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

# stress_ntr_low = slr_low_50p_2100
# stress_ntr_low = stress_ntr_low.tz_localize(None)
# ml.stressmodels['ljntr'] = ps.StressModel(
#     stress=stress_ntr_low,
#     # rfunc=ps.Gamma(),
#     # rfunc=ps.Exponential(),
#     rfunc = ps.One(),
#     name="ljntr_low",
#     settings="waterlevel")
# model_slronly_low = ml.simulate(warmup=warmupdays, tmin="2024-10-01", tmax="2100-10-01", freq=ml.settings["freq"]).reindex(model_low.index)
# ntr_slronly_low = ml.get_contribution("ljntr", tmin="2024-10-01", tmax="2100-10-01")
# offset_low = stress_ntr_low - ntr_slronly_low
# offset_low = offset_low - offset_low.min()
# offset_low = offset_low.drop(offset_low.index[-1])

# stress_ntr_intlow = slr_intlow_50p_2100
# stress_ntr_intlow = stress_ntr_intlow.tz_localize(None)
# ml.stressmodels['ljntr'] = ps.StressModel(
#     stress=stress_ntr_intlow,
#     # rfunc=ps.Gamma(),
#     # rfunc=ps.Exponential(),
#     rfunc = ps.One(),
#     name="ljntr_intlow",
#     settings="waterlevel")
# model_slronly_intlow = ml.simulate(warmup=warmupdays, tmin="2024-10-01", tmax="2100-10-01", freq=ml.settings["freq"]).reindex(model_intlow.index)
# ntr_slronly_intlow = ml.get_contribution("ljntr", tmin="2024-10-01", tmax="2100-10-01")
# offset_intlow = stress_ntr_intlow - ntr_slronly_intlow
# offset_intlow = offset_intlow - offset_intlow.min()
# offset_intlow = offset_intlow.drop(offset_intlow.index[-1])

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

# stress_ntr_inthigh = slr_inthigh_50p_2100
# stress_ntr_inthigh = stress_ntr_inthigh.tz_localize(None)
# ml.stressmodels['ljntr'] = ps.StressModel(
#     stress=stress_ntr_inthigh,
#     # rfunc=ps.Gamma(),
#     # rfunc=ps.Exponential(),
#     rfunc = ps.One(),
#     name="ljntr_inthigh",
#     settings="waterlevel")
# model_slronly_inthigh = ml.simulate(warmup=warmupdays, tmin="2024-10-01", tmax="2100-10-01", freq=ml.settings["freq"]).reindex(model_inthigh.index)
# ntr_slronly_inthigh = ml.get_contribution("ljntr", tmin="2024-10-01", tmax="2100-10-01")
# offset_inthigh = stress_ntr_inthigh - ntr_slronly_inthigh
# offset_inthigh = offset_inthigh - offset_inthigh.min()
# offset_inthigh = offset_inthigh.drop(offset_inthigh.index[-1])

# stress_ntr_high = slr_high_50p_2100
# stress_ntr_high = stress_ntr_high.tz_localize(None)
# ml.stressmodels['ljntr'] = ps.StressModel(
#     stress=stress_ntr_high,
#     # rfunc=ps.Gamma(),
#     # rfunc=ps.Exponential(),
#     rfunc = ps.One(),
#     name="ljntr_high",
#     settings="waterlevel")
# model_slronly_high = ml.simulate(warmup=warmupdays, tmin="2024-10-01", tmax="2100-10-01", freq=ml.settings["freq"]).reindex(model_high.index)
# ntr_slronly_high = ml.get_contribution("ljntr", tmin="2024-10-01", tmax="2100-10-01")
# offset_high = stress_ntr_high - ntr_slronly_high
# offset_high = offset_high - offset_high.min()
# offset_high = offset_high.drop(offset_high.index[-1])

# stress_ntr_low_17p = slr_low_17p_2100
# stress_ntr_low_17p = stress_ntr_low_17p.tz_localize(None)
# ml.stressmodels['ljntr'] = ps.StressModel(
#     stress=stress_ntr_low_17p,
#     rfunc = ps.One(),
#     name="ljntr_low_17p",
#     settings="waterlevel")
# model_slronly_low_17p = ml.simulate(warmup=warmupdays, tmin="2024-10-01", tmax="2100-10-01", freq=ml.settings["freq"]).reindex(model_low_17p.index)
# ntr_slronly_low_17p = ml.get_contribution("ljntr", tmin="2024-10-01", tmax="2100-10-01")
# offset_low_17p = stress_ntr_low_17p - ntr_slronly_low_17p
# offset_low_17p = offset_low_17p - offset_low_17p.min()
# offset_low_17p = offset_low_17p.drop(offset_low_17p.index[-1])

# stress_ntr_low_83p = slr_low_83p_2100
# stress_ntr_low_83p = stress_ntr_low_83p.tz_localize(None)
# ml.stressmodels['ljntr'] = ps.StressModel(
#     stress=stress_ntr_low_83p,
#     rfunc = ps.One(),
#     name="ljntr_low_83p",
#     settings="waterlevel")
# model_slronly_low_83p = ml.simulate(warmup=warmupdays, tmin="2024-10-01", tmax="2100-10-01", freq=ml.settings["freq"]).reindex(model_low_83p.index)
# ntr_slronly_low_83p = ml.get_contribution("ljntr", tmin="2024-10-01", tmax="2100-10-01")
# offset_low_83p = stress_ntr_low_83p - ntr_slronly_low_83p
# offset_low_83p = offset_low_83p - offset_low_83p.min()
# offset_low_83p = offset_low_83p.drop(offset_low_83p.index[-1])

# stress_ntr_intlow_17p = slr_intlow_17p_2100
# stress_ntr_intlow_17p = stress_ntr_intlow_17p.tz_localize(None)
# ml.stressmodels['ljntr'] = ps.StressModel(
#     stress=stress_ntr_intlow_17p,
#     rfunc = ps.One(),
#     name="ljntr_intlow_17p",
#     settings="waterlevel")
# model_slronly_intlow_17p = ml.simulate(warmup=warmupdays, tmin="2024-10-01", tmax="2100-10-01", freq=ml.settings["freq"]).reindex(model_intlow_17p.index)
# ntr_slronly_intlow_17p = ml.get_contribution("ljntr", tmin="2024-10-01", tmax="2100-10-01")
# offset_intlow_17p = stress_ntr_intlow_17p - ntr_slronly_intlow_17p
# offset_intlow_17p = offset_intlow_17p - offset_intlow_17p.min()
# offset_intlow_17p = offset_intlow_17p.drop(offset_intlow_17p.index[-1])

# stress_ntr_intlow_83p = slr_intlow_83p_2100
# stress_ntr_intlow_83p = stress_ntr_intlow_83p.tz_localize(None)
# ml.stressmodels['ljntr'] = ps.StressModel(
#     stress=stress_ntr_intlow_83p,
#     rfunc = ps.One(),
#     name="ljntr_intlow_83p",
#     settings="waterlevel")
# model_slronly_intlow_83p = ml.simulate(warmup=warmupdays, tmin="2024-10-01", tmax="2100-10-01", freq=ml.settings["freq"]).reindex(model_intlow_83p.index)
# ntr_slronly_intlow_83p = ml.get_contribution("ljntr", tmin="2024-10-01", tmax="2100-10-01")
# offset_intlow_83p = stress_ntr_intlow_83p - ntr_slronly_intlow_83p
# offset_intlow_83p = offset_intlow_83p - offset_intlow_83p.min()
# offset_intlow_83p = offset_intlow_83p.drop(offset_intlow_83p.index[-1])

# stress_ntr_int_17p = slr_int_17p_2100
# stress_ntr_int_17p = stress_ntr_int_17p.tz_localize(None)
# ml.stressmodels['ljntr'] = ps.StressModel(
#     stress=stress_ntr_int_17p,
#     rfunc = ps.One(),
#     name="ljntr_int_17p",
#     settings="waterlevel")
# model_slronly_int_17p = ml.simulate(warmup=warmupdays, tmin="2024-10-01", tmax="2100-10-01", freq=ml.settings["freq"]).reindex(model_int_17p.index)
# ntr_slronly_int_17p = ml.get_contribution("ljntr", tmin="2024-10-01", tmax="2100-10-01")
# offset_int_17p = stress_ntr_int_17p - ntr_slronly_int_17p
# offset_int_17p = offset_int_17p - offset_int_17p.min()
# offset_int_17p = offset_int_17p.drop(offset_int_17p.index[-1])

# stress_ntr_int_83p = slr_int_83p_2100
# stress_ntr_int_83p = stress_ntr_int_83p.tz_localize(None)
# ml.stressmodels['ljntr'] = ps.StressModel(
#     stress=stress_ntr_int_83p,
#     rfunc = ps.One(),
#     name="ljntr_int_83p",
#     settings="waterlevel")
# model_slronly_int_83p = ml.simulate(warmup=warmupdays, tmin="2024-10-01", tmax="2100-10-01", freq=ml.settings["freq"]).reindex(model_int_83p.index)
# ntr_slronly_int_83p = ml.get_contribution("ljntr", tmin="2024-10-01", tmax="2100-10-01")
# offset_int_83p = stress_ntr_int_83p - ntr_slronly_int_83p
# offset_int_83p = offset_int_83p - offset_int_83p.min()
# offset_int_83p = offset_int_83p.drop(offset_int_83p.index[-1])

# stress_ntr_inthigh_17p = slr_inthigh_17p_2100
# stress_ntr_inthigh_17p = stress_ntr_inthigh_17p.tz_localize(None)
# ml.stressmodels['ljntr'] = ps.StressModel(
#     stress=stress_ntr_inthigh_17p,
#     rfunc = ps.One(),
#     name="ljntr_inthigh_17p",
#     settings="waterlevel")
# model_slronly_inthigh_17p = ml.simulate(warmup=warmupdays, tmin="2024-10-01", tmax="2100-10-01", freq=ml.settings["freq"]).reindex(model_inthigh_17p.index)
# ntr_slronly_inthigh_17p = ml.get_contribution("ljntr", tmin="2024-10-01", tmax="2100-10-01")
# offset_inthigh_17p = stress_ntr_inthigh_17p - ntr_slronly_inthigh_17p
# offset_inthigh_17p = offset_inthigh_17p - offset_inthigh_17p.min()
# offset_inthigh_17p = offset_inthigh_17p.drop(offset_inthigh_17p.index[-1])

# stress_ntr_inthigh_83p = slr_inthigh_83p_2100
# stress_ntr_inthigh_83p = stress_ntr_inthigh_83p.tz_localize(None)
# ml.stressmodels['ljntr'] = ps.StressModel(
#     stress=stress_ntr_inthigh_83p,
#     rfunc = ps.One(),
#     name="ljntr_inthigh_83p",
#     settings="waterlevel")
# model_slronly_inthigh_83p = ml.simulate(warmup=warmupdays, tmin="2024-10-01", tmax="2100-10-01", freq=ml.settings["freq"]).reindex(model_inthigh_83p.index)
# ntr_slronly_inthigh_83p = ml.get_contribution("ljntr", tmin="2024-10-01", tmax="2100-10-01")
# offset_inthigh_83p = stress_ntr_inthigh_83p - ntr_slronly_inthigh_83p
# offset_inthigh_83p = offset_inthigh_83p - offset_inthigh_83p.min()
# offset_inthigh_83p = offset_inthigh_83p.drop(offset_inthigh_83p.index[-1])

# stress_ntr_high_17p = slr_high_17p_2100
# stress_ntr_high_17p = stress_ntr_high_17p.tz_localize(None)
# ml.stressmodels['ljntr'] = ps.StressModel(
#     stress=stress_ntr_high_17p,
#     rfunc = ps.One(),
#     name="ljntr_high_17p",
#     settings="waterlevel")
# model_slronly_high_17p = ml.simulate(warmup=warmupdays, tmin="2024-10-01", tmax="2100-10-01", freq=ml.settings["freq"]).reindex(model_high_17p.index)
# ntr_slronly_high_17p = ml.get_contribution("ljntr", tmin="2024-10-01", tmax="2100-10-01")
# offset_high_17p = stress_ntr_high_17p - ntr_slronly_high_17p
# offset_high_17p = offset_high_17p - offset_high_17p.min()
# offset_high_17p = offset_high_17p.drop(offset_high_17p.index[-1])

# stress_ntr_high_83p = slr_high_83p_2100
# stress_ntr_high_83p = stress_ntr_high_83p.tz_localize(None)
# ml.stressmodels['ljntr'] = ps.StressModel(
#     stress=stress_ntr_high_83p,
#     rfunc = ps.One(),
#     name="ljntr_high_83p",
#     settings="waterlevel")
# model_slronly_high_83p = ml.simulate(warmup=warmupdays, tmin="2024-10-01", tmax="2100-10-01", freq=ml.settings["freq"]).reindex(model_high_83p.index)
# ntr_slronly_high_83p = ml.get_contribution("ljntr", tmin="2024-10-01", tmax="2100-10-01")
# offset_high_83p = stress_ntr_high_83p - ntr_slronly_high_83p
# offset_high_83p = offset_high_83p - offset_high_83p.min()
# offset_high_83p = offset_high_83p.drop(offset_high_83p.index[-1])

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
        
        # stress_ntr = pd.Series(ntr_low_17p.iloc[:, ntr_num].values, index=pd.to_datetime(ntr_low_17p.index))
        # stress_ntr = stress_ntr.tz_localize(None)
        # ml.stressmodels['ljntr'] = ps.StressModel(
        #     stress=stress_ntr,
        #     rfunc=ps.Exponential(),
        #     name="ljntr_low_17p",
        #     settings="waterlevel")
        # model_low_17p.iloc[:, i] = ml.simulate(tmin="2024-10-01", tmax="2100-10-01", freq=ml.settings["freq"]).reindex(model_low_17p.index)
        # gwt_low_17p.iloc[:, i] = model_low_17p.iloc[:, i] + offset_low_17p

        # stress_ntr = pd.Series(ntr_low.iloc[:, ntr_num].values, index=pd.to_datetime(ntr_low.index))
        # stress_ntr = stress_ntr.tz_localize(None)
        # ml.stressmodels['ljntr'] = ps.StressModel(
        #     stress=stress_ntr,
        #     rfunc=ps.Exponential(),
        #     name="ljntr_low",
        #     settings="waterlevel")
        # model_low.iloc[:, i] = ml.simulate(tmin="2024-10-01", tmax="2100-10-01", freq=ml.settings["freq"]).reindex(model_low.index)
        # gwt_low.iloc[:, i] = model_low.iloc[:, i] + offset_low

        # stress_ntr = pd.Series(ntr_low_83p.iloc[:, ntr_num].values, index=pd.to_datetime(ntr_low_83p.index))
        # stress_ntr = stress_ntr.tz_localize(None)
        # ml.stressmodels['ljntr'] = ps.StressModel(
        #     stress=stress_ntr,
        #     rfunc=ps.Exponential(),
        #     name="ljntr_low_83p",
        #     settings="waterlevel")
        # model_low_83p.iloc[:, i] = ml.simulate(tmin="2024-10-01", tmax="2100-10-01", freq=ml.settings["freq"]).reindex(model_low_83p.index)
        # gwt_low_83p.iloc[:, i] = model_low_83p.iloc[:, i] + offset_low_83p

        # stress_ntr = pd.Series(ntr_intlow_17p.iloc[:, ntr_num].values, index=pd.to_datetime(ntr_intlow_17p.index))
        # stress_ntr = stress_ntr.tz_localize(None)
        # ml.stressmodels['ljntr'] = ps.StressModel(
        #     stress=stress_ntr,
        #     rfunc=ps.Exponential(),
        #     name="ljntr_intlow_17p",
        #     settings="waterlevel")
        # model_intlow_17p.iloc[:, i] = ml.simulate(tmin="2024-10-01", tmax="2100-10-01", freq=ml.settings["freq"]).reindex(model_intlow_17p.index)
        # gwt_intlow_17p.iloc[:, i] = model_intlow_17p.iloc[:, i] + offset_intlow_17p

        # stress_ntr = pd.Series(ntr_intlow.iloc[:, ntr_num].values, index=pd.to_datetime(ntr_intlow.index))
        # stress_ntr = stress_ntr.tz_localize(None)
        # ml.stressmodels['ljntr'] = ps.StressModel(
        #     stress=stress_ntr,
        #     rfunc=ps.Exponential(),
        #     name="ljntr_intlow",
        #     settings="waterlevel")
        # model_intlow.iloc[:, i] = ml.simulate(tmin="2024-10-01", tmax="2100-10-01", freq=ml.settings["freq"]).reindex(model_intlow.index)
        # gwt_intlow.iloc[:, i] = model_intlow.iloc[:, i] + offset_intlow

        # stress_ntr = pd.Series(ntr_intlow_83p.iloc[:, ntr_num].values, index=pd.to_datetime(ntr_intlow_83p.index))
        # stress_ntr = stress_ntr.tz_localize(None)
        # ml.stressmodels['ljntr'] = ps.StressModel(
        #     stress=stress_ntr,
        #     rfunc=ps.Exponential(),
        #     name="ljntr_intlow_83p",
        #     settings="waterlevel")
        # model_intlow_83p.iloc[:, i] = ml.simulate(tmin="2024-10-01", tmax="2100-10-01", freq=ml.settings["freq"]).reindex(model_intlow_83p.index)
        # gwt_intlow_83p.iloc[:, i] = model_intlow_83p.iloc[:, i] + offset_intlow_83p

        # stress_ntr = pd.Series(ntr_int_17p.iloc[:, ntr_num].values, index=pd.to_datetime(ntr_int_17p.index))
        # stress_ntr = stress_ntr.tz_localize(None)
        # ml.stressmodels['ljntr'] = ps.StressModel(
        #     stress=stress_ntr,
        #     rfunc=ps.Exponential(),
        #     name="ljntr_int_17p",
        #     settings="waterlevel")
        # model_int_17p.iloc[:, i] = ml.simulate(warmup=warmupdays, tmin="2024-10-01", tmax="2100-10-01", freq=ml.settings["freq"]).reindex(model_int_17p.index)
        # gwt_int_17p.iloc[:, i] = model_int_17p.iloc[:, i] + offset_int_17p

        stress_ntr = pd.Series(ntr_int.iloc[:, ntr_num].values, index=pd.to_datetime(ntr_int.index))
        stress_ntr = stress_ntr.tz_localize(None)
        ml.stressmodels['ljntr'] = ps.StressModel(
            stress=stress_ntr,
            rfunc=ps.Exponential(),
            name="ljntr_int",
            settings="waterlevel")
        model_int.iloc[:, i] = ml.simulate(warmup=warmupdays, tmin="2024-10-01", tmax="2100-10-01", freq=ml.settings["freq"]).reindex(model_int.index)
        gwt_int.iloc[:, i] = model_int.iloc[:, i] + offset_int

        # stress_ntr = pd.Series(ntr_int_83p.iloc[:, ntr_num].values, index=pd.to_datetime(ntr_int_83p.index))
        # stress_ntr = stress_ntr.tz_localize(None)
        # ml.stressmodels['ljntr'] = ps.StressModel(
        #     stress=stress_ntr,
        #     rfunc=ps.Exponential(),
        #     name="ljntr_int_83p",
        #     settings="waterlevel")
        # model_int_83p.iloc[:, i] = ml.simulate(warmup=warmupdays, tmin="2024-10-01", tmax="2100-10-01", freq=ml.settings["freq"]).reindex(model_int_83p.index)
        # gwt_int_83p.iloc[:, i] = model_int_83p.iloc[:, i] + offset_int_83p

        # stress_ntr = pd.Series(ntr_inthigh_17p.iloc[:, ntr_num].values, index=pd.to_datetime(ntr_inthigh_17p.index))
        # stress_ntr = stress_ntr.tz_localize(None)
        # ml.stressmodels['ljntr'] = ps.StressModel(
        #     stress=stress_ntr,
        #     rfunc=ps.Exponential(),
        #     name="ljntr_inthigh_17p",
        #     settings="waterlevel")
        # model_inthigh_17p.iloc[:, i] = ml.simulate(tmin="2024-10-01", tmax="2100-10-01", freq=ml.settings["freq"]).reindex(model_inthigh_17p.index)
        # gwt_inthigh_17p.iloc[:, i] = model_inthigh_17p.iloc[:, i] + offset_inthigh_17p

        # stress_ntr = pd.Series(ntr_inthigh.iloc[:, ntr_num].values, index=pd.to_datetime(ntr_inthigh.index))
        # stress_ntr = stress_ntr.tz_localize(None)
        # ml.stressmodels['ljntr'] = ps.StressModel(
        #     stress=stress_ntr,
        #     rfunc=ps.Exponential(),
        #     name="ljntr_inthigh",
        #     settings="waterlevel")
        # model_inthigh.iloc[:, i] = ml.simulate(tmin="2024-10-01", tmax="2100-10-01", freq=ml.settings["freq"]).reindex(model_inthigh.index)
        # gwt_inthigh.iloc[:, i] = model_inthigh.iloc[:, i] + offset_inthigh

        # stress_ntr = pd.Series(ntr_inthigh_83p.iloc[:, ntr_num].values, index=pd.to_datetime(ntr_inthigh_83p.index))
        # stress_ntr = stress_ntr.tz_localize(None)
        # ml.stressmodels['ljntr'] = ps.StressModel(
        #     stress=stress_ntr,
        #     rfunc=ps.Exponential(),
        #     name="ljntr_inthigh_83p",
        #     settings="waterlevel")
        # model_inthigh_83p.iloc[:, i] = ml.simulate(tmin="2024-10-01", tmax="2100-10-01", freq=ml.settings["freq"]).reindex(model_inthigh_83p.index)
        # gwt_inthigh_83p.iloc[:, i] = model_inthigh_83p.iloc[:, i] + offset_inthigh_83p

        # stress_ntr = pd.Series(ntr_high_17p.iloc[:, ntr_num].values, index=pd.to_datetime(ntr_high_17p.index))
        # stress_ntr = stress_ntr.tz_localize(None)
        # ml.stressmodels['ljntr'] = ps.StressModel(
        #     stress=stress_ntr,
        #     rfunc=ps.Exponential(),
        #     name="ljntr_high_17p",
        #     settings="waterlevel")
        # model_high_17p.iloc[:, i] = ml.simulate(tmin="2024-10-01", tmax="2100-10-01", freq=ml.settings["freq"]).reindex(model_high_17p.index)
        # gwt_high_17p.iloc[:, i] = model_high_17p.iloc[:, i] + offset_high_17p

        # stress_ntr = pd.Series(ntr_high.iloc[:, ntr_num].values, index=pd.to_datetime(ntr_high.index))
        # stress_ntr = stress_ntr.tz_localize(None)
        # ml.stressmodels['ljntr'] = ps.StressModel(
        #     stress=stress_ntr,
        #     rfunc=ps.Exponential(),
        #     name="ljntr_high",
        #     settings="waterlevel")
        # model_high.iloc[:, i] = ml.simulate(tmin="2024-10-01", tmax="2100-10-01", freq=ml.settings["freq"]).reindex(model_high.index)
        # gwt_high.iloc[:, i] = model_high.iloc[:, i] + offset_high

        # stress_ntr = pd.Series(ntr_high_83p.iloc[:, ntr_num].values, index=pd.to_datetime(ntr_high_83p.index))
        # stress_ntr = stress_ntr.tz_localize(None)
        # ml.stressmodels['ljntr'] = ps.StressModel(
        #     stress=stress_ntr,
        #     rfunc=ps.Exponential(),
        #     name="ljntr_high_83p",
        #     settings="waterlevel")
        # model_high_83p.iloc[:, i] = ml.simulate(tmin="2024-10-01", tmax="2100-10-01", freq=ml.settings["freq"]).reindex(model_high_83p.index)
        # gwt_high_83p.iloc[:, i] = model_high_83p.iloc[:, i] + offset_high_83p

    ## Make all values into floats
    # gwt_low_17p = gwt_low_17p.astype(float)
    # gwt_low = gwt_low.astype(float)
    # gwt_low_83p = gwt_low_83p.astype(float)

    # gwt_intlow_17p = gwt_intlow_17p.astype(float)
    # gwt_intlow = gwt_intlow.astype(float)
    # gwt_intlow_83p = gwt_intlow_83p.astype(float)

    gwt_int_17p = gwt_int_17p.astype(float)
    gwt_int = gwt_int.astype(float)
    gwt_int_83p = gwt_int_83p.astype(float)

    # gwt_inthigh_17p = gwt_inthigh_17p.astype(float)
    # gwt_inthigh = gwt_inthigh.astype(float)
    # gwt_inthigh_83p = gwt_inthigh_83p.astype(float)

    # gwt_high_17p = gwt_high_17p.astype(float)
    # gwt_high = gwt_high.astype(float)
    # gwt_high_83p = gwt_high_83p.astype(float)

#%% Projections: Resample the projected groundwater table to the daily max
gwt_hindcast_dailymax = hindcast_2003.resample('D').max()
gwt_hindcast_dailymax_annualmean = gwt_hindcast_dailymax.resample('Y').mean()
gwt_hindcast_monthlymax_annualmean = hindcast_2003.resample('M').max().resample('Y').mean()
gwt_hindcast_annualmax = hindcast_2003.resample('Y').max()
## Drop last row, incomplete based on <1 year
# gwt_hindcast_dailymax_annualmean.drop(gwt_hindcast_dailymax_annualmean.tail(1).index, inplace=True)

# gwt_low_dailymax = gwt_low.resample('D').max()
# gwt_intlow_dailymax = gwt_intlow.resample('D').max()
gwt_int_dailymax = gwt_int.resample('D').max()
# gwt_inthigh_dailymax = gwt_inthigh.resample('D').max()
# gwt_high_dailymax = gwt_high.resample('D').max()

# ## Ensemble Mean and Standard Deviation
# gwt_low_17p_mean = gwt_low_17p.mean(axis=1, numeric_only=True)
# gwt_low_mean = gwt_low.mean(axis=1, numeric_only=True)
# gwt_low_83p_mean = gwt_low_83p.mean(axis=1, numeric_only=True)
# gwt_low_std = gwt_low.std(axis=1, numeric_only=True)

# gwt_intlow_17p_mean = gwt_intlow_17p.mean(axis=1, numeric_only=True)
# gwt_intlow_mean = gwt_intlow.mean(axis=1, numeric_only=True)
# gwt_intlow_83p_mean = gwt_intlow_83p.mean(axis=1, numeric_only=True)
# gwt_intlow_std = gwt_intlow.std(axis=1, numeric_only=True)

# gwt_int_17p_mean = gwt_int_17p.mean(axis=1, numeric_only=True)
# gwt_int_mean = gwt_int.mean(axis=1, numeric_only=True)
# gwt_int_83p_mean = gwt_int_83p.mean(axis=1, numeric_only=True)
# gwt_int_std = gwt_int.std(axis=1, numeric_only=True)

# gwt_inthigh_17p_mean = gwt_inthigh_17p.mean(axis=1, numeric_only=True)
# gwt_inthigh_mean = gwt_inthigh.mean(axis=1, numeric_only=True)
# gwt_inthigh_83p_mean = gwt_inthigh_83p.mean(axis=1, numeric_only=True)
# gwt_inthigh_std = gwt_inthigh.std(axis=1, numeric_only=True)

# gwt_high_17p_mean = gwt_high_17p.mean(axis=1, numeric_only=True)
# gwt_high_mean = gwt_high.mean(axis=1, numeric_only=True)
# gwt_high_83p_mean = gwt_high_83p.mean(axis=1, numeric_only=True)
# gwt_high_std = gwt_high.std(axis=1, numeric_only=True)

# gwt_low_dailymax_mean = gwt_low_dailymax.mean(axis=1, numeric_only=True)
# gwt_low_dailymax_std = gwt_low_dailymax.std(axis=1, numeric_only=True)
# gwt_intlow_dailymax_mean = gwt_intlow_dailymax.mean(axis=1, numeric_only=True)
# gwt_intlow_dailymax_std = gwt_intlow_dailymax.std(axis=1, numeric_only=True)
gwt_int_dailymax_mean = gwt_int_dailymax.mean(axis=1, numeric_only=True)
gwt_int_dailymax_std = gwt_int_dailymax.std(axis=1, numeric_only=True)
# gwt_inthigh_dailymax_mean = gwt_inthigh_dailymax.mean(axis=1, numeric_only=True)
# gwt_inthigh_dailymax_std = gwt_inthigh_dailymax.std(axis=1, numeric_only=True)
# gwt_high_dailymax_mean = gwt_high_dailymax.mean(axis=1, numeric_only=True)
# gwt_high_dailymax_std = gwt_high_dailymax.std(axis=1, numeric_only=True)

## Annual average of the daily maximum groundwater table ANNUAL MEAN OF DAILY MAXIMUM OF ENSEMBLE MEAN
# gwt_low_dailymax_annualmean = gwt_low_dailymax_mean.resample('Y').mean()
# gwt_intlow_dailymax_annualmean = gwt_intlow_dailymax_mean.resample('Y').mean()
# gwt_int_dailymax_annualmean = gwt_int_dailymax_mean.resample('Y').mean()
# gwt_inthigh_dailymax_annualmean = gwt_inthigh_dailymax_mean.resample('Y').mean()
# gwt_high_dailymax_annualmean = gwt_high_dailymax_mean.resample('Y').mean()

## Annual average of daily maximum groundwater table ANNUAL MEAN OF DAILY MAXIMUM OF EACH ENSEMBLE
gwt_int_dailymax_annualmean = gwt_int_dailymax.resample('Y').mean()

## Annual maximum of the scenario means
# gwt_low_17p_annualmax = gwt_low_17p_mean.resample('Y').max()
# gwt_low_annualmax = gwt_low_mean.resample('Y').max()
# gwt_low_83p_annualmax = gwt_low_83p_mean.resample('Y').max()
# gwt_low_annualmax_std = gwt_low.resample('Y').max().std(axis=1, numeric_only=True)

# gwt_intlow_17p_annualmax = gwt_intlow_17p_mean.resample('Y').max()
# gwt_intlow_annualmax = gwt_intlow_mean.resample('Y').max()
# gwt_intlow_83p_annualmax = gwt_intlow_83p_mean.resample('Y').max()
# gwt_intlow_annualmax_std = gwt_intlow.resample('Y').max().std(axis=1, numeric_only=True)

# gwt_int_17p_annualmax = gwt_int_17p_mean.resample('Y').max()
# gwt_int_annualmax = gwt_int_mean.resample('Y').max()
# gwt_int_83p_annualmax = gwt_int_83p_mean.resample('Y').max()
# gwt_int_annualmax_std = gwt_int.resample('Y').max().std(axis=1, numeric_only=True)

# gwt_inthigh_17p_annualmax = gwt_inthigh_17p_mean.resample('Y').max()
# gwt_inthigh_annualmax = gwt_inthigh_mean.resample('Y').max()
# gwt_inthigh_83p_annualmax = gwt_inthigh_83p_mean.resample('Y').max()
# gwt_inthigh_annualmax_std = gwt_inthigh.resample('Y').max().std(axis=1, numeric_only=True)

# gwt_high_17p_annualmax = gwt_high_17p_mean.resample('Y').max()
# gwt_high_annualmax = gwt_high_mean.resample('Y').max()
# gwt_high_83p_annualmax = gwt_high_83p_mean.resample('Y').max()
# gwt_high_annualmax_std = gwt_high.resample('Y').max().std(axis=1, numeric_only=True)

#%% Uncertainty Estimation: Hindcast & Projections

## Hindcast
## error1 = var(residuals) + error_obs
residuals = data4comparison - hindcast
# error_obs = 0.04**2 ## 0.04 m estimated observational standard deviation
error_obs = 0
error1 = np.nanvar(residuals) + error_obs
twosigma_hindcast = 2*np.sqrt(error1)


## Projections (not including SLR curves)
## Variance over the ensembles at each time point and average of these variances over time
## error2 = mean(var(y_forecast))
error2_dailymax = np.mean(np.var(gwt_int_dailymax, axis = 1))
twosigma_projection = 2*np.sqrt(error1 + error2_dailymax)
# Calculate the 83rd percentile of the normal distribution
p83_ensemble = norm.ppf(0.83, loc=0, scale=twosigma_projection/2)

## SLR Curves: 83rd and 17th percentiles
p83_slr_int = (slr_int_83p_2100 - slr_int_50p_2100).resample('D').mean()
p17_slr_int = (slr_int_50p_2100 - slr_int_17p_2100).resample('D').mean()

## Final 83rd and 17th percentile of Hindcast and Projections
p83_hindcast = gwt_hindcast_dailymax + norm.ppf(0.83, loc=0, scale=twosigma_hindcast/2)
p17_hindcast = gwt_hindcast_dailymax + norm.ppf(0.17, loc=0, scale=twosigma_hindcast/2)

p83_projection = gwt_int_dailymax.mean(axis=1) + p83_ensemble + p83_slr_int
p17_projection = gwt_int_dailymax.mean(axis=1) - p83_ensemble - p17_slr_int
#%% Intermediate Scenario, Daily Maximum
%matplotlib qt
fig, ax = plt.subplots(1,1,figsize=(18,18),sharex=True)

## Hindcast: daily maxima with 83rd and 17th percentiles
ax.plot(gwt_hindcast_dailymax, color='black', label='Hindcast', linewidth=0.3)
ax.fill_between(gwt_hindcast_dailymax.index, p17_hindcast, p83_hindcast, color='black', alpha=0.4, label='17th-83rd Percentiles')

## Projections: daily maxima with 83rd and 17th percentiles
ax.plot(gwt_int_dailymax.median(axis=1), color='purple', linestyle="-", linewidth=0.3, label='50th Percentile Projection')
ax.fill_between(gwt_int_dailymax.median(axis=1).index, p17_projection, p83_projection, color='purple', alpha=0.2, label='17th-83rd Percentiles')

## Add horizontal lines at road elevations
ax.axhline(y=roadelevation_sseacoast, color='red', linestyle='--', label='Intersection Elevations')
# ax.text(pd.to_datetime('2002-01-01'), roadelevation_sseacoast, 'D0038', verticalalignment='bottom', horizontalalignment='left', color='red', fontsize=fontsize-2)
ax.text(pd.to_datetime('2002-01-01'), roadelevation_sseacoast, 'S Seacoast', verticalalignment='bottom', horizontalalignment='left', color='red', fontsize=fontsize-2)
ax.axhline(y=roadelevation_descanso, color = 'seagreen', linestyle='--', label='D0041 Road Elevation')
# ax.text(pd.to_datetime('2002-01-01'), roadelevation_descanso, 'D0041', verticalalignment='bottom', horizontalalignment='left', color='seagreen', fontsize=fontsize-2)
ax.text(pd.to_datetime('2002-01-01'), roadelevation_descanso, 'Descanso', verticalalignment='bottom', horizontalalignment='left', color='seagreen', fontsize=fontsize-2)
ax.axhline(y=roadelevation_encanto, color = 'darkorange', linestyle='--', label='D0043 Road Elevation')
# ax.text(pd.to_datetime('2002-01-01'), roadelevation_encanto, 'D0043', verticalalignment='bottom', horizontalalignment='left', color='darkgreen', fontsize=fontsize-2)
ax.text(pd.to_datetime('2002-01-01'), roadelevation_encanto, 'Encanto', verticalalignment='bottom', horizontalalignment='left', color='darkorange', fontsize=fontsize-2)
ax.axhline(y=roadelevation_cortez, color = 'darkgreen', linestyle='--', label='D0045 Road Elevation')
# ax.text(pd.to_datetime('2002-01-01'), roadelevation_cortez, 'D0045', verticalalignment='bottom', horizontalalignment='left', color='darkorange', fontsize=fontsize-2)
ax.text(pd.to_datetime('2002-01-01'), roadelevation_cortez, 'Cortez', verticalalignment='bottom', horizontalalignment='left', color='darkgreen', fontsize=fontsize-2)
ax.axhline(y=roadelevation_palm, color = 'tomato', linestyle='--', label='D0057 Road Elevation')
# ax.text(pd.to_datetime('2002-01-01'), roadelevation_palm, 'D0057', verticalalignment='bottom', horizontalalignment='left', color='tomato', fontsize=fontsize-2)
ax.text(pd.to_datetime('2002-01-01'), roadelevation_palm, 'Palm', verticalalignment='bottom', horizontalalignment='left', color='tomato', fontsize=fontsize-2)

order = [0, 1, 2, 3, 4]
ax.set_title(f'Seacoast Groundwater Table\nIntermediate SLR Scenario\nDaily Maximum', fontsize=fontsize)
# ax.set_ylim([1.47, 2.6])
# ax.set_yticks(np.arange(1.5, 2.61, 0.1))


handles, labels = ax.get_legend_handles_labels()
ax.legend([handles[idx] for idx in order], [labels[idx] for idx in order], loc='lower right', fontsize=fontsize-2)
ax.set_ylabel('Groundwater Table Elevation\n(m, NAVD88)', fontsize=fontsize)

ax.tick_params(axis='both', which='major', labelsize=fontsize)

ax.set_xlim([pd.Timestamp('2000-01-01'), pd.Timestamp('2100-01-01')])
ax.set_xlabel('Year', fontsize=fontsize)

ax.grid(True)

plt.tight_layout()
plt.show()
#%%