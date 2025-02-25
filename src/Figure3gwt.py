"""
Figure 3: Groundwater table measurements
3 subplots:
1) Hourly average groundwater table measurements
For Seacoast, include cosine-Lanczos filtered data
La Jolla tide gauge hourly average
Include cosine-Lanczos filtered data
2) Hourly average groundwater table measurements relative to local land surface
3) Daily Precipitation from TJRTLMET station

November 2024
Austin Barnes
"""
#%% Imports
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib
import numpy as np
import matplotlib.dates as mdates
import matplotlib.patheffects as pe

## Groundwater table
path_to_gwt_seacoast = '../data/seacoast_20240514_1124_QC.h5'
path_to_NTR_seacoast = '../data/seacoast_NTR.h5'
path_to_gwt_fifthgrove = '../data/fifthgrove_20240514_1124_QC.h5'
path_to_gwt_pubworks = '../data/pubworks_20240514_1435_QC.h5'
path_to_gwt_eleventhebony = '../data/eleventhebony_20240514_1124_QC.h5'

## Tide gauge
path_to_ljtide = '../data/external/ljtide.h5'
path_to_ntr_dailymax_hindcast = '../data/external/ljtide_dailymax_hindcast.h5'

## Precipitation
path_to_IB_precip = '../data/external/IB_precip.h5'

#%% Load Groundwater table data
## Load Seacoast
gwt_seacoast = pd.read_hdf(path_to_gwt_seacoast)
## Load Seacoast NTR
NTR_seacoast = pd.read_hdf(path_to_NTR_seacoast)
NTR_seacoast = NTR_seacoast.resample('D').mean()
## Load Fifthgrove
gwt_fifthgrove = pd.read_hdf(path_to_gwt_fifthgrove)
## Load Pubworks
gwt_pubworks = pd.read_hdf(path_to_gwt_pubworks)
## Load Eleventhebony
gwt_eleventhebony = pd.read_hdf(path_to_gwt_eleventhebony)

# Create new dataframe with hourly averages
seacoast_hourly = gwt_seacoast.resample('1H', on='Timestamps').mean().reset_index()
fifthgrove_hourly = gwt_fifthgrove.resample('1H', on='Timestamps').mean().reset_index()
pubworks_hourly = gwt_pubworks.resample('1H', on='Timestamps').mean().reset_index()
eleventhebony_hourly = gwt_eleventhebony.resample('1H', on='Timestamps').mean().reset_index()

#%% Load the La Jolla tide gauge data
lj_tide = pd.read_hdf(path_to_ljtide, key='lj_tide')
lj_tide = lj_tide.resample('H').mean()
## Load the NTR hindcast data
ntr_dailymax_hindcast = pd.read_hdf(path_to_ntr_dailymax_hindcast)
# ntr_dailymax_hindcast = ntr_dailymax_hindcast + lj_tide.mean()

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

#%% Create Figure, 3 subplots
## 1) Hourly average groundwater table measurements
## For Seacoast, include cosine-Lanczos filtered data
## Include La Jolla tide gauge hourly average and cosine-Lanczos filtered data
## 2) Hourly average groundwater table measurements relative to local land surface
## 3) Daily Precipitation from TJRTLMET station

ntr_offset = NTR_seacoast.mean() - seacoast_hourly['landsurf'].mean()

%matplotlib qt

fontsize = 18
alpha = 1.0

fig, axs = plt.subplots(3, 1, sharex=True, figsize=(15, 10), gridspec_kw={'height_ratios': [3, 2, 1]})

## 1) Hourly average groundwater table measurements
## For Seacoast, include cosine-Lanczos filtered data
## Include La Jolla tide gauge hourly average and cosine-Lanczos filtered data
axs[0].plot(lj_tide, label='Tide Gauge', linestyle='-', alpha=alpha-0.6, linewidth = 0.3, color='k')

## Horizontal lines for Befus et al. (2020) LMSL & MHHW for Seacoast
axs[0].axhline(y=0.77, label='Befus et al. (2020) Seacoast LMSL', color='yellow', linestyle='dotted', alpha=1.0, linewidth=2, path_effects=[pe.Stroke(linewidth=4, foreground='black'), pe.Normal()])
axs[0].axhline(y=1.56, label='Befus et al. (2020) Seacoast MHHW', color='yellow', linestyle='dashdot', alpha=1.0, linewidth=2, path_effects=[pe.Stroke(linewidth=4, foreground='black'), pe.Normal()])

axs[0].plot(seacoast_hourly['Timestamps'], seacoast_hourly['NAVD88'], label='Well 1', alpha=alpha-0.4, color='C0')
axs[0].plot(fifthgrove_hourly['Timestamps'], fifthgrove_hourly['NAVD88'], label='Well 2', linewidth = 3, alpha=alpha, color='C1')
axs[0].plot(pubworks_hourly['Timestamps'], pubworks_hourly['NAVD88'], label='Well 3', linewidth = 3, alpha=alpha, color='C2')
axs[0].plot(eleventhebony_hourly['Timestamps'], eleventhebony_hourly['NAVD88'], label='Well 4', linewidth = 3, alpha=alpha, color='blueviolet')

# axs[0].plot(ntr_dailymax_hindcast, label='Sea Level (non-tidal)', linestyle='--', linewidth = 2, alpha=alpha, color='black')
# axs[0].plot(NTR_seacoast, label='Well 1 (non-tidal)', linestyle = '--', linewidth = 2, alpha=alpha, color='blue')
axs[0].scatter(ntr_dailymax_hindcast.index, ntr_dailymax_hindcast, label='Sea Level (non-tidal)', marker='.', s=10, alpha=alpha, color='black', zorder=10)
axs[0].scatter(NTR_seacoast.index, NTR_seacoast, label='Well 1 (non-tidal)', marker='.', s=10, alpha=alpha, color='blue', zorder=10)

# axs[0].legend(fontsize = fontsize - 4)
# axs[0].legend(fontsize = fontsize - 4, loc='lower left', bbox_to_anchor=(0.01, 0.3), fancybox=True, shadow=True, ncol=5)
handles, labels = axs[0].get_legend_handles_labels()
order = [0, 7, 2, 1, 3, 8, 4, 5, 6]
legend = axs[0].legend([handles[idx] for idx in order], [labels[idx] for idx in order], loc='lower center',ncol=5, markerscale = 4,fontsize=fontsize-4)
legend.legend_handles[0].set_linewidth(3)
legend.legend_handles[4].set_linewidth(3)

axs[0].set_ylabel('NAVD88 (m)', fontsize = fontsize)
axs[0].tick_params(axis='y', labelsize=fontsize-4)
axs[0].set_title('Hourly Groundwater Table Levels', fontsize = fontsize)

axs[0].set_ylim(-1.5, 2.25)

## Add bold 'a' to the top left of subplot 1
axs[0].text(-0.06, 1.05, 'a', transform=axs[0].transAxes, fontsize=fontsize, fontweight='bold', va='top', ha='right')

## 2) Hourly average groundwater table measurements relative to local land surface
axs[1].plot(seacoast_hourly['Timestamps'], seacoast_hourly['landsurf'], label='Well 1', linewidth = 0.3, alpha=alpha-0.2, color='C0')
# axs[1].plot(NTR_seacoast-ntr_offset, label='Well 1 (non-tidal)', linestyle = '--', linewidth = 3, alpha=alpha, color='blue')
axs[1].scatter(NTR_seacoast.index, NTR_seacoast-ntr_offset, label='Well 1 (non-tidal)', marker='.', s=10, alpha=alpha, color='blue', zorder=10)
axs[1].plot(fifthgrove_hourly['Timestamps'], fifthgrove_hourly['landsurf'], label='Well 2', linewidth = 3, alpha=alpha, color='C1')
axs[1].plot(pubworks_hourly['Timestamps'], pubworks_hourly['landsurf'], label='Well 3', linewidth = 3, alpha=alpha, color='C2')
axs[1].plot(eleventhebony_hourly['Timestamps'], eleventhebony_hourly['landsurf'], label='Well 4', linewidth = 3, alpha=alpha, color='blueviolet')
## Add in a horizontal line at y = 0
axs[1].axhline(y=0, color='black', linestyle='-', alpha=alpha)
axs[1].set_ylabel('Relative to\nlocal land surface (m)', fontsize = fontsize)
axs[1].tick_params(axis='y', labelsize=fontsize-4)
axs[1].set_title('Hourly Depth to Groundwater', fontsize = fontsize)

## Add bold 'b' to the top left of subplot 2
axs[1].text(-0.06, 1.05, 'b', transform=axs[1].transAxes, fontsize=fontsize, fontweight='bold', va='top', ha='right')

## 3) Daily Precipitation from TJRTLMET station
axs[2].plot(IB_precip_1day, label='Daily', color='black')
axs[2].set_xlabel('Date', fontsize = fontsize-2)
axs[2].tick_params(axis='x', labelsize=fontsize-4)
axs[2].set_ylabel('(mm)', fontsize = fontsize)
axs[2].tick_params(axis='y', labelsize=fontsize-4)
axs[2].set_title('Daily Precipitation', fontsize = fontsize)

## Add bold 'c' to the top left of subplot 3
axs[2].text(-0.06, 1.05, 'c', transform=axs[2].transAxes, fontsize=fontsize, fontweight='bold', va='top', ha='right')

## Formatting
plt.xlim(pd.Timestamp('2021-12-01'), pd.Timestamp('2024-06-01'))
plt.gcf().autofmt_xdate()
fig.tight_layout()

# Set xticks and xtick labels
# axs[2].xaxis.set_major_locator(mdates.MonthLocator(interval=1))
axs[2].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

plt.tight_layout()
plt.show()

#%% Determine the closest Seacoast got to the road
print(seacoast_hourly['landsurf'].max())
print(seacoast_hourly['Timestamps'].iloc[np.where(seacoast_hourly==seacoast_hourly['landsurf'].max())[0][0]])

## Determine tide at that time
print(lj_tide.loc[seacoast_hourly['Timestamps'].iloc[np.where(seacoast_hourly==seacoast_hourly['landsurf'].max())[0][0]]])
