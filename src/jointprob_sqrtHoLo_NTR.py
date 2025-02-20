"""
Visualize the joint probability of Non-Tidal Residuals and sqrt(HoLo) from the 24 years of data.
"""
#%% Imports & Define Directory
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib

mopnum=38

path_to_ntr_dailymax = '/Users/austinbarnes/Documents/UCSD SIO/IB Groundwater/ImperialBeach/data/LJ_tide_data/ljtide_dailymax_24.h5'
path_to_sqrtHoLo_dailymax = f'/Users/austinbarnes/Documents/UCSD SIO/IB Groundwater/ImperialBeach/data/runup/sqrtHoLo_mop{mopnum}_24.h5'

#%% Load the 24 years of Daily Maximum Non-Tidal Residuals & R2%
ntr_dailymax = pd.read_hdf(path_to_ntr_dailymax)
print('Loaded 24 Realizations of Non-Tidal Residuals from ljtide_dailymax_24.h5')
sqrtHoLo_dailymax = pd.read_hdf(path_to_sqrtHoLo_dailymax)
print(f'Loaded 24 Realizations of sqrt(HoLo) from MOP {mopnum} data from file.')

## Drop row 151
ntr_dailymax = ntr_dailymax.drop(151)
sqrtHoLo_dailymax = sqrtHoLo_dailymax.drop(151)
#%% Plot Histogram of NTR and R2%
plt.figure()
plt.hist(ntr_dailymax.values.flatten(), bins=100, alpha=0.5, label='NTR')
plt.hist(sqrtHoLo_dailymax.values.flatten(), bins=100, alpha=0.5, label='sqrt(HoLo)')
plt.legend()
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram of NTR and sqrt(HoLo)')
plt.show()


#%% Compute the joint probability of NTR and R2%
ntr = ntr_dailymax.values.flatten()
sqrtHoLo = sqrtHoLo_dailymax.values.flatten()

# Calculate 2D histogram
hist, xedges, yedges = np.histogram2d(sqrtHoLo, ntr, bins=30, density=False)

# Convert the histogram counts to probabilities (joint probability distribution)
prob_hist = hist / np.sum(hist)

#%% Plot the joint probability histogram from Numpy
%matplotlib qt
plt.figure()
plt.imshow(prob_hist.T, origin='lower', cmap='Blues',
           extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
plt.colorbar(label='Joint Probability')
plt.xlabel('sqrt(HoLo) (m)')
plt.ylabel('Non-Tidal Residual (m)')
plt.title('Joint Probability Histogram')
plt.tight_layout()
plt.show()
#%% Plot the joint probability histogram with Seaborn
sns.jointplot(x=sqrtHoLo, y=ntr, kind='hist', color='b')
plt.xlabel('sqrt(HoLo) (m)')
plt.ylabel('Non-Tidal Residual (m)')
plt.title('Joint Probability Density Estimate')
plt.show()
#%% Joint probability for winter months only
## November is day 31 through 61
## December is day 61 through 92
## January is day 92 through 123
## February is day 123 through 151
## March is day 151 through 182
winter_indices = np.arange(31,183,1)

ntr_dailymax_winter = ntr_dailymax.iloc[winter_indices]
sqrtHoLo_dailymax_winter = sqrtHoLo_dailymax.iloc[winter_indices]

ntr_winter = ntr_dailymax_winter.values.flatten()
sqrtHoLo_winter = sqrtHoLo_dailymax_winter.values.flatten()

#%% Plot the joint probability histogram with Seaborn
%matplotlib qt
sns.jointplot(x=sqrtHoLo_winter, y=ntr_winter, kind='hist', color='b')
plt.xlabel('sqrt(HoLo) (m)')
plt.ylabel('Non-Tidal Residual (m)')
plt.title('Joint Probability Density Estimate (Nov - Mar)')
plt.show()
# %%
