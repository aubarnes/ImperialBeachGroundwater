"""
Figure 1b: Plot Seacoast Road Elevation

November 2024
Austin Barnes
"""
#%% Imports
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib
import numpy as np

## Groundwater table
path_to_seacoast_elev = '../data/seacoast_elevation.csv'

#%% Load Seacoast Elevation Data
seacoast_elev = pd.read_csv(path_to_seacoast_elev)
#%% Convert to series
seacoast_elev = seacoast_elev.set_index('cds2d')

#%% Smooth the data with Gaussian filter
seacoast_elev = seacoast_elev.rolling(window=5, win_type='gaussian').mean(std=1)
seacoast_elev = seacoast_elev.dropna()
seacoast_elev = seacoast_elev.squeeze()
#%% Plot Seacoast Elevation
fontsize = 16
%matplotlib qt
plt.figure()
plt.plot(seacoast_elev, color='red', linewidth=3)
plt.xlabel('Distance from S Seacoast (m)', fontsize=fontsize)
plt.ylabel('Elevation (m, NAVD88)', fontsize=fontsize)
plt.title('Seacoast Road Elevation', fontsize=fontsize+2)
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)
plt.ylim(0,8)
plt.xlim(0,2550)
plt.grid()
plt.tight_layout()
plt.show()
# %%
