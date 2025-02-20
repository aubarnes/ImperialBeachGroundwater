#%% Imports
%matplotlib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

directory = '../data/'

#%% Load the hydraulic head data
seacoast =  pd.read_hdf(directory + "seacoast_20240514_1124_QC.h5").reset_index(drop=True)
fifthgrove = pd.read_hdf(directory + "fifthgrove_20240514_1124_QC.h5").reset_index(drop=True)
pubworks = pd.read_hdf(directory + "pubworks_20240514_1435_QC.h5").reset_index(drop=True)
eleventhebony = pd.read_hdf(directory + "eleventhebony_20240514_1124_QC.h5").reset_index(drop=True)

#%% Pull nearest non-NAN Time, Timestamps, NAVD88, and landsurf around periods of NaNs
seacoast_nans = seacoast[seacoast['NAVD88'].isna()]