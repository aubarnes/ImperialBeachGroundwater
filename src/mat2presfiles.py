#%%
'''
Converts matlab .mat file of pressure sensor data to pandas DataFrame
matlab file already was created from raw .rsk files using rsk2mat.m
.rsk files contain timestamps and raw pressure
'''
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# To load matlab files
from scipy.io import loadmat

#%%

# Directory where data exists
directory = '../data/'
#%%
# Customize what files to load

# sensor = '207490' # Seacoast
# files = [sensor+'_20220420_1222',sensor+'_20220615_0908'] # Needed time correction from LST -> UTC
# files = [sensor+'_20220830_0938'] # DONE
# files = [sensor+'_20221117_1256',
# 		 sensor+'_20221201_1522',
# 		 sensor+'_20230221_0905',
# 		 sensor+'_20230412_0926'] # DONE
sensor = '124141' # Replacement for Seacoast
# files = np.concatenate((files,[sensor+'_20240117_0814',
# 		 sensor+'_20240214_1154'])) # DONE
files = [sensor+'_20240501_1234',
		 sensor+'_20240514_1124']


# sensor = '207492' # 5th & Grove
# files = [sensor+'_20220420_1312',sensor+'_20220615_1231'] # Needed time correction from LST -> UTC
# files = [sensor+'_20220802_1623'] DONE
# files = np.concatenate((files,[sensor+'_20221117_1226',
# 		 sensor+'_20230124_1501',
# 		 sensor+'_20230210_1523',
# 		 sensor+'_20230221_0933',
# 		 sensor+'_20230412_0952',
# 		 sensor+'_20231121_1609',
# 		 sensor+'_20240117_0856'])) # DONE
sensor = '041363' # Replacement for 5th & Grove
files = np.concatenate((files,[sensor+'_20240501_1313',
		 sensor+'_20240514_1124']))

sensor = '207493' # IB Public Works
# files = [sensor+'_20220615_1021'] # Needed time correction from LST -> UTC
# files = [sensor+'_20220802_1458'] DONE
# files = np.concatenate((files,[sensor+'_20221117_1355',
# 		 sensor+'_20230124_1416',
# 		 sensor+'_20230221_0959',
# 		 sensor+'_20230412_0850',
# 		 sensor+'_20230927_1505',
# 		 sensor+'_20240117_0937',
# 		 sensor+'_20240214_1253'])) # DONE
files = np.concatenate((files,[sensor+'_20240501_1529',
		 sensor+'_20240514_1435']))

sensor = '207494' # 11th & Ebony
# files = [sensor+'_20220420_1134',sensor+'_20220615_1109'] # Needed time correction from LST -> UTC
# files = [sensor+'_20220802_1600'] DONE
# files = np.concatenate((files,[sensor+'_20221117_1057',
# 		 sensor+'_20221201_1437',
# 		 sensor+'_20230124_1440',
# 		 sensor+'_20230210_1539',
# 		 sensor+'_20230412_1011',
# 		 sensor+'_20231121_1712',
# 		 sensor+'_20240214_1223'])) # DONE
files = np.concatenate((files,[sensor+'_20240508_1002',
		 sensor+'_20240514_1124']))

#%%

# Load .mat files into Python & save as numpy arrays
for file in files:
	print('Loading %s:' %(file))
	dat = loadmat(directory+file+'.mat')
	print(dat.keys())

	# Create temporary arrays & turn into lists before creating a pandas dataframe from the two separate lists
	# Time 'tstamp' in matlab datenum format
	t_temp = dat['tstamp']
	t_temp = t_temp

	### Use only for some of the earliest files that have may have apparent timing issue (local time instead of UTC)
	# t_temp = t_temp+8/24
	# # Add 8/7 hours (in days) to match sensor time with UTC (8 hrs ahead from local Pacific w/ NO DST, 7 hrs ahead when DST)
	# for i in range(t_temp.shape[0]):
	# 	if t_temp[i]<7.385930833333334e+05:
	# 		t_temp[i] = t_temp[i]+8/24
	# 	elif t_temp[i]>7.385930833333334e+05:
	# 		t_temp[i] = t_temp[i]+7/24

	# Turn first array of times into list
	time_temp = np.squeeze(t_temp)
	# time_temp = t_temp[0].tolist()
	# # Add on subsequent arrays of times onto list, sequentially
	# for i in range(1,len(t_temp)):
	# 	time_temp = time_temp + t_temp[i].tolist()

	# Pressure 'pressure' in psi
	p_temp = dat['pressure']
	pres_temp = np.squeeze(p_temp)
	# pres_temp = p_temp[0].tolist()
	# # Add on subsequent arrays of times onto list, sequentially
	# for i in range(1,len(p_temp)):
	# 	pres_temp = pres_temp + p_temp[i].tolist()

	# Create numpy arrays from the two lists
	tp_temp = np.array(list(zip(time_temp,pres_temp)))

	# Sort by increasing time stamps
	sorted = tp_temp[np.argsort(tp_temp[:,0])]

    # Remove large pressure outliers based on pressure being more than 5 std's greater than mean
	sorted = sorted[sorted[:,1]<np.mean(sorted[:,1])+5*np.std(sorted[:,1])]

	# Create pandas dataframe from the two lists
	tp_df = pd.DataFrame(sorted,columns=['Time','Pressure'])
	# plt.plot(tp_df["Time"])
	# plt.show()

	# Save DataFrame to h5 format for storage and loading
	print('Saving %s' %file+'.h5')
	tp_df.to_hdf(directory+file+'.h5',key='df',mode='w')
	print('Saved %s' %file+'.h5')

	# To Load:
	# AmouliX = pd.read_hdf(directory+file+'.h5','df')

print('Loading .mat and saving .npz and .h5 files complete')
# %%
