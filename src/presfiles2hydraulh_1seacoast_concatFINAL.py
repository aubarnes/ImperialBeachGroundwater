#%%
'''
Combines pandas dataframe files of time and pressure from well data into single files
Interpolates through short gaps of data during sensor collection
Converts the pressure data to water table height (WTH) relative to NAVD88 and land surface

This script is for Seacoast well data from 2022-2024
'''
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# For interpolation through sensor removal periods
from scipy import interpolate
# For checking if files exist already
import os.path
from os import path

# Directory where data exists
directory = '../data/'

# Customize what files to load
sensor = '207490'
in_files = [sensor+'_20220420_1222',
            sensor+'_20220615_0908',
            sensor+'_20220830_0938', #These files up to here are from original QC but being reintegrated here
            sensor+'_20221117_1256',
		 	sensor+'_20221201_1522',
		 	sensor+'_20230221_0905',
		 	sensor+'_20230412_0926']
sensor = '124141' # Replacement for Seacoast
in_files = np.concatenate((in_files,[sensor+'_20240117_0814',
		 							 sensor+'_20240214_1154',
                                     sensor+'_20240501_1234',
                                     sensor+'_20240514_1124']))

out_file = 'seacoast'+in_files[-1][6:]+'_QC'

# Input densities rho (kg/m^3)
# Salinity of Seacoast well is most complicated, varying up to 16 kg/m^3 but only on a couple of occasions
# Mean values based on all CTD casts, Kian Bagheri (2024)
rho_seacoast = 1012.5
rho_fifthgrove = 999.0
rho_pubworks = 1000.7
rho_eleventhebony = 998.8

#%% Load and combine all data

alldata_combined = pd.DataFrame()

plt.figure()
for file in in_files:
	print('Loading & Appending %s:' %(file))
	dat = pd.read_hdf(directory+file+'.h5','df')
	# Converting the Matlab Datenum format to date time format
    # value 719529 is the datenum value of the Unix epoch start (1970-01-01), which is the default origin for pd.to_datetime()
	matlabdatenums=dat['Time']-719529
    # Using Pandas datetime conversion
	timestamps = pd.to_datetime(matlabdatenums,unit='D')
	print(timestamps[0],timestamps[-1:])
	dat['Timestamps'] = timestamps
	cols = ['Time','Timestamps','Pressure']
	dat = dat[cols]
	alldata_combined = pd.concat([alldata_combined,dat],ignore_index=True)
	plt.plot(dat['Timestamps'],dat['Pressure'])
plt.show()
#%% Subtract atmospheric pressure from raw pressure
atmofile = '20240516' # Atmospheric data file end date

## Load atmospheric pressure data for use on all files
# if already loaded and saved, skip recreation
if path.exists(directory+'sdbay'+atmofile+'.npy'):
    sdbay_np = np.load(directory+'sdbay'+atmofile+'.npy')
# If it does  not yet exist, create
else:
    ## Alt Methods for Loading .txt or .csv Data ##
    # sdbay = open("nstp6h2017.txt","r")
    # sdbay = np.loadtxt('nstp6h2017.txt', delimiter = " ")
    ## Includes 2nd row of file with units
    # sdbay = pd.read_csv(directory+'nstp6h2017.txt',sep='\s+',header=[0,1])
    ## Skipping 2nd row of file to avoid creating multi-indexed dataframe (contained units)
    # Update atmospheric pressure file from SD Bay Meteorological observations:
    # https://tidesandcurrents.noaa.gov/met.html?bdate=20220720&edate=20220802&units=standard&timezone=GMT&id=9410170&interval=6
    sdbay = pd.read_csv(directory+'SDBay_atmo_data/'+'SDBay_1Dec2021_16May2024.csv',delim_whitespace=False,header=[0])
    ## Creating Timestamp for ease of plotting (documents are in UTC, our sensors are in UTC/GMT)

    # Rename dataframe columns
    sdbay = sdbay.rename(columns={"Date":"date","Time (GMT)":"time","Baro (mb)":"PRES"})

    sdbay['timestamp']=""
    sdbay['datetime']=""
    sdbay['matlabdatenum']=""
    for i in range(len(sdbay['date'])):
        sdbay['timestamp'][i] = pd.Timestamp(sdbay.date[i]+'T'+sdbay.time[i])
        sdbay['datetime'][i] = np.datetime64(sdbay.timestamp[i])
        # Create matlab datenum for interpolation (see notes below for more info)
        sdbay['matlabdatenum'][i]=(sdbay['datetime'][i]-pd.to_datetime(['1970-01-01']))/np.timedelta64(1,'D')+719529

    # Remove atmospheric pressure null readings (pressure = 9999), and interpolate to fill in
    sdbay.PRES[sdbay.PRES==9999] = np.NaN
    sdbay.PRES[sdbay.PRES=='-'] = np.NaN
    sdbay.PRES = sdbay.PRES.interpolate(method='pad').tolist()

    # Save for speed if needed again
    sdbay.to_hdf(directory+'sdbay'+atmofile+'.h5',key='df',mode='w')

    # Save numpy array version for calculations (easier than dataframe)
    # 2 columns: matlabdatenum and atmospheric pressure
    sdbay_np = np.empty((sdbay.shape[0],2))
    sdbay_np = np.full_like(sdbay_np,np.nan)
    for i in range(0,sdbay.shape[0]):
        sdbay_np[i,0] = sdbay['matlabdatenum'][i].values
    sdbay_np[:,1] = sdbay['PRES']

    np.save(directory+'sdbay'+atmofile+'.npy',sdbay_np)

# Function to interpolate atmospheric pressure
# f = interpolate.interp1d(sdbay_np[:,0],sdbay_np[:,1],fill_value='extrapolate')
f = interpolate.interp1d(sdbay_np[:,0],sdbay_np[:,1])

# Interpolate atmospheric pressure
interp_pres = f(alldata_combined['Time'])
# Find pressure difference in Pa & convert to depth of water column using rho (kg/m^3) and 9.81 m/s^2
# water column depth = P / rho * g
alldata_combined['waterdepth'] = (alldata_combined['Pressure']*10000-interp_pres*100)/rho_seacoast/9.81

#%% Plot full time series to figure out service dates

plt.plot(alldata_combined['Timestamps'],alldata_combined['waterdepth'])
plt.gcf().autofmt_xdate()
plt.show()

#%% Plot around servicing to determine exact interval for interpolation / removal

servicestart = np.where(alldata_combined['Timestamps']>np.datetime64('2024-05-01T19:30:00'))[0][0]
servicestop = np.where(alldata_combined['Timestamps']<np.datetime64('2024-05-01T19:45:00'))[0][-1]

plt.plot(alldata_combined['Timestamps'][servicestart:servicestop],alldata_combined['waterdepth'][servicestart:servicestop])
plt.gcf().autofmt_xdate()

#%% QC Data (remove service & offsets)

timestart = np.datetime64('2021-12-08T00:00:00') # DONE
timestop = np.datetime64('2024-05-14T17:00:00') # DONE

idxstart = np.where(alldata_combined['Timestamps']>timestart)[0][0]
idxstop = np.where(alldata_combined['Timestamps']<timestop)[0][-1]
QCdata = alldata_combined[idxstart:idxstop].reset_index(drop=True)
QCdata['NAVD88'] = np.full_like(QCdata['waterdepth'],np.nan)
QCdata['landsurf'] = np.full_like(QCdata['waterdepth'],np.nan)

servicestart1 = np.where(QCdata['Timestamps']>np.datetime64('2022-02-23T23:45:00'))[0][0] #DONE
servicestop1 = np.where(QCdata['Timestamps']<np.datetime64('2022-02-24T00:10:00'))[0][-1] #DONE

servicenans = pd.DataFrame(QCdata,copy=True)
servicenans['waterdepth'][servicestart1:servicestop1] = np.NaN
QCdata['waterdepth'][servicestart1-1:servicestop1+1] = servicenans['waterdepth'][servicestart1-1:servicestop1+1].interpolate()

servicestart2 = np.where(QCdata['Timestamps']>np.datetime64('2022-04-20T19:15:00'))[0][0] #DONE
servicestop2 = np.where(QCdata['Timestamps']<np.datetime64('2022-04-20T19:45:00'))[0][-1] #DONE

servicenans = pd.DataFrame(QCdata,copy=True)
servicenans['waterdepth'][servicestart2:servicestop2] = np.NaN
QCdata['waterdepth'][servicestart2-1:servicestop2+1] = servicenans['waterdepth'][servicestart2-1:servicestop2+1].interpolate()

servicestart3 = np.where(QCdata['Timestamps']>np.datetime64('2022-06-06T06:00:00'))[0][0] #DONE
servicestop3 = np.where(QCdata['Timestamps']<np.datetime64('2022-08-03T01:00:00'))[0][-1] #DONE

servicenans = pd.DataFrame(QCdata,copy=True)
servicenans['waterdepth'][servicestart3:servicestop3] = np.NaN
QCdata['waterdepth'][servicestart3-1:servicestop3+1] = servicenans['waterdepth'][servicestart3-1:servicestop3+1]

servicestart3B = np.where(QCdata['Timestamps']>np.datetime64('2022-08-30T16:30:00'))[0][0] #DONE
servicestop3B = np.where(QCdata['Timestamps']<np.datetime64('2022-08-30T16:50:00'))[0][-1] #DONE

servicenans = pd.DataFrame(QCdata,copy=True)
servicenans['waterdepth'][servicestart3B:servicestop3B] = np.NaN
QCdata['waterdepth'][servicestart3B-1:servicestop3B+1] = servicenans['waterdepth'][servicestart3B-1:servicestop3B+1].interpolate()

servicestart4 = np.where(QCdata['Timestamps']>np.datetime64('2022-11-17T20:50:00'))[0][0] # DONE
servicestop4 = np.where(QCdata['Timestamps']<np.datetime64('2022-11-17T21:10:00'))[0][-1] # DONE

servicenans = pd.DataFrame(QCdata,copy=True)
servicenans['waterdepth'][servicestart4:servicestop4] = np.NaN
QCdata['waterdepth'][servicestart4-1:servicestop4+1] = servicenans['waterdepth'][servicestart4-1:servicestop4+1].interpolate()

servicestart5 = np.where(QCdata['Timestamps']>np.datetime64('2022-12-01T23:00:00'))[0][0] # DONE
servicestop5 = np.where(QCdata['Timestamps']<np.datetime64('2023-02-10T23:00:00'))[0][-1] # DONE

servicenans = pd.DataFrame(QCdata,copy=True)
servicenans['waterdepth'][servicestart5:servicestop5] = np.NaN
QCdata['waterdepth'][servicestart5:servicestop5] = servicenans['waterdepth'][servicestart5:servicestop5]

# Convert to water table height (WTH) relative to NAVD88:
# WTH rel NAVD88 = Elev of well head rel to NAVD88 - Depth of sensor + depth of water column
QCdata['NAVD88'][:servicestart5] = 1.741-4.521+QCdata['waterdepth'][:servicestart5]

servicestart6 = np.where(QCdata['Timestamps']>np.datetime64('2023-02-21T17:00:00'))[0][0] # DONE
servicestop6 = np.where(QCdata['Timestamps']<np.datetime64('2023-02-21T17:15:00'))[0][-1] # DONE

servicenans = pd.DataFrame(QCdata,copy=True)
servicenans['waterdepth'][servicestart6:servicestop6] = np.NaN
QCdata['waterdepth'][servicestart6-1:servicestop6+1] = servicenans['waterdepth'][servicestart6-1:servicestop6+1].interpolate()

servicestart7 = np.where(QCdata['Timestamps']>np.datetime64('2023-04-12T16:00:00'))[0][0] # DONE
servicestop7 = np.where(QCdata['Timestamps']<np.datetime64('2023-11-22T01:00:00'))[0][-1] # DONE

servicenans = pd.DataFrame(QCdata,copy=True)
servicenans['waterdepth'][servicestart7:servicestop7] = np.NaN
QCdata['waterdepth'][servicestart7:servicestop7] = servicenans['waterdepth'][servicestart7:servicestop7]

# Convert to water table height (WTH) relative to NAVD88:
# WTH rel NAVD88 = Elev of well head rel to NAVD88 - depth of water table
QCdata['NAVD88'][servicestop5:servicestart7] = 1.741-(0.85+QCdata['waterdepth'][servicestop5])+QCdata['waterdepth'][servicestop5:servicestart7]

servicestart8 = np.where(QCdata['Timestamps']>np.datetime64('2024-01-17T16:10:00'))[0][0] # DONE
servicestop8 = np.where(QCdata['Timestamps']<np.datetime64('2024-01-17T16:30:00'))[0][-1] # DONE

servicenans = pd.DataFrame(QCdata,copy=True)
servicenans['waterdepth'][servicestart8:servicestop8] = np.NaN
QCdata['waterdepth'][servicestart8-1:servicestop8+1] = servicenans['waterdepth'][servicestart8-1:servicestop8+1].interpolate()

QCdata['NAVD88'][servicestop7:servicestart8] = 1.741-(0.521+QCdata['waterdepth'][servicestop7])+QCdata['waterdepth'][servicestop7:servicestart8]

servicestart9 = np.where(QCdata['Timestamps']>np.datetime64('2024-02-14T19:50:00'))[0][0] # DONE
servicestop9 = np.where(QCdata['Timestamps']<np.datetime64('2024-02-14T20:10:00'))[0][-1] # DONE

servicenans = pd.DataFrame(QCdata,copy=True)
servicenans['waterdepth'][servicestart9:servicestop9] = np.NaN
QCdata['waterdepth'][servicestart9-1:servicestop9+1] = servicenans['waterdepth'][servicestart9-1:servicestop9+1].interpolate()

QCdata['NAVD88'][servicestop8:servicestart9] = 1.741-(0.873+QCdata['waterdepth'][servicestop8])+QCdata['waterdepth'][servicestop8:servicestart9]

servicestart10 = np.where(QCdata['Timestamps']>np.datetime64('2024-05-01T19:30:00'))[0][0] # DONE
servicestop10 = np.where(QCdata['Timestamps']<np.datetime64('2024-05-01T19:45:00'))[0][-1] # DONE

servicenans = pd.DataFrame(QCdata,copy=True)
servicenans['waterdepth'][servicestart10:servicestop10] = np.NaN
QCdata['waterdepth'][servicestart10-1:servicestop10+1] = servicenans['waterdepth'][servicestart10-1:servicestop10+1].interpolate()

QCdata['NAVD88'][servicestop9:servicestart10] = 1.741-(0.563+QCdata['waterdepth'][servicestop9])+QCdata['waterdepth'][servicestop9:servicestart10]

QCdata['NAVD88'][servicestop10:] = 1.741-(1.075+QCdata['waterdepth'][servicestop10])+QCdata['waterdepth'][servicestop10:]

# Convert to water table height relative to land surface:
# WTH rel LS  =  WTH rel NAVD88 - Elev of ground surface rel to NAVD88
QCdata['landsurf'] = QCdata['NAVD88']-1.846

#%% Plot to check QC'd data
plt.plot(QCdata['Timestamps'][0:-1:10],QCdata['NAVD88'][0:-1:10],'.')
ax = plt.twinx()
ax.plot(QCdata['Timestamps'][0:-1:10],QCdata['landsurf'][0:-1:10],'r.')
plt.gcf().autofmt_xdate()
plt.show()

#%% SAVE QC'd data
print('Saving combined QCd data in %s' %out_file+'.h5')
QCdata.to_hdf(directory+out_file+'.h5',key='df',mode='w')
print('Saved %s' %out_file+'.h5')
