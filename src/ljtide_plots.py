#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

path_to_ljtide_full = '../data/ljtide_1924.h5'
ljtide_full = pd.read_hdf(path_to_ljtide_full, 'ljtide')
#%% 1 year running mean of ljtide_full
ljtide_1yrrun = ljtide_full.rolling(window=365*24, min_periods=1).mean()
ljtide_month_high = ljtide_full.resample('M').max()
# %%
plt.plot(ljtide_1yrrun[365*24:])
#%% FFT of ljtide_1yrrun
ljtide_1yrrun_fft = np.fft.fft(ljtide_1yrrun[365*24:])
ljtide_1yrrun_fft = np.abs(ljtide_1yrrun_fft)
ljtide_1yrrun_fft = ljtide_1yrrun_fft[:len(ljtide_1yrrun_fft)//2]
ljtide_1yrrun_fft_freq = np.fft.fftfreq(len(ljtide_1yrrun[365*24:]), 1/24)[:len(ljtide_1yrrun_fft)//2]

plt.plot(ljtide_1yrrun_fft_freq, ljtide_1yrrun_fft)
plt.xlim(0, 0.5)
plt.xlabel('Frequency (cycles per day)')
plt.ylabel('Amplitude')
plt.title('FFT of 1 year running mean of ljtide_full')
plt.grid()
plt.show()
# %%
