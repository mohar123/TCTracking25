import netCDF4 as nc4
import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
import matplotlib.dates as mdates
from scipy.spatial import distance
from mpl_toolkits.basemap import Basemap
from matplotlib.dates import DayLocator, HourLocator, DateFormatter, drange
from numpy import arange

import pandas as pd
import glob,os
import sys
import csv
import xarray as xr
import cartopy.crs       as     ccrs
import cartopy.feature   as     cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

path = '/discover/nobackup/projects/gmao/obsdev/mchattop/TC_stuff/track_error_stats/EPac/2024' # use your path

all_files = glob.glob(os.path.join(path ,"summary_4allstats_*.txt"))
print(all_files)

li = []

for filename in all_files:
    print(filename)
    df = pd.read_csv(filename,delim_whitespace=True,usecols=lambda col: col not in ["FCST_HR"])
    li.append(df)

frame = pd.concat(li, axis=1, ignore_index=True)
print(frame.iloc[:,30:41])

frame['mean'] = frame.iloc[:,:41].mean(axis=1)
frame['stddev'] = frame.iloc[:,:41].std(axis=1)
frame['count'] = frame.iloc[:,:41].count(axis=1)
frame['FCST_HR'] = frame.index*6
#frame['obscnt'] = frame['FCST_HR'].astype(int).astype(str) +"("+ frame["count"].astype(str)+")"
frame['obscnt'] = frame['FCST_HR'].astype(int).astype(str)+"\n" +"("+frame["count"].astype(str)+")"
print(frame.iloc[18:,:])

#fig = plt.figure(figsize=(9, 12))
#ax=frame.iloc[0:, :].plot(x='FCST_HR',y='mean',yerr='stddev',capsize=3,xticks=frame.iloc[0:, :]['FCST_HR'])
ax=frame.iloc[1:, :].plot(x='FCST_HR',y='mean',xticks=frame.iloc[1:, :]['FCST_HR'])
ax.fill_between(frame.iloc[1:, :]['FCST_HR'], frame.iloc[1:, :]['mean'] - frame.iloc[1:, :]['stddev'],frame.iloc[1:, :]['mean'] + frame.iloc[1:, :]['stddev'], alpha=0.2)
ax.set_xticklabels(frame.iloc[1:, :]['obscnt'])
plt.xticks(fontsize=8)
plt.ylim(0,350)
plt.xlim(6,120)
ax.set_xlabel('FCST_HR (Data Count)')
#plt.xlabel('FCST_HR\n(Data Count)',fontsize=7)
ax.set_ylabel('Nautical Miles')
ax.legend(['Mean Track Error'])
ax.set_title('East Pacific Tropical Cyclone for 2024')
plt.savefig('EPacAllMeanTrkErr_2024.png')
plt.show()

sys.exit()

