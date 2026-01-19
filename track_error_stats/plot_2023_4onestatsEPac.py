import netCDF4 as nc4
import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
import matplotlib.dates as mdates
from scipy.spatial import distance
from mpl_toolkits.basemap import Basemap
from matplotlib.dates import DayLocator, HourLocator, DateFormatter, drange
from numpy import arange
from datetime import datetime, timedelta

import pandas as pd
import glob,os
import sys
import csv
import xarray as xr
import cartopy.crs       as     ccrs
import cartopy.feature   as     cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

path = '/discover/nobackup/projects/gmao/obsdev/mchattop/TC_stuff/track_error_stats/EPac/2023' # use your path

all_files = glob.glob(os.path.join(path ,"summary_4onestats_*.txt"))
print(all_files)

li = []

for filename in all_files:
    if (filename == "/discover/nobackup/projects/gmao/obsdev/mchattop/TC_stuff/track_error_stats/EPac/2023/summary_4onestats_Dora1.txt"):
        df = pd.read_csv(filename,delim_whitespace=True,usecols=['FCST_HR','mean'])
    else:    
        df = pd.read_csv(filename,delim_whitespace=True,usecols=['mean'])
    li.append(df)

frame = pd.concat(li, axis=1,ignore_index=True)
frame.columns = ['Greg','Fernanda','Jova','Otis','Kenneth','Hilary','Norma','FCST_HR','Dora','Pilar','Calvin','Lidia','Adrian']
print(frame)

ax=frame.iloc[0:, :].plot(x='FCST_HR',y=['Adrian','Calvin','Dora','Fernanda','Greg','Hilary','Jova','Kenneth','Lidia','Norma','Otis','Pilar'],xticks=frame.iloc[1:, :]['FCST_HR'])
ax.set_xlabel('FCST_HR')
ax.set_ylabel('Nautical Miles')
plt.ylim(0,225)
plt.xticks(fontsize=8)
ax.set_title('East Pacific Tropical Cyclones 2023')
plt.savefig('EPacTC_2023.png')
plt.show()
sys.exit()

