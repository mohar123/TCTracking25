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

# to create ncdf4 files need the following tools
import datetime  # Python standard library datetime  module
import time
from netCDF4 import Dataset,num2date, date2num  # http://code.google.com/p/netcdf4-python/

#---
# calculate track error
from math import radians, cos, sin, asin, sqrt
def distance(lat1, lat2, lon1, lon2):

    # The math module contains a function named
    # radians which converts from degrees to radians.
    lon1 = radians(lon1)
    lon2 = radians(lon2)
    lat1 = radians(lat1)
    lat2 = radians(lat2)

    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2

    c = 2 * asin(sqrt(a))

 # Radius of earth in kilometers. Use 3956 for miles
    r = 6371

    # calculate the result
    return(c * r)
#---
#ONE
#---
# Obtain the date/time and initial lat/lon from command line:
dirm = "/archive/u/dao_ops/f5294_fp/forecast/Y2022/M06/D15/H00/"
col = "GEOS.fp.fcst.inst3_3d_asm_Np.20220615_00+"
beg_date = datetime.datetime(year=2022, month=6, day=15, hour=00)
end_date = datetime.datetime(year=2022, month=6, day=21, hour=6)

delta_date = datetime.timedelta(hours=6)

list_files = list()
list_fileso = list()

cur_date = beg_date

while cur_date < end_date:
    file = cur_date.strftime(dirm+col+"%Y%m%d_%H%M.V01.nc4")
    list_files.append(file)
    cur_date += delta_date

print(list_files)
lon_all, lat_all, slp_min_all, tim_all,fcst_all = [], [], [], [],[]
del_lat = 1.5
del_lon = 2.0
lat_slice1 = 14.5
lat_slice2 = 18.
lon_slice1 = -108.
lon_slice2 = -101.5

for file in list_files:
    with xr.open_dataset(file, engine='netcdf4') as f:  
        
        #print(f.keys())
        slp_1 = f.SLP.sel(lat=slice(lat_slice1,lat_slice2),lon=slice(lon_slice1,lon_slice2))
        slp_1_min = float(slp_1.min())
        loc1 = np.where(slp_1[0,:,:] == np.amin(slp_1[0,:,:]))
        lat1 = float(slp_1.lat[loc1[0]].values)
        lon1 = float(slp_1.lon[loc1[1]].values)
        tim1 = (slp_1.time.values) 
        
        ds = str(tim1[0]).partition('T')[0].split('-')
        #print(date_str,type(date_str))
        time_str = str(tim1[0]).partition('T')[2][0:2]
        #print(time_str,type(time_str))
        ds.append(time_str)
        #print(date_str)
        d = datetime.datetime(int(ds[0]), int(ds[1]), int(ds[2]), int(ds[3]))
        ds1 = d.strftime("%Y%m%d%H")
        fcst_diff1 = (d-beg_date).total_seconds() / 3600

        if lat_all:
           if (lat1 <= lat_all[-1]+del_lat) and (lat1 >= lat_all[-1]-del_lat) and \
              (lon1 <= lon_all[-1]+del_lon) and (lon1 >= lon_all[-1]-del_lon):
              lat_all.append(lat1)
              slp_min_all.append(slp_1_min)
              lon_all.append(lon1)
              tim_all.append(ds1)
              fcst_all.append(fcst_diff1)
        else:
            slp_min_all.append(slp_1_min)
            lon_all.append(lon1)
            lat_all.append(lat1)
            tim_all.append(ds1)
            fcst_all.append(fcst_diff1)

df1 = pd.DataFrame((zip(tim_all,lat_all,lon_all,slp_min_all,fcst_all)), columns = ['c1','c2','c3','c4','c5'])
print(df1)
#sys.exit()
#------------------------------
#---
#TWO
#---
# Obtain the date/time and initial lat/lon from command line:
dirm = "/archive/u/dao_ops/f5294_fp/forecast/Y2022/M06/D16/H00/"
col = "GEOS.fp.fcst.inst3_3d_asm_Np.20220616_00+"
beg_date = datetime.datetime(year=2022, month=6, day=16, hour=00)
end_date = datetime.datetime(year=2022, month=6, day=21, hour=6)

delta_date = datetime.timedelta(hours=6)

list_files = list()
list_fileso = list()

cur_date = beg_date

while cur_date < end_date:
    file = cur_date.strftime(dirm+col+"%Y%m%d_%H%M.V01.nc4")
    list_files.append(file)
    cur_date += delta_date

print(list_files)
lon_all1, lat_all1, slp_min_all1, tim_all1,fcst_all1 = [], [], [], [],[]
del_lat = 1.5
del_lon = 2.0
lat_slice1 = 14.5
lat_slice2 = 20.
lon_slice1 = -113.
lon_slice2 = -102.

for file in list_files:
    with xr.open_dataset(file, engine='netcdf4') as f:  
        slp_1 = f.SLP.sel(lat=slice(lat_slice1,lat_slice2),lon=slice(lon_slice1,lon_slice2))
        slp_1_min = float(slp_1.min())
        loc1 = np.where(slp_1[0,:,:] == np.amin(slp_1[0,:,:]))
        lat1 = float(slp_1.lat[loc1[0]].values)
        lon1 = float(slp_1.lon[loc1[1]].values)
        tim1 = (slp_1.time.values) 
        
        ds = str(tim1[0]).partition('T')[0].split('-')
        #print(date_str,type(date_str))
        time_str = str(tim1[0]).partition('T')[2][0:2]
        #print(time_str,type(time_str))
        ds.append(time_str)
        #print(date_str)
        d = datetime.datetime(int(ds[0]), int(ds[1]), int(ds[2]), int(ds[3]))
        ds1 = d.strftime("%Y%m%d%H")
        fcst_diff1 = (d-beg_date).total_seconds() / 3600

        if lat_all1:
           if (lat1 <= lat_all1[-1]+del_lat) and (lat1 >= lat_all1[-1]-del_lat) and \
              (lon1 <= lon_all1[-1]+del_lon) and (lon1 >= lon_all1[-1]-del_lon):
              lat_all1.append(lat1)
              slp_min_all1.append(slp_1_min)
              lon_all1.append(lon1)
              tim_all1.append(ds1)
              fcst_all1.append(fcst_diff1)
        else:
            slp_min_all1.append(slp_1_min)
            lon_all1.append(lon1)
            lat_all1.append(lat1)
            tim_all1.append(ds1)
            fcst_all1.append(fcst_diff1)

df2 = pd.DataFrame((zip(tim_all1,lat_all1,lon_all1,slp_min_all1,fcst_all1)), columns = ['c1','c2','c3','c4','c5'])

#---
#THREE
#---
# Obtain the date/time and initial lat/lon from command line:
dirm = "/archive/u/dao_ops/f5294_fp/forecast/Y2022/M06/D17/H00/"
col = "GEOS.fp.fcst.inst3_3d_asm_Np.20220617_00+"
beg_date = datetime.datetime(year=2022, month=6, day=17, hour=00)
end_date = datetime.datetime(year=2022, month=6, day=21, hour=6)

delta_date = datetime.timedelta(hours=6)

list_files = list()
list_fileso = list()

cur_date = beg_date

while cur_date < end_date:
    file = cur_date.strftime(dirm+col+"%Y%m%d_%H%M.V01.nc4")
    list_files.append(file)
    cur_date += delta_date

print(list_files)
lon_all2, lat_all2, slp_min_all2, tim_all2,fcst_all2 = [], [], [], [],[]
del_lat = 1.
del_lon = 2.0
lat_slice1 = 16.
lat_slice2 = 20.
lon_slice1 = -113.
lon_slice2 = -104.

for file in list_files:
    with xr.open_dataset(file, engine='netcdf4') as f:  
        slp_1 = f.SLP.sel(lat=slice(lat_slice1,lat_slice2),lon=slice(lon_slice1,lon_slice2))
        slp_1_min = float(slp_1.min())
        loc1 = np.where(slp_1[0,:,:] == np.amin(slp_1[0,:,:]))
        lat1 = float(slp_1.lat[loc1[0]].values)
        lon1 = float(slp_1.lon[loc1[1]].values)
        tim1 = (slp_1.time.values) 
        
        ds = str(tim1[0]).partition('T')[0].split('-')
        #print(date_str,type(date_str))
        time_str = str(tim1[0]).partition('T')[2][0:2]
        #print(time_str,type(time_str))
        ds.append(time_str)
        #print(date_str)
        d = datetime.datetime(int(ds[0]), int(ds[1]), int(ds[2]), int(ds[3]))
        ds1 = d.strftime("%Y%m%d%H")
        fcst_diff1 = (d-beg_date).total_seconds() / 3600

        if lat_all2:
           if (lat1 <= lat_all2[-1]+del_lat) and (lat1 >= lat_all2[-1]-del_lat) and \
              (lon1 <= lon_all2[-1]+del_lon) and (lon1 >= lon_all2[-1]-del_lon):
              lat_all2.append(lat1)
              slp_min_all2.append(slp_1_min)
              lon_all2.append(lon1)
              tim_all2.append(ds1)
              fcst_all2.append(fcst_diff1)
        else:
            slp_min_all2.append(slp_1_min)
            lon_all2.append(lon1)
            lat_all2.append(lat1)
            tim_all2.append(ds1)
            fcst_all2.append(fcst_diff1)

df3 = pd.DataFrame((zip(tim_all2,lat_all2,lon_all2,slp_min_all2,fcst_all2)), columns = ['c1','c2','c3','c4','c5'])

#---
#FOUR
#---
# Obtain the date/time and initial lat/lon from command line:
dirm = "/archive/u/dao_ops/f5294_fp/forecast/Y2022/M06/D18/H00/"
col = "GEOS.fp.fcst.inst3_3d_asm_Np.20220618_00+"
beg_date = datetime.datetime(year=2022, month=6, day=18, hour=00)
end_date = datetime.datetime(year=2022, month=6, day=21, hour=6)

delta_date = datetime.timedelta(hours=6)

list_files = list()
list_fileso = list()

cur_date = beg_date

while cur_date < end_date:
    file = cur_date.strftime(dirm+col+"%Y%m%d_%H%M.V01.nc4")
    list_files.append(file)
    cur_date += delta_date

print(list_files)
lon_all3, lat_all3, slp_min_all3, tim_all3,fcst_all3 = [], [], [], [],[]
del_lat = 1.5
del_lon = 2.0
lat_slice1 = 17.
lat_slice2 = 19.
lon_slice1 = -113.
lon_slice2 = -109.

for file in list_files:
    with xr.open_dataset(file, engine='netcdf4') as f:  
        slp_1 = f.SLP.sel(lat=slice(lat_slice1,lat_slice2),lon=slice(lon_slice1,lon_slice2))
        slp_1_min = float(slp_1.min())
        loc1 = np.where(slp_1[0,:,:] == np.amin(slp_1[0,:,:]))
        lat1 = float(slp_1.lat[loc1[0]].values)
        lon1 = float(slp_1.lon[loc1[1]].values)
        tim1 = (slp_1.time.values) 
        ds = str(tim1[0]).partition('T')[0].split('-')
        #print(date_str,type(date_str))
        time_str = str(tim1[0]).partition('T')[2][0:2]
        #print(time_str,type(time_str))
        ds.append(time_str)
        #print(date_str)
        d = datetime.datetime(int(ds[0]), int(ds[1]), int(ds[2]), int(ds[3]))
        ds1 = d.strftime("%Y%m%d%H")
        fcst_diff1 = (d-beg_date).total_seconds() / 3600

        if lat_all3:
           if (lat1 <= lat_all3[-1]+del_lat) and (lat1 >= lat_all3[-1]-del_lat) and \
              (lon1 <= lon_all3[-1]+del_lon) and (lon1 >= lon_all3[-1]-del_lon):
              lat_all3.append(lat1)
              slp_min_all3.append(slp_1_min)
              lon_all3.append(lon1)
              tim_all3.append(ds1)
              fcst_all3.append(fcst_diff1)
        else:
            slp_min_all3.append(slp_1_min)
            lon_all3.append(lon1)
            lat_all3.append(lat1)
            tim_all3.append(ds1)
            fcst_all3.append(fcst_diff1)

df4 = pd.DataFrame((zip(tim_all3,lat_all3,lon_all3,slp_min_all3,fcst_all3)), columns = ['c1','c2','c3','c4','c5'])        
#------------------------------
diro = "/nfs3m/archive/sfa_cache08/projects/input/dao_ops/ops/flk/tcvitals/text/TCVITALS/Y2022/M06/"
colo = ".syndata.tcvitals"

beg_date = datetime.datetime(year=2022, month=6, day=15, hour=0)
end_date = datetime.datetime(year=2022, month=6, day=19, hour=12)

delta_date = datetime.timedelta(hours=6)
cur_date = beg_date

lon_obs, lat_obs, slp_min_obs, tim_obs = [], [], [], []
while cur_date < end_date:
    print(cur_date)
    file = cur_date.strftime(diro+"gfs.%y%m%d.t%Hz"+colo)
    list_fileso.append(file)
    cur_date += delta_date
print(list_fileso)
for file in list_fileso:
    with open(file,'r') as f:
        for line in f:
            #print(line)
            cells = line.strip()
            vmax = float(cells[67:69])
            #if (("HOWARD" in line)&(vmax >20.0)) :
            if ("BLAS" in line):
                yyyymmdd = cells[19:27]
                print(yyyymmdd)
                hh = cells[28:30]
                tim = yyyymmdd+hh
                lat = cells[33:37]
                if ((lat.endswith('N'))==True):
                    lat=float(lat[0:3])*0.1
                elif ((lat.endswith('S'))==True):
                    lat=float(lat[0:3])*-0.1
                lon = cells[38:43]
                if ((lon.endswith('E'))==True):
                    lon=float(lon[0:4])*0.1
                elif ((lon.endswith('W'))==True):
                    lon=float(lon[0:4])*-0.1
                pres = float(cells[52:56])

                slp_min_obs.append(pres)
                lon_obs.append(lon)
                lat_obs.append(lat)
                tim_obs.append(tim)

            

#print(tim_obs,lat_obs,lon_obs,slp_min_obs)
dfo = pd.DataFrame((zip(tim_obs,lat_obs,lon_obs,slp_min_obs)), columns = ['c1','c2','c3','c4'])
print(dfo)
#----------------------------------------
idx_obs,idx_all=[],[]

trk_err_all,trk_tim_all,trk_lat_all,trk_lon_all,trk_slp_min_all,trk_fcst_all = [],[],[],[],[],[]
for idx1, val1 in enumerate(tim_obs):
  for idx2, val2 in enumerate(tim_all):
      if (tim_obs[idx1]==tim_all[idx2]):
          err=(distance(lat_all[idx2], lat_obs[idx1], lon_all[idx2], lon_obs[idx1]))*0.54
          trk_err_all.append(err)
          trk_tim_all.append(tim_all[idx2])
          trk_lat_all.append(lat_all[idx2])
          trk_lon_all.append(lon_all[idx2])
          trk_slp_min_all.append(slp_min_all[idx2])
          trk_fcst_all.append(fcst_all[idx2])
dft1 = pd.DataFrame((zip(trk_fcst_all,trk_err_all)), columns = ['FCST_HR','15/00Z'])
#print(dft1)

trk_err_all1,trk_tim_all1,trk_lat_all1,trk_lon_all1,trk_slp_min_all1,trk_fcst_all1 = [],[],[],[],[],[]
for idx1, val1 in enumerate(tim_obs):
  for idx2, val2 in enumerate(tim_all1):
      if (tim_obs[idx1]==tim_all1[idx2]):
          err=(distance(lat_all1[idx2], lat_obs[idx1], lon_all1[idx2], lon_obs[idx1]))*0.54
          trk_err_all1.append(err)
          trk_tim_all1.append(tim_all1[idx2])
          trk_lat_all1.append(lat_all1[idx2])
          trk_lon_all1.append(lon_all1[idx2])
          trk_slp_min_all1.append(slp_min_all1[idx2])
          trk_fcst_all1.append(fcst_all1[idx2])
#dft2 = pd.DataFrame((zip(trk_tim_all1,trk_lat_all1,trk_lon_all1,trk_slp_min_all1,trk_err_all1)), columns = ['c1','c2','c3','c4','c5'])
dft2 = pd.DataFrame((zip(trk_fcst_all1,trk_err_all1)), columns = ['FCST_HR','16/00Z'])


trk_err_all2,trk_tim_all2,trk_lat_all2,trk_lon_all2,trk_slp_min_all2,trk_fcst_all2 = [],[],[],[],[],[]
for idx1, val1 in enumerate(tim_obs):
  for idx2, val2 in enumerate(tim_all2):
      if (tim_obs[idx1]==tim_all2[idx2]):
          err=(distance(lat_all2[idx2], lat_obs[idx1], lon_all2[idx2], lon_obs[idx1]))*0.54
          trk_err_all2.append(err)
          trk_tim_all2.append(tim_all2[idx2])
          trk_lat_all2.append(lat_all2[idx2])
          trk_lon_all2.append(lon_all2[idx2])
          trk_slp_min_all2.append(slp_min_all2[idx2])
          trk_fcst_all2.append(fcst_all2[idx2])
#dft3 = pd.DataFrame((zip(trk_tim_all2,trk_lat_all2,trk_lon_all2,trk_slp_min_all2,trk_err_all2)), columns = ['c1','c2','c3','c4','c5'])
dft3 = pd.DataFrame((zip(trk_fcst_all2,trk_err_all2)), columns = ['FCST_HR','17/00Z'])
#print(dft3)

trk_err_all3,trk_tim_all3,trk_lat_all3,trk_lon_all3,trk_slp_min_all3,trk_fcst_all3 = [],[],[],[],[],[]
for idx1, val1 in enumerate(tim_obs):
  for idx2, val2 in enumerate(tim_all3):
      if (tim_obs[idx1]==tim_all3[idx2]):
          err=(distance(lat_all3[idx2], lat_obs[idx1], lon_all3[idx2], lon_obs[idx1]))*0.54
          trk_err_all3.append(err)
          trk_tim_all3.append(tim_all3[idx2])
          trk_lat_all3.append(lat_all3[idx2])
          trk_lon_all3.append(lon_all3[idx2])
          trk_slp_min_all3.append(slp_min_all3[idx2])
          trk_fcst_all3.append(fcst_all3[idx2])
dft4 = pd.DataFrame((zip(trk_fcst_all3,trk_err_all3)), columns = ['FCST_HR','18/00Z'])

summarydf = pd.concat([dft1,dft2['16/00Z'],dft3['17/00Z'],dft4['18/00Z']], axis=1)
print(summarydf)
with open('summary_4allstats_Blas.txt', 'w') as f:
    f.write(summarydf.to_string())

summarydf['mean'] = summarydf[['15/00Z','16/00Z','17/00Z','18/00Z']].mean(axis=1)
with open('summary_4onestats_Blas.txt', 'w') as f:
    f.write(summarydf.to_string())
summarydf['stddev'] = summarydf[['15/00Z','16/00Z','17/00Z','18/00Z']].std(axis=1)
summarydf['count'] = summarydf[['15/00Z','16/00Z','17/00Z','18/00Z']].count(axis=1)
summarydf['obscnt'] = summarydf['FCST_HR'].astype(int).astype(str) +"("+ summarydf["count"].astype(str)+")"
print(summarydf)

#ax=summarydf.iloc[0:, :].plot(x='FCST_HR',y='mean',yerr='stddev',capsize=3,xticks=summarydf.iloc[0:, :]['FCST_HR'])
#ax.fill_between(summarydf.iloc[0:, :]['FCST_HR'], summarydf.iloc[0:, :]['mean'] - summarydf.iloc[0:, :]['stddev'],summarydf.iloc[0:, :]['mean'] + summarydf.iloc[0:, :]['stddev'], alpha=0.35)
#ax.set_xticklabels(summarydf.iloc[0:, :]['obscnt'])
#ax.set_xlabel('FCST_HR(Data Count)')
#ax.set_ylabel('Nautical Miles')
#ax.legend(['Mean Track Error'])
#ax.set_title('TC Howard: 5 - 11 Aug 2022')
#plt.savefig('TCHoward2022_trcerr.png')
#plt.show()
#sys.exit()

##TESTING OUT
##ax1 = ax.twinx()
##summarydf.iloc[1:, :].plot('FCST_HR','count',ax=ax1, color='r')
#----------------------------------------


sys.exit()







