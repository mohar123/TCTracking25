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
import matplotlib.ticker as mticker

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

#---------------------------#TC TRACKS CALCULATION#-----------------------------

#---------------------------#TC VITALS#-----------------------------

diro = "/nfs3m/archive/sfa_cache08/projects/input/dao_ops/ops/flk/tcvitals/text/TCVITALS/Y2024/"
colo = ".syndata.tcvitals"

list_files = list()
list_fileso = list()

beg_date = datetime.datetime(year=2024, month=9, day=25, hour=0)
end_date = datetime.datetime(year=2024, month=9, day=27, hour=18)

delta_date = datetime.timedelta(hours=6)
cur_date = beg_date

lon_obs, lat_obs, slp_min_obs, tim_obs = [], [], [], []
while cur_date < end_date:
    print(cur_date)
    file = cur_date.strftime(diro+"M%m/gfs.%y%m%d.t%Hz"+colo)
    list_fileso.append(file)
    cur_date += delta_date
print(list_fileso)
for file in list_fileso:
    with open(file,'r') as f:
        for line in f:
            #print(line)
            cells = line.strip()
            if "HELENE" in line:
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
#sys.exit()
#############TC - 
#---
#ONE
#___

# Obtain the date/time and initial lat/lon from command line:
dirm = "/home/gmao_ops/f5295_fp/run/.../archive/forecast/Y2024/M09/D25/H00/"
col = "GEOS.fp.fcst.inst3_3d_asm_Np.20240925_00+"
beg_date = datetime.datetime(year=2024, month=9, day=25, hour=0)
end_date = datetime.datetime(year=2024, month=9, day=27, hour=18)

lon_all1, lat_all1, slp_min_all1, tim_all1, fcst_all1 = [], [], [], [], []
del_lat = 5.0
del_lon = 5.0
sel_lat_lon = (dfo[['c1','c2','c3']][dfo['c1']== beg_date.strftime("%Y%m%d%H")])
print(sel_lat_lon.iloc[0]['c2'])
lat_slice1 = (sel_lat_lon.iloc[0]['c2'])
lat_slice2 = lat_slice1 + 10.0
print(sel_lat_lon.iloc[0]['c3'])
lon_slice2 = (sel_lat_lon.iloc[0]['c3'])
lon_slice1 = lon_slice2 + (-3.)

delta_date = datetime.timedelta(hours=6)

list_files = list()
list_fileso = list()

cur_date = beg_date

while cur_date < end_date:
    file = cur_date.strftime(dirm+col+"%Y%m%d_%H%M.V01.nc4")
    list_files.append(file)
    cur_date += delta_date

#lat_slice1 = 8.5
#lat_slice2 = 16.5 
#lon_slice1 = -64.5
#lon_slice2 = -42.

for file in list_files:
    with xr.open_dataset(file, engine='netcdf4') as f:  
        
        slp_1 = f.SLP.sel(lat=slice(lat_slice1,lat_slice2),lon=slice(lon_slice1,lon_slice2))
        slp_1_min = float(slp_1.min())*0.01
        loc1 = np.where(slp_1[0,:,:] == np.amin(slp_1[0,:,:]))
        lat1 = float(slp_1.lat[loc1[0][0]].values)
        lon1 = float(slp_1.lon[loc1[1][0]].values)
        tim1 = (slp_1.time.values) 

# tim1 is a yyyymmdd(64bit data)
        ds = str(tim1[0]).partition('T')[0].split('-')
# when in doubt about the data type always print(type(varname)) 
        #print(ds,type(ds))

        time_str = str(tim1[0]).partition('T')[2][0:2]
# when in doubt about the data type always print(type(varname)) 
        #print(time_str,type(time_str))

        ds.append(time_str)

        d = datetime.datetime(int(ds[0]), int(ds[1]), int(ds[2]), int(ds[3]))
        ds1 = d.strftime("%Y%m%d%H")
        fcst_diff1 = (d-beg_date).total_seconds() / 3600
#list if lat >=dely ....
        
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

        lat_slice1 = lat_slice1+.5
        lat_slice2 = lat_slice2+.5
        lon_slice1 = lon_slice1+.5
        lon_slice2 = lon_slice2+.5
        #print(lat_slice1, lat_slice2, lon_slice1, lon_slice2)
#print(tim_all1,lat_all1,lon_all1,slp_min_all1)
df1 = pd.DataFrame((zip(tim_all1,lat_all1,lon_all1,slp_min_all1,fcst_all1)), columns = ['c1','c2','c3','c4','c5'])
print(df1)
#sys.exit()
#------------------------------
#---
#TWO
#___
# Obtain the date/time and initial lat/lon from command line:
dirm = "/home/gmao_ops/f5295_fp/run/.../archive/forecast/Y2024/M09/D26/H00/"
col = "GEOS.fp.fcst.inst3_3d_asm_Np.20240926_00+"
beg_date = datetime.datetime(year=2024, month=9, day=26, hour=0)
end_date = datetime.datetime(year=2024, month=9, day=27, hour=18)

lon_all2, lat_all2, slp_min_all2, tim_all2, fcst_all2 = [], [], [], [], []
del_lat = 5.0
del_lon = 5.0
sel_lat_lon = (dfo[['c1','c2','c3']][dfo['c1']== beg_date.strftime("%Y%m%d%H")])
print(sel_lat_lon.iloc[0]['c2'])
lat_slice1 = (sel_lat_lon.iloc[0]['c2'])
lat_slice2 = lat_slice1 + 10.
print(sel_lat_lon.iloc[0]['c3'])
lon_slice2 = (sel_lat_lon.iloc[0]['c3'])
lon_slice1 = lon_slice2 + (-3.)

delta_date = datetime.timedelta(hours=6)

list_files = list()
list_fileso = list()

cur_date = beg_date

while cur_date < end_date:
    file = cur_date.strftime(dirm+col+"%Y%m%d_%H%M.V01.nc4")
    list_files.append(file)
    cur_date += delta_date

#print(list_files)
#lat_slice1 = 9.5
#lat_slice2 = 20.
#lon_slice1 = -69.
#lon_slice2 = -49.
for file in list_files:
    with xr.open_dataset(file, engine='netcdf4') as f:  
        
        slp_1 = f.SLP.sel(lat=slice(lat_slice1,lat_slice2),lon=slice(lon_slice1,lon_slice2))
        slp_1_min = float(slp_1.min())*0.01
        loc1 = np.where(slp_1[0,:,:] == np.amin(slp_1[0,:,:]))
        lat1 = float(slp_1.lat[loc1[0][0]].values)
        lon1 = float(slp_1.lon[loc1[1][0]].values)
        tim1 = (slp_1.time.values) 

# tim1 is a yyyymmdd(64bit data)
        ds = str(tim1[0]).partition('T')[0].split('-')
# when in doubt about the data type always print(type(varname)) 
        #print(ds,type(ds))

        time_str = str(tim1[0]).partition('T')[2][0:2]
# when in doubt about the data type always print(type(varname)) 
        #print(time_str,type(time_str))

        ds.append(time_str)

        d = datetime.datetime(int(ds[0]), int(ds[1]), int(ds[2]), int(ds[3]))
        ds1 = d.strftime("%Y%m%d%H")
        fcst_diff1 = (d-beg_date).total_seconds() / 3600
#list if lat >=dely ....

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
            
        lat_slice1 = lat_slice1+0.8
        lat_slice2 = lat_slice2+0.8
        lon_slice1 = lon_slice1+.5
        lon_slice2 = lon_slice2+.5
       
df2 = pd.DataFrame((zip(tim_all2,lat_all2,lon_all2,slp_min_all2,fcst_all2)), columns = ['c1','c2','c3','c4','c5'])
print(df2)
#sys.exit()
#------------------------------
#---
#THREE
#___
# Obtain the date/time and initial lat/lon from command line:
dirm = "/home/gmao_ops/f5295_fp/run/.../archive/forecast/Y2024/M09/D27/H00/"
col = "GEOS.fp.fcst.inst3_3d_asm_Np.20240927_00+"
beg_date = datetime.datetime(year=2024, month=9, day=27, hour=0)
end_date = datetime.datetime(year=2024, month=9, day=27, hour=18)

lon_all3, lat_all3, slp_min_all3, tim_all3, fcst_all3 = [], [], [], [], []
del_lat = 10.
del_lon = 5.0
sel_lat_lon = (dfo[['c1','c2','c3']][dfo['c1']== beg_date.strftime("%Y%m%d%H")])
print(sel_lat_lon.iloc[0]['c2'])
lat_slice1 = (sel_lat_lon.iloc[0]['c2'])
lat_slice2 = lat_slice1 + 5.0
print(sel_lat_lon.iloc[0]['c3'])
lon_slice2 = (sel_lat_lon.iloc[0]['c3'])
lon_slice1 = lon_slice1 + (-5.)

delta_date = datetime.timedelta(hours=6)

list_files = list()
list_fileso = list()

cur_date = beg_date

while cur_date < end_date:
    file = cur_date.strftime(dirm+col+"%Y%m%d_%H%M.V01.nc4")
    list_files.append(file)
    cur_date += delta_date

#print(list_files)
#lat_slice1 = 10.5
#lat_slice2 = 20.5
#lon_slice1 = -90
#lon_slice2 = -56
for file in list_files:
    with xr.open_dataset(file, engine='netcdf4') as f:  
        
        slp_1 = f.SLP.sel(lat=slice(lat_slice1,lat_slice2),lon=slice(lon_slice1,lon_slice2))
        slp_1_min = float(slp_1.min())*0.01
        loc1 = np.where(slp_1[0,:,:] == np.amin(slp_1[0,:,:]))
        lat1 = float(slp_1.lat[loc1[0][0]].values)
        lon1 = float(slp_1.lon[loc1[1][0]].values)
        tim1 = (slp_1.time.values) 

# tim1 is a yyyymmdd(64bit data)
        ds = str(tim1[0]).partition('T')[0].split('-')
# when in doubt about the data type always print(type(varname)) 
        #print(ds,type(ds))

        time_str = str(tim1[0]).partition('T')[2][0:2]
# when in doubt about the data type always print(type(varname)) 
        #print(time_str,type(time_str))

        ds.append(time_str)

        d = datetime.datetime(int(ds[0]), int(ds[1]), int(ds[2]), int(ds[3]))
        ds1 = d.strftime("%Y%m%d%H")
        fcst_diff1 = (d-beg_date).total_seconds() / 3600
#list if lat >=dely ....

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
            
        lat_slice1 = lat_slice1+.8
        lat_slice2 = lat_slice2+.8
        lon_slice1 = lon_slice1+.5
        lon_slice2 = lon_slice2+.5
        #print(lat_slice1, lat_slice2, lon_slice1, lon_slice2)

df3 = pd.DataFrame((zip(tim_all3,lat_all3,lon_all3,slp_min_all3,fcst_all3)), columns = ['c1','c2','c3','c4','c5'])
print(df3)
#sys.exit()

#---------------------------------------------------------

idx_obs,idx_all=[],[]
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
dft1 = pd.DataFrame((zip(trk_fcst_all1,trk_err_all1)), columns = ['FCST_HR','25/00Z'])
#print(dft2)
#sys.exit()

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
dft2 = pd.DataFrame((zip(trk_fcst_all2,trk_err_all2)), columns = ['FCST_HR','26/00Z'])

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
dft3 = pd.DataFrame((zip(trk_fcst_all3,trk_err_all3)), columns = ['FCST_HR','27/00Z'])

#---------------------#SUMMARY#----------------------
summarydf = pd.concat([dft1['FCST_HR'],dft1['25/00Z'],dft2['26/00Z'],dft3['27/00Z']], axis=1)
print(summarydf)
#sys.exit()
#----------------------------------------
#STAS CALC
#----------------------------------------

summarydf = pd.concat([dft1,dft2['26/00Z'],dft3['27/00Z']], axis=1)
print(summarydf)
with open('summary_4allstats_Helene.txt', 'w') as f:
    f.write(summarydf.to_string())
summarydf['mean'] = summarydf[['25/00Z','26/00Z','27/00Z']].mean(axis=1)
with open('summary_4onestats_Helene.txt', 'w') as f:
    f.write(summarydf.to_string())
summarydf['stddev'] = summarydf[['25/00Z','26/00Z','27/00Z']].std(axis=1)
summarydf['count'] = summarydf[['25/00Z','26/00Z','27/00Z']].count(axis=1)
summarydf['obscnt'] = summarydf['FCST_HR'].astype(int).astype(str) +"("+ summarydf["count"].astype(str)+")"
print(summarydf)


sys.exit()







