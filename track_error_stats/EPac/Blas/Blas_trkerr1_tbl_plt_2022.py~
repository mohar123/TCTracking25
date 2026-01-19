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

# Obtain the date/time and initial lat/lon from command line:
dirm = "/archive/u/dao_ops/f5294_fp/forecast/Y2022/M06/D15/H00/"
col = "GEOS.fp.fcst.inst3_3d_asm_Np.20220615_00+"
beg_date = datetime.datetime(year=2022, month=6, day=15, hour=00)
end_date = datetime.datetime(year=2022, month=6, day=18, hour=00)

delta_date = datetime.timedelta(hours=6)

list_files = list()
list_fileso = list()

cur_date = beg_date

while cur_date < end_date:
    file = cur_date.strftime(dirm+col+"%Y%m%d_%H%M.V01.nc4")
    list_files.append(file)
    cur_date += delta_date

print(list_files)
lon_all, lat_all, slp_min_all, tim_all = [], [], [], []
del_lat = 1.
del_lon = 1.5

for file in list_files:
    with xr.open_dataset(file, engine='netcdf4') as f:  
        
        #print(f.keys())
        slp_1 = f.SLP.sel(lat=slice(14.5,20.),lon=slice(-104., -101.5))
        slp_1_min = float(slp_1.min())
        loc1 = np.where(slp_1[0,:,:] == np.amin(slp_1[0,:,:]))
        lat1 = float(slp_1.lat[loc1[0]].values)
        lon1 = float(slp_1.lon[loc1[1]].values)
        tim1 = (slp_1.time.values) 
        #tim1 = datetime.datetime.strftime(datetime.datetime.strptime(np.array2string(slp_1.time.values),"%Y-%m-%dT%H:%M:%S.%f"))
        #print(tim1[0],type(tim1[0]),str(tim1[0]).partition('T')[0])
        ds = str(tim1[0]).partition('T')[0].split('-')
        #print(date_str,type(date_str))
        time_str = str(tim1[0]).partition('T')[2][0:2]
        #print(time_str,type(time_str))
        ds.append(time_str)
        #print(date_str)
        d = datetime.datetime(int(ds[0]), int(ds[1]), int(ds[2]), int(ds[3]))
        ds1 = d.strftime("%Y%m%d%H")
        #print(d.strftime("%Y%m%d%H"))
#list if lat >=dely ....

        if lat_all:
           if (lat1 <= lat_all[-1]+del_lat) and (lat1 >= lat_all[-1]-del_lat) and \
              (lon1 <= lon_all[-1]+del_lon) and (lon1 >= lon_all[-1]-del_lon):
              lat_all.append(lat1)
              slp_min_all.append(slp_1_min)
              lon_all.append(lon1)
              #tim_all.append(tim1)
              tim_all.append(ds1)
        else:
            slp_min_all.append(slp_1_min)
            lon_all.append(lon1)
            lat_all.append(lat1)
            #tim_all.append(tim1)
            tim_all.append(ds1)
        
#print(tim_all,lat_all,lon_all,slp_min_all)
df1 = pd.DataFrame((zip(tim_all,lat_all,lon_all,slp_min_all)), columns = ['c1','c2','c3','c4'])
print(df1)
#sys.exit()
#------------------------------
## Obtain the date/time and initial lat/lon from command line:
#dirm = "/archive/u/dao_ops/f5294_fp/forecast/Y2022/M05/D29/H00/"
#col = "GEOS.fp.fcst.inst3_3d_asm_Np.20220529_00+"
#beg_date = datetime.datetime(year=2022, month=5, day=29, hour=0)
#end_date = datetime.datetime(year=2022, month=5, day=30, hour=18)

#delta_date = datetime.timedelta(hours=6)

#list_files = list()
#list_fileso = list()

#cur_date = beg_date

#while cur_date < end_date:
#    file = cur_date.strftime(dirm+col+"%Y%m%d_%H%M.V01.nc4")
#    list_files.append(file)
#    cur_date += delta_date

#print(list_files)
#lon_a06, lat_a06, slp_min_a06, tim_a06 = [], [], [], []
#for file in list_files:
#    with xr.open_dataset(file, engine='netcdf4') as f:
        #print(f.keys())
#        slp_1 = f.SLP.sel(lat=slice(10.,20.),lon=slice(-103.,-93.))
#        slp_1_min = float(slp_1.min())
#        loc1 = np.where(slp_1[0,:,:] == np.amin(slp_1[0,:,:]))
#        lat1 = float(slp_1.lat[loc1[0]].values)
#        lon1 = float(slp_1.lon[loc1[1]].values)
#        tim1 = (slp_1.time.values)
#        slp_min_a06.append(slp_1_min)
#        lon_a06.append(lon1)
#        lat_a06.append(lat1)
#        tim_a06.append(tim1)
        
#print(tim_a06,lat_a06,lon_a06,slp_min_a06)
#df2 = pd.DataFrame((zip(tim_a06,lat_a06,lon_a06,slp_min_a06)), columns = ['c1','c2','c3','c4'])
#print(df2)

#------------------------------
diro = "/nfs3m/archive/sfa_cache08/projects/input/dao_ops/ops/flk/tcvitals/text/TCVITALS/Y2022/M06/"
colo = ".syndata.tcvitals"

beg_date = datetime.datetime(year=2022, month=6, day=15, hour=0)
end_date = datetime.datetime(year=2022, month=6, day=15, hour=12)

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
            cells = line.split( "  " )
            #tmp = cells[3]
            print(cells[4])
            yyyymmdd = (cells[4][0:8])
            hh = (cells[4][9:11])
            tim = yyyymmdd+hh
            lat = (cells[4][14:18])
            if ((lat.endswith('N'))==True):
                lat=float(lat[0:3])*0.1
            elif ((lat.endswith('S'))==True):
                lat=float(lat[0:3])*-0.1
            print(lat)
            lon = (cells[4][19:24])
            if ((lon.endswith('E'))==True):
                lon=float(lon[0:4])*0.1
            elif ((lon.endswith('W'))==True):
                lon=float(lon[0:4])*-0.1
            print(lon)
            pres = float(cells[4][33:37]) 
            
            slp_min_obs.append(pres)
            lon_obs.append(lon)
            lat_obs.append(lat)
            tim_obs.append(tim)

#print(tim_obs,lat_obs,lon_obs,slp_min_obs)
dfo = pd.DataFrame((zip(tim_obs,lat_obs,lon_obs,slp_min_obs)), columns = ['c1','c2','c3','c4'])
print(dfo)

#----------------------------------------
trk_err_all = []
for i in range(0,len(lat_obs)):
    print(tim_obs[i])
    d = datetime.datetime.strptime(tim_obs[i],"%Y%m%d%H")
    print(d.strftime("%Y-%m-%d-%H:%M:%S"))
    if (tim_obs[i] == tim_all[i]):
       print(distance(lat_all[i], lat_obs[i], lon_all[i], lon_obs[i]))
       err = distance(lat_all[i], lat_obs[i], lon_all[i], lon_obs[i])
       trk_err_all.append(err)

print(trk_err_all)

df1 = pd.DataFrame((zip(tim_all,lat_all,lon_all,slp_min_all,trk_err_all)), columns = ['c1','c2','c3','c4','c5'])
print(df1)
#----------------------------------------
wlon,elon,nlat,slat = -105,-97,21,13

#borders = cfeature.NaturalEarthFeature(category='cultural',name='admin_1_states_provinces_lakes_shp',scale= '50m',facecolor='none',edgecolor='gray')

fig = plt.figure(figsize=(12, 9))
map_projection = ccrs.PlateCarree()
data_transform = ccrs.PlateCarree()

ax = plt.axes(projection=map_projection)
ax.stock_img()
#ax.add_feature(cfeature.LAND, zorder=100, edgecolor='k')
ax.set_extent(((wlon-0.01),(elon+0.01),(slat-0.01),(nlat+0.01)), crs=ccrs.PlateCarree())
ax.coastlines(resolution='50m',              linewidth=0.5)
#ax.add_feature(borders, facecolor='#e0e0e0', linewidth=0.5)
ax.plot([lon_obs], [lat_obs], 'r*', transform=data_transform,color="purple", markersize=8)
ax.plot([lon_all], [lat_all], 'r*', transform=data_transform,color="green", markersize=8)
#ax.plot([lon_a06], [lat_a06], 'r*', transform=data_transform,color="blue", markersize=8)

for i,j,k in zip(lat_all,lon_all,trk_err_all):
    ax.annotate('{0:.2f}'.format(k),xy=(j,i),xytext=(j+0.2,i),color="green",fontsize=8,horizontalalignment='left', verticalalignment='top')

ax.set_title('TC Track : Blas 2022')

p1, = ax.plot(lon_obs,lat_obs, label="TC Vitals",transform=data_transform,color="purple")
p2, = ax.plot(lon_all,lat_all, label="Fcst at 15-00Z",transform=data_transform,color="green")
#ax.plot(lon_a06,lat_a06, transform=data_transform,color="blue")

ax.legend(handles=[p1,p2])

gl = ax.gridlines(draw_labels=True)
gl.xlabels_top = False
gl.ylabels_left = False
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER

plt.savefig('TCBlas1_2022_3.png')
plt.show()




sys.exit()







