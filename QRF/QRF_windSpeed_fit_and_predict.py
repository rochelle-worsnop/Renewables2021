from netCDF4 import Dataset, stringtochar, num2date, date2num, chartostring
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from skgarden import RandomForestQuantileRegressor
from datetime import datetime, timedelta
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import cartopy.feature as cfeature
import copy
from numpy import ma
import math
from scipy import stats
import properscoring as ps
import pandas as pd
import os

try:
    from numba import njit
except ImportError: 
    raise ImportError("Numba is not installed. If its installation can't be done, remove the decorator '@njit' in the functions where it's used. The code will be slower.")


## For whatever reason, to be able to open netCDF files on /Projects/era5/ I need to set the environment variable 'HDF5_USE_FILE_LOCKING' to 'FALSE':
## (it seems that not everybody needs to do that, so feel free to test without)
##   - If this script is run in a Jupyter Lab kernel, uncomment the following line:
# %env HDF5_USE_FILE_LOCKING=FALSE
##   - If this script is executed with the python command, for the first time:
##      1. create a .bashrc file in the same location as this script: $ nano .bashrc
##      2. Add in this file the line: export HDF5_USE_FILE_LOCKING='FALSE'
##      3. Execute the command with: $ source .bashrc


# ## User inputs, that are provided in the launch script:
# #-------------------------------------------
# input_path_GEFSv12 = '/Projects/GEFSv12_RENEWABLES/'
# input_path_ERA5 = '/Projects/era5/'

# output_path_QRF_fcsts = '/volumes/DataCaddy/Joseph_stuff/Project_Renewable_energy/QRF_fcsts/windSpeed/'

# bounds_area = [40, 40.25, -105, -104.75]   # small rectangle around Boulder
# month = 1                                  # from 1 to 12
# lead = 72                                  # from 3 to 240 (Day1-10 only), by 3h interval
# avail_years = range(2000,2020)             # (if we want to use all years)
# nens = 50                                  # size of the QRF and climatological ensemble forecasts (can be different from the size of the raw GEFS, i.e. 5 members)

# ## QRF hyperparameters:
# n_estimators = 1000     # Number of trees in the forest. Corresponds to 'ntree' in the R package quantregForest, where the description is: Number of trees to grow
# min_samples_leaf = 20   # Minimum number of samples required to be at a leaf node. Corresponds to 'nodesize' in the R package quantregForest, where the description is: Minimum size of terminal nodes
# max_features = 2        # Number of features to consider when looking for the best split. Corresponds to 'mtry' in the R package quantregForest, where the description is: Number of variables randomly sampled as candidates at each split

# seed = 0                # Random seed to reproduce the experiment
# #-------------------------------------------





nday = [31,29,31,30,31,30,31,31,30,31,30,31][month-1]   # number of days for the month selected. The code might need to be adapted for February...
name_month = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'][month-1]

nyear = len(avail_years)
ntime = nday * nyear



## Functions to load the GEFS and ERA5 data from netCDFs, over a limited area:
#-------------------------------------------
def load_data_GEFS(ncfile, name_var, id_lats_gefs, id_lons_gefs, lead):
    nc = Dataset(ncfile)
    id_lead = np.where(nc.variables['fhour'][:] == lead)[0][0]
    data = nc.variables[name_var][0,id_lead,:,:,:][:,id_lats_gefs,:][:,:,id_lons_gefs]
    nc.close()
    return np.moveaxis(data,0,-1)    # we put the ensemble dimension last


def get_latLon_GEFS(ncfile, bounds_area, return_id_latLon=False):
    lb_lat, ub_lat, lb_lon, ub_lon = bounds_area
    nc = Dataset(ncfile)
    id_lons_gefs = np.where(np.logical_and(nc.variables['lon'][:] >= lb_lon, nc.variables['lon'][:] <= ub_lon))[0]
    id_lats_gefs = np.where(np.logical_and(nc.variables['lat'][:] >= lb_lat, nc.variables['lat'][:] <= ub_lat))[0]
    lons_gefs = nc.variables['lon'][:][id_lons_gefs]
    lats_gefs = nc.variables['lat'][:][id_lats_gefs]
    nc.close()
    if return_id_latLon == True:
        return lats_gefs, lons_gefs, id_lats_gefs, id_lons_gefs
    else:
        return lats_gefs, lons_gefs

    
def load_data_ERA5(ncfile, name_var, id_lats_era5, id_lons_era5, yyyymmddhh_valid):
    nc = Dataset(ncfile)
    yyyymmddhh_valid_ERA5 = np.array([date_to_yyyymmddhh(num2date(nc.variables['time'][i], units=nc.variables['time'].units, calendar='gregorian')) for i in range(nc.dimensions['time'].size)])
    id_dates = fun_matchIndex(yyyymmddhh_valid.ravel(), yyyymmddhh_valid_ERA5)
    data =  nc.variables[name_var][:,id_lats_era5,:][:,:,id_lons_era5][id_dates.mask==False,:,:]
    nc.close()
    return data, id_dates


def get_latLon_ERA5(ncfile, bounds_area, return_id_latLon=False):
    lb_lat, ub_lat, lb_lon, ub_lon = bounds_area
    nc = Dataset(ncfile)
    lons_era5 = nc.variables['lon'][:]
    lons_era5[lons_era5>180] = lons_era5[lons_era5>180] - 360   # The longitude in ERA5 are stored 0 to 360, while we use the convention -180 to 180
    lats_era5 = nc.variables['lat'][:]
    id_lons_era5 = np.where(np.logical_and(lons_era5 >= lb_lon, lons_era5 <= ub_lon))[0]
    id_lats_era5 = np.where(np.logical_and(lats_era5 >= lb_lat, lats_era5 <= ub_lat))[0]
    lons_era5 = lons_era5[id_lons_era5]
    lats_era5 = lats_era5[id_lats_era5]
    if return_id_latLon == True:
        return lats_era5, lons_era5, id_lats_era5, id_lons_era5
    else:
        return lats_era5, lons_era5
        
        
        
## Functions to compute ensemble statistics (that will be used as predictors in the QRF) from the raw GEFS forecasts:
#-------------------------------------------
def ensemble_statistics(arr, axis):
    name_ensStats = ['mean','ctrl','median','std','MD','Q10','Q90',]
    ens_mean = np.mean(arr, axis=axis)
    ctrl = np.moveaxis(arr, axis, 0)[0,...]
    ens_std = np.std(arr, axis=axis)
    ens_md = MD(arr, axis=axis)    # ensemble mean difference (alternative measure of the dispersion, less sensitive to outliers)
    firstDec, median, ninthDec = [np.quantile(arr, q=q, axis=axis) for q in [0.1,0.5,0.9]]
    return np.stack((ens_mean, ctrl, median, ens_std, ens_md, firstDec, ninthDec), axis=-1), name_ensStats   # we put the ensemble statistics dimension last


## Function to compute the ensemble mean difference (the core is in Numba, but a wrapper is needed as np.moveaxis is not supported in Numba):
def MD(arr, axis):
    arr = np.moveaxis(arr, axis, -1)
    ori_shape = arr.shape
    arr = arr.reshape(-1, arr.shape[-1])
    md = MD_njit(arr)
    return md.reshape(ori_shape[:-1])
    
@njit
def MD_njit(arr):
    md = np.zeros(arr.shape[0])
    for i in range(arr.shape[0]):
        ens = arr[i,:]
        md[i] = np.sum(np.abs(np.expand_dims(ens,-1) - np.transpose(ens))) / arr.shape[1]**2
    return md



## Other necessary functions:
#-------------------------------------------
def date_to_yyyymmddhh(dt_time):
    return 1000000*dt_time.year + 10000*dt_time.month + 100*dt_time.day + dt_time.hour

def yyyymmddhh_to_time(yyyymmddhh, time_unit='hours since 1900-01-01 00:00:00', time_calendar='gregorian'):
    dates = pd.to_datetime(yyyymmddhh.astype(str), format='%Y%m%d%H')
    return date2num(dates.tolist(), time_unit, time_calendar)

def fun_matchIndex(X, Y):
    ## Function that returns a masked vector with same size as Y, which contains
    ##   the position in X of each element of Y. If an element of Y is not in X, the
    ##   returned value is masked.
    ##   Exemple:
    ##       X = np.array([3,5,7,1,9,8,12,11])
    ##       Y = np.array([2,1,5,10])
    ##       id_match
    ##       >>> masked_array(data=[--, 3, 1, --])
    
    index = np.argsort(X)
    sorted_X = X[index]
    sorted_index = np.searchsorted(sorted_X, Y)
    Yindex = np.take(index, sorted_index, mode="clip")
    mask = X[Yindex] != Y
    id_match = np.ma.array(Yindex, mask=mask)
    return id_match








## Load the GEFS forecasts, of wind speed, but also of the meteorological predictors:
#-------------------------------------------
lats_gefs, lons_gefs, id_lats_gefs, id_lons_gefs = get_latLon_GEFS(input_path_GEFSv12+'u100m/2019/ugrd_hgt_2019010100_Days1-10.nc' , bounds_area, return_id_latLon=True)   # date doesn't matter
nx, ny = lons_gefs.size, lats_gefs.size

yyyymmddhh_init = np.zeros((nyear,nday), dtype=np.int32)
yyyymmddhh_valid = np.zeros((nyear,nday), dtype=np.int32)
rawGEFS = ma.array(np.zeros((nyear,nday,ny,nx,5)), mask=True)
nmetPred = 1     # we only load one meteorological predictor here, the mean sea level pressure
metPred = ma.array(np.zeros((nyear,nday,ny,nx,nmetPred)), mask=True)

for y in range(nyear):
    year = avail_years[y]
    print(name_month+'; lead '+str(lead)+'h; Loading GEFS for year '+str(year)+'...')
    for d in range(nday):
        day = d+1
        
        yyyymmddhh_init[y,d] = date_to_yyyymmddhh(datetime(year,month,day, 0, 0))
        yyyymmddhh_valid[y,d] = date_to_yyyymmddhh(datetime(year,month,day, 0, 0) + pd.Timedelta(hours=lead))
        
        ## Raw GEFS forecast of the predictant (here wind speed, which is sqrt(u^2 +v^2)):
        ncfile_U = input_path_GEFSv12 + 'u100m/'+str(year)+'/ugrd_hgt_'+str(yyyymmddhh_init[y,d])+'_Days1-10.nc'
        ncfile_V = input_path_GEFSv12 + 'v100m/'+str(year)+'/vgrd_hgt_'+str(yyyymmddhh_init[y,d])+'_Days1-10.nc'
        rawGEFS[y,d,:,:,:] = np.sqrt(np.square(load_data_GEFS(ncfile_U,'100u', id_lats_gefs, id_lons_gefs, lead)) + np.square(load_data_GEFS(ncfile_V,'100v', id_lats_gefs, id_lons_gefs, lead)))

        ## Meteorological predictors from GEFS (for each, we take the ensemble mean):
        ##---------------
        
        ## Predictor 1: Mean sea level pressure:
        ncfile_pres_msl = input_path_GEFSv12 + 'pres_msl/'+str(year)+'/pres_msl_'+str(yyyymmddhh_init[y,d])+'_Days1-10.nc'
        metPred[y,d,:,:,0] = np.mean(load_data_GEFS(ncfile_pres_msl,'msl', id_lats_gefs, id_lons_gefs, lead), axis=-1)

        ## Load more meteorological predictors here if needed
        
## We compute the ensemble statistics from the raw GEFS forecasts:
ensStats, name_ensStats = ensemble_statistics(rawGEFS, axis=-1)
nensStats = len(name_ensStats)

## The total number of predictors for the QRF will be:
npred = nensStats + nmetPred


## Load the ERA5 reanalysis of wind speed:
#-------------------------------------------
lats_era5, lons_era5, id_lats_era5, id_lons_era5 = get_latLon_ERA5(input_path_ERA5 + 'monolevel/vwnd.100m.2019.nc', bounds_area, return_id_latLon=True)   # date doesn't matter
assert np.all(lats_era5 == lats_gefs) and np.all(lons_era5 == lons_gefs)    # we make sure that ERA5 and GEFSv12 share the same grid 
del lats_era5, lons_era5

obsERA5 = ma.array(np.zeros((nyear*nday,ny,nx)), mask=True)

for y in range(nyear):
    year = avail_years[y]
    print(name_month+'; lead '+str(lead)+'h; Loading ERA5 for year '+str(year)+'...')

    ncfile_U = input_path_ERA5 + 'monolevel/uwnd.100m.'+str(year)+'.nc'
    ncfile_V = input_path_ERA5 + 'monolevel/vwnd.100m.'+str(year)+'.nc'
    U, id_dates = load_data_ERA5(ncfile_U, 'uwnd', id_lats_era5, id_lons_era5, yyyymmddhh_valid)
    V = load_data_ERA5(ncfile_V, 'vwnd', id_lats_era5, id_lons_era5, yyyymmddhh_valid)[0]   # (we don't need to save id_dates again)
    obsERA5[id_dates[id_dates.mask==False],:,:] = np.sqrt(np.square(U) + np.square(V))

obsERA5 = obsERA5.reshape(nyear,nday,ny,nx)



## Loop over the years, to fit a QRF model on the training data, and to make predictions on the validation data:
##    (we conduct a "leave-one-year-out" training/validation experiment)
#-------------------------------------------
ny_train = nyear - 1

## Create an instance of the QRF class:
rfqr = RandomForestQuantileRegressor(random_state=seed, n_estimators=n_estimators, min_samples_leaf=min_samples_leaf, max_features=max_features, verbose=0)

## We will store the QRF forecasts with the full ensemble and with only 5 members (as in the raw forecasts), and the climatological forecasts constructed form the data used for training the QRF:
fcstQRF = ma.array(np.zeros((nyear,nday,ny,nx,nens)), mask=True)
fcstQRF_5memb = ma.array(np.zeros((nyear,nday,ny,nx,5)), mask=True)
climato = ma.array(np.zeros((nyear,nday,ny,nx,nens)), mask=True)

## We will also store the predictor importance for each QRF fit:
pred_importance = ma.array(np.zeros((nyear,ny,nx,npred)), mask=True)

## Sampling probabilities for deriving ensemble members from the QRF
quantiles = np.arange(1,nens+1)/(nens+1) * 100
quantiles_5memb = np.arange(1,5+1)/(5+1) * 100

for y_valid in range(nyear):
    print(name_month+'; lead '+str(lead)+'h; Fitting QRF and making prediction for validation year '+str(avail_years[y_valid])+'...')
    
    y_train = np.delete(np.arange(nyear), y_valid)   # we remove the validation year from the training
    for j in range(ny):
        for i in range(nx):
            
            pred_train = np.concatenate((ensStats[y_train,:,j,i,:], metPred[y_train,:,j,i,:]), axis=-1).reshape(ny_train*nday,npred)
            pred_valid = np.concatenate((ensStats[y_valid,:,j,i,:], metPred[y_valid,:,j,i,:]), axis=-1).reshape(nday,npred)
            obsERA5_train = obsERA5[y_train,:,j,i].reshape(ny_train*nday)
            
            ## QRF:
            rfqr.fit(pred_train, obsERA5_train)
            fcstQRF[y_valid,:,j,i,:] = np.stack([rfqr.predict(pred_valid, quantile=quantile) for quantile in quantiles], axis=1)            
            fcstQRF_5memb[y_valid,:,j,i,:] = np.stack([rfqr.predict(pred_valid, quantile=quantile) for quantile in quantiles_5memb], axis=1)
            pred_importance[y_valid,j,i,:] = rfqr.feature_importances_
            
            ## Climatological forecast:
            climato[y_valid,:,j,i,:] = np.quantile(obsERA5_train, q=quantiles/100)
            
          
        
## Save the forecast fields into a NetCDF:
#-------------------------------------------
outfilename = output_path_QRF_fcsts+'test_QRF_fcst_windSpeed_month_'+str(month)+'_lead_'+str(lead)+'h.nc'
netcdf_long_name = 'GEFSv12 forecasts fields of wind speed post-processed with QRF, for '+name_month+'. and lead time '+str(lead)+'h'

nc = Dataset(outfilename,'w',format='NETCDF4_CLASSIC')

## Create the dimensions:
Dtime_nc = nc.createDimension('time',ntime)
Dlat_nc = nc.createDimension('lat',ny)
Dlon_nc = nc.createDimension('lon',nx)
Dens_nc = nc.createDimension('member',nens)
Dens_5memb_nc = nc.createDimension('memberGEFS',5)

# Create the variables:
latitude_nc = nc.createVariable('lat','f4',('lat',))
latitude_nc.long_name = 'latitude'
latitude_nc.units = 'degrees_north'

longitude_nc = nc.createVariable('lon','f4',('lon',))
longitude_nc.long_name = 'longitude'
longitude_nc.units = "degrees_east"    

time_valid_nc = nc.createVariable('time','f4',('time',))
time_valid_nc.long_name = 'Instantaneous valid time (in UTC) of the forecast'
time_valid_nc.units = 'hours since 1900-01-01 00:00:00.0 (in UTC)'

yyyymmddhh_valid_nc = nc.createVariable('yyyymmddhh_valid','i4',('time',))
yyyymmddhh_valid_nc.long_name = 'Instantaneous valid date/time (yyyymmddhh format, in UTC) of the forecast'

yyyymmddhh_init_nc = nc.createVariable('yyyymmddhh_init','i4',('time',))
yyyymmddhh_init_nc.long_name = 'Forecast initialization date/time (yyyymmddhh format, in UTC)'

ERA5_nc = nc.createVariable('obs_ERA5','f4',('time','lat','lon',), zlib=True, fill_value=-999)
ERA5_nc.units = "m/s"
ERA5_nc.long_name = 'ERA5 reanalysis of wind speed'
ERA5_nc.valid_range = [0,350]
ERA5_nc.missing_value = -999

rawGEFS_nc = nc.createVariable('fcst_GEFS','f4',('time','lat','lon','memberGEFS',), zlib=True, fill_value=-999)
rawGEFS_nc.units = "m/s"
rawGEFS_nc.long_name = 'GEFS raw forecast of wind speed'
rawGEFS_nc.valid_range = [0,350]
rawGEFS_nc.missing_value = -999

fcstQRF_5memb_nc = nc.createVariable('fcst_QRF_5memb','f4',('time','lat','lon','memberGEFS',), zlib=True, fill_value=-999)
fcstQRF_5memb_nc.units = "m/s"
fcstQRF_5memb_nc.long_name = 'QRF forecast of wind speed (5 members)'
fcstQRF_5memb_nc.valid_range = [0,350]
fcstQRF_5memb_nc.missing_value = -999

fcstQRF_nc = nc.createVariable('fcst_QRF','f4',('time','lat','lon','member',), zlib=True, fill_value=-999)
fcstQRF_nc.units = "m/s"
fcstQRF_nc.long_name = 'QRF forecast of wind speed ('+str(nens)+' members)'
fcstQRF_nc.valid_range = [0,350]
fcstQRF_nc.missing_value = -999

climato_nc = nc.createVariable('fcst_clim','f4',('time','lat','lon','member',), zlib=True, fill_value=-999)
climato_nc.units = "m/s"
climato_nc.long_name = 'climatological forecast of wind speed ('+str(nens)+' members)'
climato_nc.valid_range = [0,350]
climato_nc.missing_value = -999

## Writing data:
latitude_nc[:] = lats_gefs
longitude_nc[:] = lons_gefs
yyyymmddhh_valid_nc[:] = yyyymmddhh_valid.reshape(ntime)
yyyymmddhh_init_nc[:] = yyyymmddhh_init.reshape(ntime)
time_valid_nc[:] = yyyymmddhh_to_time(yyyymmddhh_valid.reshape(ntime))
ERA5_nc[:,:,:] = obsERA5.reshape(ntime,ny,nx)
rawGEFS_nc[:,:,:,:] = rawGEFS.reshape(ntime,ny,nx,5)
fcstQRF_5memb_nc[:,:,:,:] = fcstQRF_5memb.reshape(ntime,ny,nx,5)
fcstQRF_nc[:,:,:,:] = fcstQRF.reshape(ntime,ny,nx,nens)
climato_nc[:,:,:,:] = climato.reshape(ntime,ny,nx,nens)

## Attributes of the NetCDF:
nc.stream = "s4" # ????
nc.title = netcdf_long_name
nc.Conventions = "CF-1.0"  # ????
nc.history = "Created ~Jul 2021 by  Joseph Bellier" 
nc.institution = "NOAA PSL"
nc.platform = "Model" 
nc.references = "None" 

nc.close()


## Save the statistics of the QRF fits into a .npz file:
#-------------------------------------------
outfilename = output_path_QRF_fcsts+'test_QRF_predictorImportance_windSpeed_month_'+str(month)+'_lead_'+str(lead)+'h.npz'

pred_importance = ma.filled(pred_importance, fill_value=-999)
np.savez_compressed(outfilename, pred_importance=pred_importance)

## To load them back:
##   npzfile = np.load(outfilename)
##   pred_importance = ma.array(npzfile['pred_importance'], mask=npzfile['pred_importance'] == -999)
