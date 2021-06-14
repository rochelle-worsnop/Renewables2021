#Rochelle Worsnop
#Call script to download GEFSv12 reforecast data from AWS server

import pygrib
from netCDF4 import Dataset
import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy import ma
import os, sys, glob
from datetime import timedelta

#---import GEFSv12 download function created by Joseph Bellier---
from Download_GEFSv12_function import Download_GEFSv12_function


#===================================================================
## User inputs:
#-----------------------------------------------------------
#===================================================================

## Change to where you want to store the data
outpath            = '/Volumes/Drobo2_RochelleW/Data/FireWxData/GEFSv12/025deg/perturbed/surface/3hrint_start0hr/'

## Name of the variables to be downloaded (must be the exact names used in AWS):
var_download_name  = ['dswrf_sfc','ugrd_hgt','vgrd_hgt',                                      # direct predictors
                      'hgt_hybr','pres_hybr','rh_hybr','tmp_hybr','ugrd_hybr','vgrd_hybr',    # hybrid levels predictors
                      'pres_msl','ugrd_hgt','vgrd_hgt','hgt_ceiling','pwat_eatm','tcdc_eatm'] # indirect predictors

## Should be None is the variable has only one level stored in the GRIB file (e.g., all the surface variables)
##  Otherwise, you should provide the level at which the variable is desired (e.g., 100 for the 100m wind, 850 for data at 850hPa, 1 for the 1-sigma data at hybrid levels, etc.). 
##  If a scalar is provided, the output netCDF will contain only that level. 
##  If a list of levels is provided, the output netCDF will contain all the levels.
optional_levels    = [None,100,100,
                      [1,2,3,4],[1,2,3,4],[1,2,3,4],[1,2,3,4],[1,2,3,4],[1,2,3,4],
                      None,10,10,None,None,None]

## Name of the directories under which the variables will be stored (can be different from var_download_name):
var_directory_name = ['dswrf','u100m','v100m',
                      'hgt_hybr','pres_hybr','rh_hybr','tmp_hybr','ugrd_hybr','vgrd_hybr',
                      'pres_msl','u10m','v10m','hgt_ceiling','pwat','tcc']

## Reslution of the forecasts: (choose between 'Days:1-10' and 'Days:10-16')
resol = 'Days:1-10'
#===================================================================



## Bounds of the sub-area to keep the data over:
bounds = [23.0, 51.0, -131, -63.0] # CONUS DOMAIN for the renewables project

#reforecasts GEFSv12: 01/01/2000 to 12/31/2019
dateStart     = '2000-01-01 00:00:00'
dateEnd       = '2019-12-31 00:00:00'
start_date    = datetime.datetime.strptime(dateStart,'%Y-%m-%d 00:00:00').date()
end_date      = datetime.datetime.strptime(dateEnd,'%Y-%m-%d 00:00:00').date()
delta_one_day = timedelta(days=1)

date          = start_date
list_of_dates = []
while date <= end_date:
	subfolder_path = date.strftime('%Y-%m-%d 00:00:00')
	list_of_dates.append((date.strftime('%Y%m%d 00:00:00')))
	date += delta_one_day 

#------------------------------------------------------------------------------------------------------------------------ 
#Download data from AWS server for every date in 'list_of_dates' by calling download function.

for ivar in range(len(var_download_name)):
    for idate in range(len(list_of_dates)):
        date_from = list_of_dates[idate]
        date_to   = date_from

        try:
            Download_GEFSv12_function(outpath,var_directory_name[ivar],var_download_name[ivar],optional_levels[ivar],resol,date_from,date_to,bounds)
        except FileNotFoundError as err:
            print("FileNotFoundError: {0}".format(err))
            continue  #try next date in the list, but doesn't stop the script

# When you're done downloading, you should have 7305 files for Days1-10 (and 7305 files for Days10-16 if you requested those) for each variable!
