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
outpath            = '/Volumes/Drobo2_RochelleW/Data/FireWxData/GEFSv12/025deg/perturbed/surface/3hrint_start0hr/'  #change to where you want to store the data
var_download_name  = ['dswrf_sfc','ugrd_hgt','vgrd_hgt',                                      # direct predictors
                      'hgt_hybr','pres_hybr','rh_hybr','tmp_hybr','ugrd_hybr','vgrd_hybr',    # hybrid levels predictors
                      'pres_msl','ugrd_hgt','vgrd_hgt','hgt_ceiling','pwat_eatm','tcdc_eatm'] # indirect predictors
optional_levels    = [None,100,100,
                      [1,2,3,4],[1,2,3,4],[1,2,3,4],[1,2,3,4],[1,2,3,4],[1,2,3,4],
                      None,10,10,None,None,None]
var_directory_name = ['dswrf','u100m','v100m',
                      'hgt_hybr','pres_hybr','rh_hybr','tmp_hybr','ugrd_hybr','vgrd_hybr',
                      'pres_msl','u10m','v10m','hgt_ceiling','pwat','tcc']


#Change index for the variable you want to download. ivar = 3 will download 'apcp' data
ivar = 4


#---The rest of the code shouldn't need to be changed---


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

for idate in range(len(list_of_dates)):
    date_from = list_of_dates[idate]
    date_to   = date_from

    try:
        Download_GEFSv12_function(outpath,var_directory_name[ivar],var_download_name[ivar],optional_levels[ivar],date_from,date_to)
    except FileNotFoundError as err:
        print("FileNotFoundError: {0}".format(err))
        continue  #try next date in the list

#When you're done downloading, you should have 7305 files for Days1-10 (and 7305 files for Days10-16 if you requested those) for each variable!
