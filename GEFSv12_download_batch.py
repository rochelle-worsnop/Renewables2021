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
var_directory_name = ['u10m','v10m','t2m','apcp','tcc','spfh','pres_sfc','weasd_sfc']                               #can keep the same or change to your own convention
var_download_name  = ['ugrd_hgt','vgrd_hgt','tmp_2m','apcp_sfc','tcdc_eatm','spfh_2m','pres_sfc','weasd_sfc']       #update with variables you want to download
optional_level     = ['10','10','None','None','None','None','None','None']                                          #update to correspond to levels of the variables in previous line

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
        Download_GEFSv12_function(outpath,var_directory_name[ivar],var_download_name[ivar],optional_level[ivar],date_from,date_to)
    except:
        print(' ')
        print('ERROR ' +'**** ' + date_to + ' ****' )
        print(' ')
        continue  #try next date in the list

#When you're done downloading, you should have 7305 files for Days1-10 (and 7305 files for Days10-16 if you requested those) for each variable!
