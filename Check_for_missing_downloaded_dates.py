#Rochelle Worsnop
#November 17, 2020 
#For GEFSv12 reforecasts on Amazon web services
#A quick script to check if you successfully downloaded all the files you meant to. 
#Will test for each variable or you can select an index associated with a particular variable
#Run: python -i Check_for_missing_downloaded_dates.py

import datetime
import numpy as np
import pandas as pd
from numpy import ma
import os, sys, glob
from datetime import timedelta

#Define path and variable names of the data you downloaded
data_path          = '/Volumes/Drobo2_RochelleW/Data/FireWxData/GEFSv12/025deg/perturbed/surface/3hrint_start0hr/'
outpath            = data_path
var_directory_name = ['u10m','v10m','t2m','apcp','tcc','spfh','pres_sfc','weasd_sfc']

#Loop through or isolate one particular variable to check
for ivar in [4]: #[0,1,2,3,4,5]:

    #Create list of dates that you want data for.
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
        list_of_dates.append((date.strftime('%Y%m%d')))
        date += delta_one_day
    list_of_dates = np.asarray(list_of_dates)
    n_allfiles = len(list_of_dates)
    
    
    #Combine and sort all files for Days1-10 and then again for Days10-16 
    downloaded_files         = outpath + var_directory_name[ivar] + '/ncfiles/'
    infiles_D1_10            = sorted(glob.iglob(downloaded_files + '*Days1-10.nc'))
    infiles_D10_16           = sorted(glob.iglob(downloaded_files + '*Days10-16.nc'))
    downloaded_dates_D1_10   = np.asarray([os.path.basename(i)[-22:-14] for i in infiles_D1_10])  #[7:15]
    n_downloadedfiles_D1_10  = len(downloaded_dates_D1_10)
    downloaded_dates_D10_16  = np.asarray([os.path.basename(i)[-23:-15] for i in infiles_D10_16])
    n_downloadedfiles_D10_16 = len(downloaded_dates_D10_16)
        
    
    #Find the dates that were not downloaded (i.e., missing)
    boolean_mask_D1_10       = np.isin(list_of_dates,downloaded_dates_D1_10,invert=True)
    dates_missing_D1_10      = np.array([int(i) for i in list_of_dates[boolean_mask_D1_10]])
    n_missing_D1_10          = len(dates_missing_D1_10)
    boolean_mask_D10_16      = np.isin(list_of_dates,downloaded_dates_D10_16,invert=True)
    dates_missing_D10_16     = np.array([int(i) for i in list_of_dates[boolean_mask_D10_16]])
    n_missing_D10_16         = len(dates_missing_D10_16)
    
    #Index of the missing dates from the 'list of dates'
    #---Copy and paste these indices to run in your download script again---
    idx_of_missing_dates_D1_10  = np.array(np.where(boolean_mask_D1_10==1)[0])
    idx_of_missing_dates_D10_16 = np.array(np.where(boolean_mask_D10_16==1)[0])

    print(idx_of_missing_dates_D1_10)
    print(idx_of_missing_dates_D10_16)
    print(' ')

    #Print missing dates for the variable and D1-10 and D10-16
    #---These are the missing dates from your dataset---
    print(var_directory_name[ivar],'Days1-10',n_missing_D1_10)
    print(dates_missing_D1_10)
    print(' ')
    print(var_directory_name[ivar],'Days10-16',n_missing_D10_16)
    print(dates_missing_D10_16)
    print(' ')    
    
    
    
