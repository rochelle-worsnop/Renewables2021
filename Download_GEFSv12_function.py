#Created by: Joseph Bellier 
#Modified by Rochelle Worsnop, 09/11/2020
#Modified by Joseph Bellier, 6/3/2021, to read data at pressure/hybrid levels
#Download GEFSv12 reforecasts from AWS
import pygrib
from netCDF4 import Dataset
import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy import ma
import os, sys


## User inputs:
#-----------------------------------------------------------
def Download_GEFSv12_function(outpath,vardir,name_var,levels,resol,date_st,date_en,bounds):
    
    ## Where the output NetCDFs will be placed:
    path_output_NetCDFs = outpath + vardir + '/ncfiles/test/'  
    
    ## Where the grib files will be downloaded, and erased right after the desired area has been selected:
    path_bin_gribfiles = outpath + vardir + '/grb2files/test/' 
        
    ## Initialization dates: Define date range or make date_from & date_to the same date to download only one date. 
    date_from = date_st  
    date_to   = date_en 
    
    ## Members wanted
    membs = [0,1,2,3,4]
    
    ## By default, all lead times are downloaded. If only some lead times are desired the code must be modified.
    #-----------------------------------------------------------
    
    ## In the grib2 files the longitudes go from 0 to 360, and not from -180 to +180. If the latter convention has been used by the user, we need to change the bounds:
    bounds = [bounds[b] if b<2 or bounds[b]>=0 else bounds[b]+360 for b in range(4)]
    
    
    ## Construct the arrays of initialization dates:
    #-----------------------------------------------------------
    def date_to_yyyymmddhh(dt_time):
        return np.array(1000000*dt_time.year + 10000*dt_time.month + 100*dt_time.day + dt_time.hour, dtype=np.int64)
	
    ## Initialization dates, as integers (yyyymmddhh format):
    dates_init_pd = pd.DatetimeIndex(pd.date_range(date_from, date_to, freq='1D').tolist())
    yyyymmddhh_init = date_to_yyyymmddhh(dates_init_pd)
    ndates = yyyymmddhh_init.size
    
    ## Initialization dates, as time (hours since '1900-01-01 00:00:00'):
    date_origin = pd.to_datetime('1900-01-01 00:00:00')
    time_init = np.array((dates_init_pd - date_origin)/np.timedelta64(1,'h'), dtype=np.int64)
    
    
    nmembs = len(membs)
    
    nlevels = 1 if (np.isscalar(levels) or levels is None) else len(levels)

    if resol == 'Days:1-10':
        leads = range(3, 240+1, 3)
    elif resol == 'Days:10-16':
        leads = range(246, 384+1, 6)
    nleads = len(leads)


    ## Valid dates, as integers (yyyymmddhh format):
    yyyymmddhh_valid = np.zeros((ndates,nleads), dtype=np.int64)
    for l in range(nleads):
        lead = leads[l]
        yyyymmddhh_valid[:,l] = date_to_yyyymmddhh(dates_init_pd + np.timedelta64(lead,'h'))


    for n in range(ndates):
        cyear = str(yyyymmddhh_init[n]//1000000)
        cdate = str(yyyymmddhh_init[n])

        print(name_var, resol, cdate)

        for m in range(nmembs):
            memb = membs[m]
            cmemb = 'c'+"{:0>2d}".format(memb) if m==0 else 'p'+"{:0>2d}".format(memb)
            name_file = name_var+'_'+str(yyyymmddhh_init[n])+'_'+cmemb+'.grib2'

            ## We download the file from AWS:
            ##   (if the grib files have already been downloaded we just need to adapt that part of the code)
            #-------------------------------
            url = 'https://noaa-gefs-retrospective.s3.amazonaws.com/GEFSv12/reforecast/'+cyear+'/'+cdate+'/'+cmemb+'/'+resol+'/'+name_file
            cmd = 'wget -P '+path_bin_gribfiles+' '+url
            istat = os.system(cmd)
            ## Once a week the reforecasts go to 35 days, so the resol 'Days:10-16' is replaced by 'Days:10-35'
            if istat != 0 and resol == 'Days:10-16':
                url = 'https://noaa-gefs-retrospective.s3.amazonaws.com/GEFSv12/reforecast/'+cyear+'/'+cdate+'/'+cmemb+'/Days:10-35/'+name_file
                cmd = 'wget -P '+path_bin_gribfiles+' '+url
                istat = os.system(cmd)
            if istat != 0:
                raise FileNotFoundError('Trying to download '+cdate+'/'+cmemb+'/'+resol+'/'+name_file+' but the file is not found on AWS')
            #-------------------------------



            ## If this is the first grib file downloaded for that variable/resol, we read from only 1 grib message the information such as the grid, the variable's long name, the unit...
            if n == 0 and m == 0:
                grbs = pygrib.open(path_bin_gribfiles + name_file)

                ## To make sure we read the GRIB messages corresponding to the correct level(s):
                grb_all = grbs.read()
                levels_in_grib = list(set([grb.level for grb in grb_all]))		

                if nlevels == 1:   # There is only one level specified by the user (can be None)
                    if len(levels_in_grib) != 1:     # There is more than one level stored in this grib file
                        if levels is None or levels not in levels_in_grib:
                            raise ValueError('There is more than one level stored in the GRIB files for the variable '+name_var+', but no level is provided, or the single level provided does not match')
                        else:
                            level = levels
                    else:
                        level = levels_in_grib[0]
                    grb = grbs.select(level=level)[0]    

                else:             # There is more than one level specified in 'levels'
                    ## Check that all desired levels are present in the GRIB file:
                    for k in range(nlevels):
                        if levels[k] not in levels_in_grib:
                            raise ValueError('For the variable '+name_var+', level '+level[k]+' is provided, but it is not found in the GRIB file')
                    grb = grbs.select(level=levels[0])[0]   # We select the first level to read the GRIB information from


                lats_mat_grib, lons_mat_grib = grb.latlons()
                lats_grib, lons_grib = lats_mat_grib[:,0], lons_mat_grib[0,:]

                ## We derive the indices (in lat and lon) of the sub-area we want:
                id_lats = np.logical_and(lats_grib >= bounds[0], lats_grib <=bounds[1])
                id_lons = np.logical_and(lons_grib >= bounds[2], lons_grib <=bounds[3])

                ## Dimension of the sub-area:
                lats, lons = lats_grib[id_lats], lons_grib[id_lons]
                ny, nx = lats.size, lons.size

                ## Variable long name and units:
                longName_var  = grb.name
                shortName_var = grb.shortName
                units_var     = grb.parameterUnits
                if nlevels > 1:
                    typeOfLevel_var = grb.typeOfLevel
                    if typeOfLevel_var == 'isobaricInhPa':
                        levelUnits_var = 'hPa'
                        levelName_var = 'Isobaric surface'                        
                        levelShortName_var = 'pressure'                        
                    elif typeOfLevel_var == 'hybrid':
                        levelUnits_var = 'sigma'
                        levelName_var = 'Hybrid level'
                        levelShortName_var = 'hybrid'
                    else:
                        raise ValueError("typeOfLevel in GRIB file is '"+typeOfLevel_var+"', but the current code doesn't read it")

                ## Type of the variable: accumulated, instantaneous or averaged
                type_var = grb.stepTypeInternal

                ## Array where will be store the data:
                if nlevels == 1: 
                    array_var = ma.array(np.zeros((ndates,nleads,nmembs,ny,nx)), dtype=np.float64, mask=True)
                else:
                    array_var = ma.array(np.zeros((ndates,nleads,nmembs,nlevels,ny,nx)), dtype=np.float64, mask=True)


            ## Read the data:
            grbs = pygrib.open(path_bin_gribfiles + name_file) 
            for l in range(nleads):   # We read the messages that corresponds to each lead time 

                for k in range(nlevels):    # And to each level
                    if nlevels == 1:
                        pass   # ('level' should have already been defined earlier)
                    else:
                        level = levels[k]


                    ## Read in data from each lead time. This does not deaccumulate. 
                    if (type_var == 'accum') or (type_var == 'avg'):
                        lead_beg = leads[l] - 6 if (leads[l] % 6) == 0 else leads[l] - 3
                        lead_end = leads[l]
                        grb = grbs.select(level=level, startStep=lead_beg, endStep=lead_end)[0]
                    elif type_var == 'instant':
                        grb = grbs.select(level=level, forecastTime=leads[l])[0]
                    else:
                        raise ValueError("type_var in GRIB file is '"+type_var+"', but the current code doesn't read it")

                    if nlevels == 1:
                        array_var[n,l,m,:,:] = grb.values[id_lats,:][:,id_lons]
                    else:
                        array_var[n,l,m,k,:,:] = grb.values[id_lats,:][:,id_lons]


            ## We finally delete the grib file since we don't need it anymore:
            #cmd = 'mv '+path_bin_gribfiles + name_file + '/Volumes/Drobo2_RochelleW/Data/FireWxData/GEFSv12/025deg/perturbed/surface/3hrint_start0hr/tcc/grb2files/test/'
            cmd = 'rm '+path_bin_gribfiles + name_file
            os.system(cmd)

    ## Save the variable into a netCDF:
    #-----------------------------------------------------------
    try:
        os.makedirs(path_output_NetCDFs)
    except FileExistsError:
        pass

    if ndates > 1 :   #Include date range in file name
        ncfile = path_output_NetCDFs +name_var+'_'+str(yyyymmddhh_init[0])+'_'+str(yyyymmddhh_init[-1])+'_'+resol.replace(':','')+'.nc'
    else:
        ncfile = path_output_NetCDFs +name_var+'_'+str(yyyymmddhh_init[0])+'_'+resol.replace(':','')+'.nc'   #Just one date

    nc = Dataset(ncfile,'w',format='NETCDF4_CLASSIC')

    ## Create the dimensions:
    Dtime_nc = nc.createDimension('time',ndates)
    Dlat_nc = nc.createDimension('lat',ny)
    Dlon_nc = nc.createDimension('lon',nx)
    Dens_nc = nc.createDimension('ens',nmembs)
    Dfhour_nc = nc.createDimension('fhour',nleads)
    if nlevels > 1:
        Dlevel_nc = nc.createDimension(levelShortName_var, nlevels)

    ## Create the variables
    latitude_nc = nc.createVariable('lat','f4',('lat',))
    latitude_nc.long_name = 'Latitude'
    latitude_nc.units = 'degrees_north'

    longitude_nc = nc.createVariable('lon','f4',('lon',))
    longitude_nc.long_name = 'Longitude'
    longitude_nc.units = "degrees_east"    

    yyyymmddhh_init_nc = nc.createVariable('intTime','i4',('time',))
    yyyymmddhh_init_nc.long_name = 'Forecast initialization date/time (yyyymmddhh format, in UTC)'

    yyyymmddhh_valid_nc = nc.createVariable('intValidTime','i4',('time','fhour'))
    yyyymmddhh_valid_nc.long_name = 'Instantaneous valid date/time (yyyymmddhh format, in UTC)'+(' of the end of the accumulation period (3h or 6h)' if type_var=='accum' else '')

    time_init_nc = nc.createVariable('time','i4',('time',))
    time_init_nc.long_name = 'Forecast initialization date/time'
    time_init_nc.units = "hours since 1900-01-01 00:00:00"

    ens_nc = nc.createVariable('ens','i4',('ens',))
    ens_nc.long_name = 'ensemble'
    ens_nc.description = '0 is control, other values are perturbation numbers'

    fhour_nc = nc.createVariable('fhour','i4',('fhour',))
    fhour_nc.long_name = 'forecast hour'
    fhour_nc.units = 'hours'

    if nlevels == 1:
        variable_nc = nc.createVariable(shortName_var,'f8',('time','fhour','ens','lat','lon',), zlib=True)
        variable_nc.long_name = longName_var
        variable_nc.units = units_var

    else:
        level_nc = nc.createVariable(levelShortName_var,'i4',(levelShortName_var,))
        level_nc.long_name = levelName_var
        level_nc.units = levelUnits_var

        variable_nc = nc.createVariable(shortName_var,'f8',('time','fhour','ens',levelShortName_var,'lat','lon',), zlib=True)
        variable_nc.long_name = longName_var +' @ '+levelShortName_var
        variable_nc.units = units_var

    ## Writing data:
    latitude_nc[:] = lats
    longitude_nc[:] = lons - 360.0      #Want -180 to 180 deg
    yyyymmddhh_init_nc[:] = yyyymmddhh_init
    yyyymmddhh_valid_nc[:,:] = yyyymmddhh_valid
    time_init_nc[:] = time_init
    ens_nc[:] = membs
    fhour_nc[:] = leads
    if nlevels == 1:
        variable_nc[:,:,:,:,:] = array_var
    else:
        level_nc[:] = levels
        variable_nc[:,:,:,:,:,:] = array_var

    ## Attributes of the NetCDF:
    nc.title = 'Subset of GEFSv12 reforecasts for the variable '+longName_var+' over the area delineated by '+str(bounds)
    nc.Conventions = "CF-1.6"
    nc.history = "Created ~Sep 2020 by Joseph Bellier, some modifications by Rochelle Worsnop Sep2020" 
    nc.institution = "NOAA/ESRL Physical Sciences Laboratory"

    nc.close()
        
