#Purpose: univariate and multivariate post-processing of local noon wind speed 
#Univariate method:   EMOS regression...minimum CRPS to determine regression coefficients. Ensemble mean and variance as predictors
#Multivariate method: ECC-Q standard...rank of raw ensemble is used to dictate rank of quantiles of the predictive Gaussian distribution
#Rochelle Worsnop, rochelle.worsnop@noaa.gov, July 2021
#---NOTES---#
#---------------------------------------------------------------------------------------------------------------------------------
#Training period     = Add dates for a given month & lead time. All years except the verification year (leave one year out cross-validation) 
#Verification period = Add dates for a given month & lead time. Only the verification year.  
#Don't have to calculate climatology of wind speed, because our training period is just 1 month instead of many months
#(e.g. like required for precip accumulation),where climatology could vary.
#We are esentially assuming here that the climatology is stagnant for the verification month, but changes month-by-month.
#!!!We don't remove climatology, but it's accounted for by our short training window. We can have a shorter training window for
#wind speed as opposed to precip, because even zero values are meaningful for the regression, since there is wind every day unlike precip.
#---------------------------------------------------------------------------------------------------------------------------------

import numpy as np
import scipy as sp
import math
import os, sys, glob

from netCDF4 import Dataset
from numpy import ma
from numpy.random import random_sample
from numpy.linalg import solve
from scipy.interpolate import Rbf
from scipy.ndimage.filters import gaussian_filter
from scipy import stats
from scipy.stats import gamma
from scipy.stats import rankdata
from scipy.special import beta
from scipy.optimize import minimize_scalar
from scipy.optimize import minimize
import datetime
from datetime import timedelta, date
from dateutil.relativedelta import relativedelta
from matplotlib import pyplot
from calendar import monthrange
from scipy.stats import linregress
from scipy.stats.mstats import mquantiles


#---PASS INPUTS TO PYTHON SCRIPT FOR EACH MONTH, YEAR, AND LEAD TIME---
cyear             = sys.argv[1] 
cmonth            = sys.argv[2] 
cleade            = sys.argv[3] 
print(cmonth, cyear, cleade)

#---Uncomment to give inputs for testing rather than feeding from a batch script---
#cyear  = '00' #'17' 
#cmonth = '02' 
#cleade = '01' 

##################################################################################################################################
imonth                   = int(cmonth)
iyear                    = int(cyear)
ileade                   = int(cleade)
nmem_fcst                = 5   #GEFSv12 reforecasts have 5 members
nhindcast_yrs            = 20  #GEFSv12 reforecasts has 20 years of hindcasts 

output_path              = '/home/rworsnop/Phase2NFDRS_GEFF/Data/FireWxData/GEFSv12/HRES_05deg/surface/GEFF_inputs/postprocessed_daily_fcst/'
data_path_fcst           = '/home/rworsnop/Phase2NFDRS_GEFF/Data/FireWxData/GEFSv12/HRES_05deg/surface/GEFF_inputs/raw_daily_fcst/allmembers/' + 'wspd/'
data_path_anl            = '/home/rworsnop/Phase2NFDRS_GEFF/Data/FireWxData/ERA5/HRES_025deg/surface/GEFF_inputs/raw_daily_fcst/upscaled_0pt5deg/' + 'wsnoon/'

#---DATA MASK---
maskpath                 = '/home/rworsnop/Phase2NFDRS_GEFF/GEFF/' + 'CONUS_MASK_ERA5anl_025deg.nc'
masknc                   = Dataset(maskpath)
DST_CONUS_MASK           = masknc.variables["CONUS_MASK"][::2,::2]   #DESTINATION ANALYSIS GRID
masknc.close()

#----LOAD REFORECAST & REANALYSES FOR A PARTICULAR LEAD TIME---
##################################################################################################################################
infile_fcst              = data_path_fcst + format(ileade,'02') + '_wspdnoon_alldates.nc'
infile_anl               = data_path_anl  + 'localwspeed_alldates_ERA5upscaled05deg.nc'
nc_fcst                  = Dataset(infile_fcst)
nc_anl                   = Dataset(infile_anl)

yyyymmdd_init_anl        = np.array([int(x) for x in nc_anl.variables['yyyymmdd_init'][:]])
yyyymmdd_init_fcst	 = nc_fcst.variables['yyyymmdd_init_fcst'][:]
time_valid_fcst          = nc_fcst.variables['yyyymmddhh_valid_fcst'][:]
yyyymmdd_valid_fcst	 = np.asarray([int((datetime.datetime(1900,1,1) + relativedelta(hours = int(x))).strftime("%Y%m%d%H")) for x in time_valid_fcst]) // 100
lons_fcst                = nc_fcst.variables['lon'][:]
lats_fcst                = nc_fcst.variables['lat'][:]
lons_anl                 = lons_fcst
lats_anl                 = lats_fcst
wspd_fcst_ens            = nc_fcst.variables['wspdnoon_all'][:]
wspd_anal                = np.squeeze(nc_anl.variables['wspdnoon_all'][:]) 

ntimes_anl,nlat_anl,nlon_anl                   = wspd_anal.shape
ntimes_fcst,nmem_fcst,nlat_fcst,nlon_fcst      = wspd_fcst_ens.shape
nc_fcst.close()
nc_anl.close()

#---Make sure you're using the same dates for the analyses and forecasts---
goodidx_anl        = np.intersect1d(yyyymmdd_init_anl,yyyymmdd_valid_fcst ,return_indices=True)[1] #indicies of yyyymmdd_init_anl that have values in common with yyyymmdd_init_fcst
goodidx_fcst	   = np.intersect1d(yyyymmdd_init_anl,yyyymmdd_valid_fcst ,return_indices=True)[2] #indices of yyyymmdd_init_fcst that have values in common with yyyymmdd_init_anl
yyyymmdd_init_anl  = yyyymmdd_init_anl[goodidx_anl]
yyyymmdd_init_fcst = yyyymmdd_init_fcst[goodidx_fcst]
time_valid_fcst    = time_valid_fcst[goodidx_fcst]
wspd_anal          = wspd_anal[goodidx_anl,:,:]
wspd_fcst_ens	   = wspd_fcst_ens[goodidx_fcst,:,:,:]
ntimes_fcst, nmem_fcst, nlat_fcst, nlon_fcst = wspd_fcst_ens.shape
ntimes_anl,nlat_anl,nlon_anl                 = wspd_anal.shape
nyrs               = nhindcast_yrs


#----CREATE WINDOW OF DATES THAT ARE IN VERIFICATION MONTH----
##################################################################################################################################
#Training period:     all dates in the defined month, all years except verif year
#Verification period: all dates in verif month, verif year only
ndays_in_month              = monthrange(2016,imonth)[1]   #You can use any year that is a leap-year. 2016 was a leap year. 
date_window                 = np.array([datetime.date(2018-(19-iyear),imonth,day) for day in range(1,ndays_in_month+1)])
mmdd_date_window            = np.array([int(x.strftime("%Y%m%d")) - 10000*(int(x.strftime("%Y"))) for x in date_window])   #mmdd of each date in the window
len_window                  = len(mmdd_date_window)
nverif                      = len_window

#----CREATE WINDOW OF DATES THAT ARE  +/- 5 DAYS AROUND VERIFICATION MONTH----
##################################################################################################################################
#This is used to calculate climatology for each DOY in the verification month...used as benchmark
date_window_climbuffer = []
for iday in range(-5,len(date_window)+5+1):
        date_window_climbuffer.append(date_window[0] + timedelta(days=iday))
mmdd_date_window_climbuffer = np.array([int(x.strftime("%Y%m%d")) - 10000*(int(x.strftime("%Y"))) for x in date_window_climbuffer])
len_climbuffer              = len(mmdd_date_window_climbuffer)



#---FIND TRAINING & VERIFICATION INDICIES OF ANALYSES THAT FALL WITHIN THE DATE WINDOW---
##################################################################################################################################
ind_anl_climbuff = ma.array(np.zeros((nhindcast_yrs,len_climbuffer),dtype=np.int32),mask=True)
leapyr_idx	 = []
for iyr in range(nhindcast_yrs):
	for iwd in range(len_climbuffer):
		mmdd     = mmdd_date_window_climbuffer[iwd]
		yyyymmdd = ((rfcst_startyr-nyrs+1)+iyr)*10000+mmdd		
		if ((yyyymmdd // 10000) % 4 !=0) and (mmdd == 229):  #exception for feb. in a leap year. 
			continue  #don't search for 2/29 if it's not a leap year
		if yyyymmdd in yyyymmdd_init_anl:
			ind_anl_climbuff[iyr,iwd] = np.where(yyyymmdd_init_anl==yyyymmdd)[0][0]
	if (yyyymmdd // 10000) % 4 == 0:
		leapyr_idx.append(iyr)

#Subset climbuffer window to just include verification dates (i.e., dates in verification month) & verif year
verif_ind_anl	         = ma.copy(ind_anl_climbuff[iyear,5:(ndays_in_month+5)])    

maxdays_in_mnth          = monthrange(2016,imonth)[1] #pick leapyear to return maximum number of days
train_ind_anl            = ma.array(np.zeros((nyrs,maxdays_in_mnth),dtype=np.int32),mask=True)
verif_ind_anl_5daybuffer = ma.array(np.zeros((nyrs,ind_anl_climbuff.shape[1]),dtype=np.int32),mask=True)
if imonth == 2:  #Need to make adjustments for leapyears
        for iyr in range(nyrs):
                if iyr == iyear:  #Don't collect verification year
                        continue
                if iyr in leapyr_idx:
                        train_ind_anl[iyr,:] = ind_anl_climbuff[iyr,5:(maxdays_in_mnth+5)]  #There are 29 days (i.e., max ndays in Feb.)
                else:
                     	train_ind_anl[iyr,:-1] = ind_anl_climbuff[iyr,5:(maxdays_in_mnth-1+5)]
                verif_ind_anl_5daybuffer[iyr,:] = ind_anl_climbuff[iyr,:]   #verif year is skipped in the loop so it's already masked out.
else: #This should work for any other month
        ind_anl_climbuff[iyear,:] = ma.masked
        train_ind_anl             = ind_anl_climbuff[:,5:(ndays_in_month+5)]  #Training set includes all years except for verification year

        ###### 5 DAY BUFFER BELOW IS USED TO CALCULATE ANLAYSIS CLIMATOLOGY TO COMPARE FORECASTS TO, SUCH AS FOR CALCULATION OF CRPSS#####
         #Use this +/- 5 day window around each verif DOY for all years to calculate analysis climatology for sample CRPS
        verif_ind_anl_5daybuffer = ind_anl_climbuff[:,:]  #already masked out verification year above.


#---FIND TRAINING & VERIFICATION INDICIES OF REFORECASTS THAT FALL WITHIN THE DATE WINDOW---
##################################################################################################################################
yyyy_valid_fcst       = yyyymmdd_valid_fcst // 10000
mmdd_valid_fcst       = yyyymmdd_valid_fcst - 10000*yyyy_valid_fcst

ind_fcst_climbuff = ma.array(np.zeros((nyrs,len_climbuffer),dtype=np.int32),mask=True)
leapyr_idx	 = []
for iyr in range(nyrs):
	for iwd in range(len_climbuffer):
		mmdd = mmdd_date_window_climbuffer[iwd]
		yyyymmdd = ((2019-nyrs+1)+iyr)*10000+mmdd              #start year of the 'model year'.
		#if mmdd<607:                                                   #mmdd start of ECMWF reforecast starts #reforecasts $
		#        yyyymmdd = ((2018-nyrs+1)+iyr)*10000+mmdd              #of the last reforecast start Ending year of the 'mo$
		#else:
		#     	yyyymmdd = ((2017-nyrs+1)+iyr)*10000+mmdd              #start year of the 'model year'. First reforecast st$
		if ((yyyymmdd // 10000) % 4 !=0) and (mmdd == 229):
			continue  #don't search for 2/29 if it's not a leap year
		if yyyymmdd in yyyymmdd_valid_fcst.flatten():
			ind_fcst_climbuff[iyr,iwd] = np.where(yyyymmdd_valid_fcst.flatten()==yyyymmdd)[0][0]  #Indicies of reforecasts within clim buffer window

	if (yyyymmdd // 10000) % 4 == 0:
		leapyr_idx.append(iyr)

verif_ind_fcst = ma.copy(ind_fcst_climbuff[iyear,5:(ndays_in_month+5)])   #Subset climbuffer window to just include verification. This has to be defined separately otherwise it changes when the mask changes below

train_ind_fcst = ma.array(np.zeros((nyrs,ndays_in_month),dtype=np.int32),mask=True)
if imonth == 2:  #Need to make adjustments for leapyears
        for iyr in range(nyrs):
                if iyr == iyear:  #Don't collect verification year
                        continue
                if iyr in leapyr_idx:
                        train_ind_fcst[iyr,:] = ind_fcst_climbuff[iyr,5:(ndays_in_month+5)]  #There are 29 days 
                else:
                     	train_ind_fcst[iyr,:-1] = ind_fcst_climbuff[iyr,5:(ndays_in_month-1+5)]    #verif year skipped so already masked out for month = 2
else: #This should work for any other month
        ind_fcst_climbuff[iyear,:]   = ma.masked                    #mask out verif year...don't include it in training data
        train_ind_fcst = ind_fcst_climbuff[:,5:(ndays_in_month+5)]


#---NOW THAT WE HAVE THE VERIFICATION & TRAINING INDICIES, SUBSET THE TIME AND VARIABLE ARRAYS----
##################################################################################################################################
anl_verif_dates                 = yyyymmdd_init_anl[verif_ind_anl[~verif_ind_anl.mask]] #verification period includes all dates in the month, one year 
anl_train_dates                 = yyyymmdd_init_anl[train_ind_anl[~train_ind_anl.mask]] #training period includes all dates in the month, all yrs except verif year. 
anl_verif_dates_5daybuffer      = yyyymmdd_init_anl[verif_ind_anl_5daybuffer[~verif_ind_anl_5daybuffer.mask]] #+/-5 days around verif dates. verif year excluded.
fcst_verif_dates                = yyyymmdd_valid_fcst.flatten()[verif_ind_fcst[~verif_ind_fcst.mask]] 
fcst_train_dates                = yyyymmdd_valid_fcst.flatten()[train_ind_fcst[~train_ind_fcst.mask]] 

wspd_anl_verif_tmp              = wspd_anal[verif_ind_anl[~verif_ind_anl.mask],:,:]                                             
wspd_anl_train_tmp              = wspd_anal[train_ind_anl[~train_ind_anl.mask],:,:]                       
wspd_anl_verif_5daybuffer_tmp   = wspd_anal[verif_ind_anl_5daybuffer[~verif_ind_anl_5daybuffer.mask],:,:]
wspd_ensfcst_verif_tmp          = wspd_fcst_ens[verif_ind_fcst[~verif_ind_fcst.mask],:,:,:]               
wspd_ensfcst_train_tmp          = wspd_fcst_ens[train_ind_fcst[~train_ind_fcst.mask],:,:,:]               

#---PUT FORECAST ARRAY INTO SAME SIZE AS ANALYSIS ARRAY...FILL MISSING DATES WITH MASK---
##################################################################################################################################
fcst_train_datetimes     = np.array([datetime.datetime.strptime(str(fcst_train_dates[:][x]),'%Y%m%d').date() for x in range(len(fcst_train_dates[:]))])
anl_train_datetimes	 = np.array([datetime.datetime.strptime(str(anl_train_dates[:][x]),'%Y%m%d').date() for x in range(len(anl_train_dates[:]))])

wspd_ensfcst_train_new   = ma.array(np.zeros((len(anl_train_datetimes),nmem_fcst,nlat_anl,nlon_anl)),mask=True)
fcst_train_dates_new     = ma.array(np.zeros((len(anl_train_datetimes)),dtype='int'),mask=True)
for i in range(len(anl_train_datetimes)):
        if anl_train_datetimes[i] in fcst_train_datetimes:
                idx                              = np.where(fcst_train_datetimes == anl_train_datetimes[i])[0][0]
                wspd_ensfcst_train_new[i,:,:,:]  = wspd_ensfcst_train_tmp[idx,:,:,:]
                fcst_train_dates_new[i]          = fcst_train_dates[idx]
wspd_ensfcst_train_tmp = wspd_ensfcst_train_new

wspd_ensfcst_verif_new = ma.array(np.zeros((nverif,nmem_fcst,nlat_anl,nlon_anl)),mask=True)
fcst_verif_dates_full= ma.array(np.zeros((nverif),dtype='int'),mask=True)
for i in range(nverif):
        if anl_verif_dates[i] in fcst_verif_dates:
                idx                            = np.where(fcst_verif_dates == anl_verif_dates[i])
                wspd_ensfcst_verif_new[i,:,:,:]  = wspd_ensfcst_verif_tmp[idx,:,:,:]
                fcst_verif_dates_full[i]       = fcst_verif_dates[idx]
wspd_ensfcst_verif_tmp = wspd_ensfcst_verif_new

ntrain_yrsNdts       = nhindcast_yrs*nverif  #wspd_anl_train_tmp.shape[0]   


#---FIT REGRESSION MODEL BASED ON MINIMIZING THE CRPS & ESTIMATE DIFFERENT CANDIDATE DISTRIBUTION'S FORECAST PARAMETERS---
##################################################################################################################################
print('Fitting regression model')

Pwr = 0.50  #square root power transform
def crpsCondTGaussian(par,obs,ensmean,ensvar):
	#Average CRPS for truncated Gaussian distribution conditional on the ensemble statistics 
	#close-form equation of CRPS for truncated Normal distribution. Scheuerer, M. and Moller, D. 2015
	mu         = par[0] + par[1]*ensmean    
	sigma      = ma.sqrt(par[2] + par[3]*ensvar)  
	mu_div_sig = mu/sigma 
	ystd       = ((obs-mu)/sigma)   #standardized 
	term1      = ( sigma*(stats.norm.cdf(mu_div_sig))**-2 )   
	term2      = ( ystd*stats.norm.cdf(mu_div_sig)*(2.*stats.norm.cdf(ystd) + stats.norm.cdf(mu_div_sig) - 2.)  )
	term3      = ( 2.*stats.norm.pdf(ystd)*stats.norm.cdf(mu_div_sig) )
	term4      = ( (1./np.sqrt(math.pi))*stats.norm.cdf(np.sqrt(2.0)*mu_div_sig)   )
	crps       = term1 * (term2+term3-term4)
	return ma.mean(crps) 

def crpsCondsqrtTGaussian(par,obs,obs_sqrt,ensmean_sqrt,ensvar_sqrt):
	#Average CRPS for square root-transformed truncated normal distribution conditional on the ensemble statistics
	#close-form equation of sqrtTGaussian CRPS that uses untranformed and sqrt_transformed obs, so that it's comparable with TGuassian CRPS score. 
	#CRPS is calculated with sqrt data, but evaluated on the original untransformed data.
	#Equation (c) from Appendix (B) in Taillardat et al. 2016. https://journals.ametsoc.org/doi/pdf/10.1175/MWR-D-15-0260.1 
        #exp1        = 2.7183
        #mu_sqrt     = par[0] + par[1]*ensmean_sqrt
        #sigma_sqrt  = np.sqrt(par[2]*np.log(exp1+par[3]*ensmean_sqrt) + par[4]*ensvar_sqrt)

	mu_sqrt    = par[0] + par[1]*ensmean_sqrt             #mean of sqrt forecast training data
	sigma_sqrt = ma.sqrt(par[2] + par[3]*ensvar_sqrt)     #stdev of sqrt forecast training data
	omega      = ((obs_sqrt - mu_sqrt)/sigma_sqrt) 
	q          = stats.norm.cdf(-mu_sqrt/sigma_sqrt) 
	p          = (1.0 - q)
	term1      = (mu_sqrt**2 + sigma_sqrt**2 - obs)
	term2      = (1.0 - 2.0*(stats.norm.cdf(omega) - q)/p)
	term3      = ( (2.0*stats.norm.pdf(omega)/p)*(omega*sigma_sqrt**2 + 2.0*sigma_sqrt*mu_sqrt)) 
	term4      = ((sigma_sqrt/p)*stats.norm.pdf(-mu_sqrt/sigma_sqrt))**2
	term5      = (((2.0*sigma_sqrt*mu_sqrt)/(p**2*np.sqrt(np.pi)))*(1.0-stats.norm.cdf(-mu_sqrt*np.sqrt(2.0)/sigma_sqrt)))
	crps       = term1 * term2 + term3 - term4 -term5 
	return ma.mean(crps)  

def crpsCondTLogistic(par,obs,ensmean,ensvar):
	#Average CRPS for truncated Logistic distribution conditional on the ensemble statistics
	#close-form equation of CRPS for truncated Logistic distribution. Scheuerer, M. and Moller, D. 2015
        #exp1        = 2.7183
        #mu          = par[0] + par[1]*ensmean
        #sigma       = np.sqrt(par[2]*np.log(exp1+par[3]*ensmean) + par[4]*ensvar)

	mu         = par[0] + par[1]*ensmean
	sigma      = ma.sqrt(par[2] + par[3]*ensvar)   
	location_l = mu 
	scale_S    = (sigma*np.sqrt(3.0)/math.pi) 
	plog0      = stats.logistic.cdf(0.0,loc=location_l,scale=scale_S)
	plogy      = stats.logistic.cdf(obs,loc=location_l,scale=scale_S) 	
	
	#Need to do some complicated indexing here to ensure that terms properly cancel out or get overwritten during certain situations
	#When some of these scenarios happen, you may get an error inside of the np.where statements,because it evaluates the equations first 
	#rather than automatically subbing in the new value when the described condition is met(i.e.,index=True). These errors can be ignored, becuase 
	#the value at that True index gets overwritten with the conditional value. 
	ind_plogy_eq0or1             = np.logical_or((plogy==0.0),(plogy==1.0))
	ind_plog0_eq1                = (plog0 == 1.)
	ind_plog0_eq0or1             = np.logical_or((plog0==0.0),(plog0==1.0))
	
	logitpy = ma.where(ind_plogy_eq0or1, 0.0, (np.log(plogy) - np.log(1.-plogy)) )
	term1   = ma.where(ind_plog0_eq1, obs, ((obs-location_l) * ((2.*plogy-1.-plog0)/(1.-plog0))) )                #term 1 only modified if plog0=1
	term2   = ma.where(ind_plog0_eq1, 0.0, (np.log(1.-plog0)))                                                    #term 2 only modified if plog=1. 
	term3   = ma.where(ind_plogy_eq0or1, (1./(1.+plog0)),((1.+2.*np.log(1.-plogy)+2.*plogy*logitpy)/(1.-plog0)))
	term3   = ma.where(ind_plog0_eq1, 0.0,term3 )                                                                 #plog0=1 should be able to overwrite this condition             
	term4   = ma.where(ind_plog0_eq0or1,0.0, ((plog0**2*np.log(plog0))/(1.-plog0)**2) ) 	
	crps    = term1 + scale_S*(term2 - term3 - term4)
	return ma.mean(crps) #this set of parameterization values will yield a Tlogistic distribution with this mean CRPS
	
def crpsCondGamma(par,obs,ensmean,ensvar):  
        #Average CRPS for Gamma distribution conditional on the ensemble statistics
	#close-form equation of CRPS for Gamma distribution. Scheuerer, M. and Moller, D. 2015
        #exp1        = 2.7183
        #mu          = np.maximum(par[0] + par[1]*ensmean,0.001)
        #sigma       = np.sqrt(par[2]*np.log(exp1+par[3]*ensmean) + par[4]*ensvar)

	mu          = ma.maximum(par[0] + par[1]*ensmean,0.001)  #limit mu so that it doesn't go below 0.001. mu should equal k*theta, y=b+mx EMOS mean   mean of non-truncated gamma 
	sigma       = ma.sqrt(par[2] + par[3]*ensvar)            #variance = c+dS^2  std of non-truncated gamma 
	shape_k     = ((mu/sigma)**2.)                           #shape parameter
	scale_theta = ((sigma**2.)/mu)                           #scale parameter. scale = 1/rate 
	Betafunc    = sp.special.beta(shape_k+0.5,0.5)           #Beta function with shape parameters. 
	crps        = obs*(2.*stats.gamma.cdf(obs,shape_k,loc=0,scale=scale_theta)-1.) - (shape_k*scale_theta)*(2.*stats.gamma.cdf(obs,shape_k+1.,loc=0,scale=scale_theta)-1.) - (shape_k*scale_theta/math.pi)*Betafunc 
	return ma.mean(crps)

def crpsCondGEV(par,obs,ensmean,ensvar):  #!!!! crps value is actually min log score. 
        m         = par[0] + par[1]*ensmean
        sigma     = ma.sqrt(par[2] + par[3]*ensvar)
        shape     = par[4]  #=0: Gumbel, <0:Frechet, >0:Weibull (sign is based on how python defines shape parameter)
        if shape == 0.0:
                mu = m - sigma*0.5772
        else:
             	mu = m - sigma*(sp.special.gamma(1-shape)-1.)/shape
        crps	  = -np.log(stats.genextreme.pdf(obs,shape,loc=mu,scale=sigma)) #not actually CRPS. max likelihood estimation. min log score
        return ma.mean(crps)


#---ESTIMATE 4 PARAMETERS OF THE EMOS REGRESSION. PARAMETERS TO DEFINE THE MEAN AND VARIANCE OF THE ENSEMBLE---
##################################################################################################################################
#Testing a set of regression parameter values, calculate the CRPS for each training period-ensemble mean and -observation pair.
#Pick which set of parameters yields the minimum MEAN CRPS over all the training dates. (i.e.,Minimum CRPS for all training dates).
#Recall: Training dates are for ilead and imonth for all other years except for iyear.
#Overall this yields a set of regression parameters for each month,year,leadtime,and for each grid location.
#Want to keep the minimum possible avgCRPS (avg over all training dates) you can get based on the best parameters... 
#This is helpful to pick distribution based on training data (instead of verification data)

n_param                   = 4 
maxntrain_days            = 31*(nhindcast_yrs-1) #maximum number of training dates there could be for a given month. 31daysmaxinmonth*19years
ensmean_all               = ma.mean(wspd_ensfcst_train_tmp,axis=1)
ensvar_all	          = ma.var(wspd_ensfcst_train_tmp,axis=1)
ensmean_all_sqrt          = ma.mean(wspd_ensfcst_train_tmp**Pwr,axis=1)
ensvar_all_sqrt           = ma.var(wspd_ensfcst_train_tmp**Pwr,axis=1)

par_reg_Tgaussian         = ma.array(np.zeros((n_param,nlat_anl,nlon_anl), dtype=np.float32), mask=True)
par_reg_sqrtTgaussian     = ma.array(np.zeros((n_param,nlat_anl,nlon_anl), dtype=np.float32), mask=True)
par_reg_Tlogistic         = ma.array(np.zeros((n_param,nlat_anl,nlon_anl), dtype=np.float32), mask=True)
par_reg_gamma             = ma.array(np.zeros((n_param,nlat_anl,nlon_anl), dtype=np.float32), mask=True)

min_avgcrps_Tgaussian     = ma.array(np.zeros((nlat_anl,nlon_anl), dtype=np.float32), mask=True) 
min_avgcrps_sqrtTgaussian = ma.array(np.zeros((nlat_anl,nlon_anl), dtype=np.float32), mask=True)
min_avgcrps_Tlogistic     = ma.array(np.zeros((nlat_anl,nlon_anl), dtype=np.float32), mask=True)
min_avgcrps_gamma         = ma.array(np.zeros((nlat_anl,nlon_anl), dtype=np.float32), mask=True)

PIT_Tgaussian             = ma.array(np.zeros((maxntrain_days,nlat_anl,nlon_anl),dtype=np.float32), mask=True)
PIT_sqrtTgaussian         = ma.array(np.zeros((maxntrain_days,nlat_anl,nlon_anl),dtype=np.float32), mask=True)
PIT_Tlogistic             = ma.array(np.zeros((maxntrain_days,nlat_anl,nlon_anl),dtype=np.float32), mask=True)
PIT_gamma                 = ma.array(np.zeros((maxntrain_days,nlat_anl,nlon_anl),dtype=np.float32), mask=True)

for ix in range(nlon_anl):
	for jy in range(nlat_anl):
		if DST_CONUS_MASK[jy,ix] == False:   #skip everythign below & go to next grid point if the current grid point is masked
			continue       
		ensmean            = ensmean_all[:,jy,ix].astype('float64')  
		useidx             = ensmean.mask==False   #Indicies that aren't masked...only use unmasked values with some of the functions	below
		ensmean            = ensmean[useidx]
		ensvar             = ensvar_all[useidx,jy,ix].astype('float64') 
		obs                = wspd_anl_train_tmp[useidx,jy,ix].astype('float64') #only want to use observations for which there is a corresponding reforecast.
				
		#---Transformed data...square root--- 
		obs_sqrt       	  = wspd_anl_train_tmp[useidx,jy,ix]**Pwr
		obs_sqrt       	  = obs_sqrt.astype('float64')
		ensmean_sqrt      = ensmean_all_sqrt[useidx,jy,ix].astype('float64')
		ensvar_sqrt       = ensvar_all_sqrt[useidx,jy,ix].astype('float64') 
		
		#---Define first guess and bounds of regression parameters--- 
		slope,intercept   = stats.linregress(ensmean,obs)[0:2]
		regression_fit    = intercept+ensmean*slope
		resid             = obs-regression_fit
		#First parameter guess to give minimizer function
		#Parameter "first-guess" estimates based on regression between training ensemble mean and training observations 
		pst               = [intercept, slope, ma.mean(resid**2),stats.linregress(ma.sqrt(ensvar),abs(resid))[0] ]  
           		#Use for 4 param model w/o log variance term. ,0.1*np.mean(resid**2)] #3rd one: estimate of error 4th one: slope of the regression between ensemble variance and residuals
		
		slope_sqrt,intercept_sqrt  = stats.linregress(ensmean_sqrt,obs_sqrt)[0:2]
		regression_fit_sqrt        = intercept_sqrt+ensmean_sqrt*slope_sqrt  
		resid_sqrt                 = obs_sqrt-regression_fit_sqrt
		pst_sqrt                   = [intercept_sqrt, slope_sqrt, ma.mean(resid_sqrt**2),stats.linregress(ma.sqrt(ensvar_sqrt),abs(resid_sqrt))[0] ]  
			#Use for 4 param model w/o log variance term   , 0.1*np.mean(resid_sqrt**2)] 
		
		#pyplot.scatter(ensmean,obs)
		#---Check that observations and ensemble mean forecasts are properly centered.
		#pyplot.scatter(np.arange(len(obs)), obs)
		#pyplot.scatter(np.arange(len(ensmean)), ensmean)
		
		#---4-param bounds----
		bnds                       = ((pst[0]-2,pst[0]+5), (max(0.5*pst[1],0.),max(1.5*pst[1],0.1)), (0.5*pst[2],1.5*pst[2]),(0.0,max(2.0*pst[3],0.2)) )
		bnds_sqrt                  = ((pst_sqrt[0]-2,pst_sqrt[0]+5), (max(0.5*pst_sqrt[1],0.),max(1.5*pst_sqrt[1],0.1)), (0.5*pst_sqrt[2],1.5*pst_sqrt[2]),(0.0,max(2.0*pst_sqrt[3],0.2)) )
		#------------------------------------------------------------------------------------------------
		
		#----REGRESSION PARAMETERS BASED ON MODEL FITTING. BEST PARAMETERS CHOSEN BY MINIMUM AVERAGE CRPS OVER ALL TRAINING DATES----
                ##################################################################################################################################################################################################################
		result_Tgaussian                 = minimize(crpsCondTGaussian, pst, args=(obs,ensmean,ensvar),\
							 method='L-BFGS-B', bounds=bnds, tol=1e-8,options={'maxiter':6})  #maxiter=6 for 4 params, wspd scale optimizer function that tries different parameter values until minimum CRPS is found
														 #minimizes target function in no more than 6 iterations
		par_reg_Tgaussian[:,jy,ix]       = result_Tgaussian.x						 #best-fit regression parameters associated with minimum average CRPS (over all train days)
		min_avgcrps_Tgaussian[jy,ix]     = result_Tgaussian.fun						 #min avgcrps value associated with best-fit regression parameters for the Tgaussian distribution
														 #Use min avgcrps value (avg over all locations) to determine best distribution to use. 

		result_sqrtTgaussian             = minimize(crpsCondsqrtTGaussian, pst_sqrt, args=(obs,obs_sqrt,ensmean_sqrt,ensvar_sqrt),\
                                                    method='L-BFGS-B', bounds=bnds_sqrt, tol=1e-8,options={'maxiter':6})
		par_reg_sqrtTgaussian[:,jy,ix]   = result_sqrtTgaussian.x
		min_avgcrps_sqrtTgaussian[jy,ix] = result_sqrtTgaussian.fun		
		
		result_Tlogistic                 = minimize(crpsCondTLogistic, pst, args=(obs,ensmean,ensvar),\
							method='L-BFGS-B', bounds=bnds, tol=1e-8,options={'maxiter':6})
		par_reg_Tlogistic[:,jy,ix]       = result_Tlogistic.x         #use pst_gamma if using log variance model
		min_avgcrps_Tlogistic[jy,ix]     = result_Tlogistic.fun

                result_gamma                     = minimize(crpsCondGamma, pst, args=(obs,ensmean,ensvar),\
						    method='L-BFGS-B', bounds=bnds, tol=1e-8,options={'maxiter':6})
                par_reg_gamma[:,jy,ix]           = result_gamma.x    
                min_avgcrps_gamma[jy,ix]         = result_gamma.fun  

		result_gev                       = minimize(crpsCondGEV, pst, args=(obs,ensmean,ensvar),\
                                                   method='L-BFGS-B', bounds=bnds, tol=1e-8,options={'maxiter':6})
                par_reg_gev[:,jy,ix]	         = result_gev.x
                min_avgcrps_gev[jy,ix]           = result_gev.fun
		

		#---PIT BASED ON TRAINING FORECASTS AND OBSERVATIONS---- 
		##################################################################################################################################################################################################################
		#Calculate PIT values for each distribution based on TRAINING DATA (instead of verification data...do that later for verification part). Use training data PITs to help you make a decision 
		#about which of the candidate distributions are the best. 
		#Calculate PIT based on verification data later to determine if you actually made a good predictive distribution choice. 
		#Use the distribution's best regression parameters found from training data above.
		#Here: the PIT is the predictive CDF defined by training reg params & forecasted mean/var from training data, and then CDF evaluated at the training observations


                #---PIT FOR TRUNCATED GAUSSIAN (dist shape based on training forecasts, evaluated at training obs)
                #exp1             = 2.7183
                #mu_fcst          = par_reg_Tgaussian[0,jy,ix] + par_reg_Tgaussian[1,jy,ix]*ensmean
                #sigma_fcst	  = np.sqrt(par_reg_Tgaussian[2,jy,ix]*np.log(exp1+par_reg_Tgaussian[3,jy,ix]*ensmean) + par_reg_Tgaussian[4,jy,ix]*ensvar

		mu_fcst          = par_reg_Tgaussian[0,jy,ix] + par_reg_Tgaussian[1,jy,ix]*ensmean         
		sigma_fcst	 = ma.sqrt(par_reg_Tgaussian[2,jy,ix] + par_reg_Tgaussian[3,jy,ix]*ensvar) 
		location_mu	 = mu_fcst
		scale_sigma	 = sigma_fcst  
		py               = stats.norm.cdf(obs,loc=location_mu,scale=scale_sigma)                          #eval at all training obs
		p0               = stats.norm.cdf(0.0,loc=location_mu,scale=scale_sigma)                          #eval at 0, since we truncated at 0.
		cdf_Tgaussian    = (py - p0)/(1.-p0)                                                              #How to convert CDF of a regular dist into truncated #could also use: pitTgauss_test   = stats.truncnorm.cdf(ystd,-loca$
		PIT_Tgaussian[:len(cdf_Tgaussian),jy,ix] = cdf_Tgaussian

		
		#---PIT FOR SQUARE ROOT TRUNCATED GAUSSIAN (dist shape based on training forecasts, evaluated at training obs)
                #exp1             = 2.7183
                #mu_fcst          = par_reg_sqrtTgaussian[0,jy,ix] + par_reg_sqrtTgaussian[1,jy,ix]*ensmean_sqrt
                #sigma_fcst	 = np.sqrt(par_reg_sqrtTgaussian[2,jy,ix]*np.log(exp1+par_reg_sqrtTgaussian[3,jy,ix]*ensmean_sqrt) + par_reg_sqrtTgaussian[4,jy,ix]*ensvar_sqrt)

		mu_fcst          = par_reg_sqrtTgaussian[0,jy,ix] + par_reg_sqrtTgaussian[1,jy,ix]*ensmean_sqrt
		sigma_fcst	 = ma.sqrt(par_reg_sqrtTgaussian[2,jy,ix] + par_reg_sqrtTgaussian[3,jy,ix]*ensvar_sqrt) 
		location_mu	 = mu_fcst
		scale_sigma	 = sigma_fcst   
		py               = stats.norm.cdf(obs_sqrt,loc=location_mu,scale=scale_sigma)
		p0               = stats.norm.cdf(0.0,loc=location_mu,scale=scale_sigma)      
		cdf_sqrtTgaussian= (py - p0)/(1.-p0)                                         
		PIT_sqrtTgaussian[:len(cdf_sqrtTgaussian),jy,ix] = cdf_sqrtTgaussian


                #---PIT FOR TRUNCATED LOGISTIC (dist shape based on training forecasts, evaluated at training obs)
                #exp1             = 2.7183
                #mu               = par_reg_Tlogistic[0,jy,ix] + par_reg_Tlogistic[1,jy,ix]*ensmean
                #sigma            = np.sqrt(par_reg_Tlogistic[2,jy,ix]*np.log(exp1+par_reg_Tlogistic[3,jy,ix]*ensmean) + par_reg_Tlogistic[4,jy,ix]*ensvar)

		mu_fcst          = par_reg_Tlogistic[0,jy,ix] + par_reg_Tlogistic[1,jy,ix]*ensmean         
		sigma_fcst	 = ma.sqrt(par_reg_Tlogistic[2,jy,ix] + par_reg_Tlogistic[3,jy,ix]*ensvar)  
		location_l	 = mu_fcst
		scale_S          = (sigma_fcst*np.sqrt(3.0)/math.pi)
		py               = stats.logistic.cdf(obs,loc=location_l,scale=scale_S)
		p0               = stats.logistic.cdf(0.0,loc=location_l,scale=scale_S)
		cdf_Tlogistic    = (py - p0)/(1.-p0)
		PIT_Tlogistic[:len(cdf_Tlogistic),jy,ix] = cdf_Tlogistic
		
                #---PIT FOR GAMMA (dist shape based on training forecasts, evaluated at training obs)
                #exp1             = 2.7183
                #mu_fcst          = np.maximum(par_reg_gamma[0,jy,ix] + par_reg_gamma[1,jy,ix]*ensmean,0.001)
		#sigma_fcst       = np.sqrt(par_reg_gamma[2,jy,ix]*np.log(exp1+par_reg_gamma[3,jy,ix]*ensmean) + par_reg_gamma[4,jy,ix]*ensvar)

                mu_fcst          = ma.maximum(par_reg_gamma[0,jy,ix] + par_reg_gamma[1,jy,ix]*ensmean,0.001) 
                sigma_fcst	 = ma.sqrt(par_reg_gamma[2,jy,ix] + par_reg_gamma[3,jy,ix]*ensvar) 
		shape_k          = ((mu_fcst/sigma_fcst)**2.)                 #shape parameter
                scale_theta	 = ((sigma_fcst**2.)/mu_fcst)                 #scale parameter. scale = 1/rate
                cdf_gamma        = stats.gamma.cdf(obs,shape_k,loc=0,scale=scale_theta) 
                PIT_gamma[:len(cdf_gamma),jy,ix] = cdf_gamma

                #---PIT FOR GEV (dist shape based on training forecasts, evaluated at training obs)
                m_fcst           = par_reg_gev[0,jy,ix] + par_reg_gev[1,jy,ix]*ensmean
                sigma_fcst	 = np.sqrt(par_reg_gev[2,jy,ix] + par_reg_gev[3,jy,ix]*ensvar)
                shape_fcst	 = par_reg_gev[4,jy,ix]
                if shape_fcst == 0.0:
                        mu_fcst = m_fcst - sigma_fcst*0.5772
                else:
                     	mu_fcst = m_fcst - sigma_fcst*(sp.special.gamma(1-shape_fcst)-1.)/shape_fcst
                cdf_gev          = stats.genextreme.cdf(obs,shape_fcst,loc=mu_fcst,scale=sigma_fcst)
                PIT_gev[:len(cdf_gev),jy,ix] = cdf_gev



#---POST-PROCESSED FORECASTS FOR THE VERIFICATION YEAR,MONTH,& LEADTIME---
##################################################################################################################################
qt_ensfcst_postpro        = ma.array(np.zeros((nverif,nmem_fcst,nlat_anl,nlon_anl)),mask=True)            #univariate calibrated ensemble forecasts 
var_ensfcst_raw_rank	  = ma.array(np.zeros((nverif,nmem_fcst,nlat_anl,nlon_anl)),dtype=int,mask=True)  #rank of raw forecasts
var_ensfcst_ECC           = ma.array(np.zeros((nverif,nmem_fcst,nlat_anl,nlon_anl)),mask=True)            #multivariate postprocesed ensemble forecasts via ECC-Q method.

qt_levels_crps            = (np.array(range(nmem_fcst))+0.5)/nmem_fcst    #Quantiles levels optimal for crps verification metric
ensmean_all_verif	  = ma.mean(wspd_ensfcst_verif_tmp,axis=1)
ensvar_all_verif	  = ma.var(wspd_ensfcst_verif_tmp,axis=1)
for ilat in range(nlat_anl):
	for ilon in range(nlon_anl):
		if DST_CONUS_MASK[ilat,ilon] == False:
			continue
		for ivfd in range(nverif):
			if np.all(ensmean_all_verif.mask[ivfd,:,:]):
				continue
			ensmean_verif                         = ensmean_all_verif[ivfd,ilat,ilon].astype('float64') 
			ensvar_verif                          = ensvar_all_verif[ivfd,ilat,ilon].astype('float64')
			obs_verif                             = wspd_anl_verif_tmp[ivfd,ilat,ilon].astype('float64') #only want to use observations for which there is a corresponding reforecast.
			
			#NOTE: I had already tested the different distributions above & knew that the Tlogistic distribution was the best fit. If another distribution is a 
			#      better fit for your data, then you'll need to add the distribution details for the verification data in this section.

			#---Define truncated logistic distribution given fitted parameters (from training data) and new verification ensemble mean and variance
			mu_fcst                               = par_reg_Tlogistic[0,ilat,ilon] + par_reg_Tlogistic[1,ilat,ilon]*ensmean_verif
			sigma_fcst	                      = ma.sqrt(par_reg_Tlogistic[2,ilat,ilon] + par_reg_Tlogistic[3,ilat,ilon]*ensvar_verif)  			
			location_l	                      = mu_fcst   #relate mean and standard deviation to location and scale of logistic distribution 
			scale_S                               = (sigma_fcst*np.sqrt(3.0)/math.pi)
			
			#---Form finite ensemble of post-processed forecasts by sampling quantiles of calibrated distribution 
			p0                                    = stats.logistic.cdf(0.0,loc=location_l,scale=scale_S)           #returns percentile of the value 0 evaluated at CDF of distribution
			qt_levels_adj                         = p0 + qt_levels_crps*(1-p0)                                     #quantile levels adjusted for truncation at zero.
			qt_ensfcst_postpro[ivfd,:,ilat,ilon]  = stats.logistic.ppf(qt_levels_adj,loc=location_l,scale=scale_S) #quantiles of logistic regression adjusted for truncation at zero		
			
			#---STANDARD ECC-Q METHOD: impose rank of raw forecast members on the postprocessed forecast members
			var_ensfcst_raw_rank[ivfd,:,ilat,ilon] = stats.rankdata(wspd_ensfcst_verif_tmp[ivfd,:,ilat,ilon],'ordinal')-1
			var_ensfcst_ECC[ivfd,:,ilat,ilon]      = qt_ensfcst_postpro[ivfd,var_ensfcst_raw_rank[ivfd,:,ilat,ilon],ilat,ilon]
			
			

#SAVE REGRESSION PARAMETERS TO .NC FILE WITH NAME CORRESPONDING TO VERIFICATION MONTH, YEAR, AND LEAD TIME 
##################################################################################################################################
print('Making ncfile') 
varname = 'wspdnoon'
subdir  = 'ws_localnoon/'
ncfile  = output_path + subdir + format(iyear,'02') + format(imonth,'02') + '_' + format(ileade,'02') + 'DayLeads' + '_dist_params.nc'
nc      = Dataset(ncfile,'w',format='NETCDF4') 

#Define variable dimensions
nc.createDimension('lon',nlon_anl)
nc.createDimension('lat',nlat_anl)
nc.createDimension('nparam',n_param)
nc.createDimension('nverif',nverif)
nc.createDimension('nmem',nmem_fcst)
nc.createDimension('ntrain',ntrain_yrsNdts)
nc.createDimension('maxtrain_days',maxntrain_days)

#Create variables
longitude                        = nc.createVariable('lon','d',('lon'),fill_value=-9999.0)
latitude                         = nc.createVariable('lat','d',('lat'),fill_value=-9999.0)
valid_t                          = nc.createVariable('yyyymmdd_valid_fcst','i',('nverif'),fill_value=-9999)
ECC                              = nc.createVariable('fct_ECC_verif','f',('nverif','nmem','lat','lon'),fill_value=-9999.0)
var_verifying_obs                = nc.createVariable('wspdnoon_verifying_obs','d',('nverif','lat','lon'),fill_value=-9999.0)
var_verifying_ensfcst            = nc.createVariable('wspdnoon_verifying_ensfcst','d',('nverif','nmem','lat','lon'),fill_value=-9999.0)

Tgaussian_parreg                 = nc.createVariable('par_reg_Tgaussian','d',('nparam','lat','lon'),fill_value=-9999.0)
sqrtTgaussian_parreg             = nc.createVariable('par_reg_sqrtTgaussian','d',('nparam','lat','lon'),fill_value=-9999.0)
Tlogistic_parreg                 = nc.createVariable('par_reg_Tlogistic','d',('nparam','lat','lon'),fill_value=-9999.0)
gamma_parreg                     = nc.createVariable('par_reg_gamma','d',('nparam','lat','lon'),fill_value=-9999.0)
Tgaussian_minavgcrps             = nc.createVariable('train_minavgcrps_Tgaussian','d',('lat','lon'),fill_value=-9999.0)	
sqrtTgaussian_minavgcrps         = nc.createVariable('train_minavgcrps_sqrtTgaussian','d',('lat','lon'),fill_value=-9999.0)
Tlogistic_minavgcrps             = nc.createVariable('train_minavgcrps_Tlogistic','d',('lat','lon'),fill_value=-9999.0)
gamma_minavgcrps                 = nc.createVariable('train_minavgcrps_gamma','d',('lat','lon'),fill_value=-9999.0)
Tgaussian_PIT                    = nc.createVariable('trainPIT_Tgaussian','d',('maxtrain_days','lat','lon'),fill_value=-9999.0)
sqrtTgaussian_PIT                = nc.createVariable('trainPIT_sqrtTgaussian','d',('maxtrain_days','lat','lon'),fill_value=-9999.0)
Tlogistic_PIT                    = nc.createVariable('trainPIT_Tlogistic','d',('maxtrain_days','lat','lon'),fill_value=-9999.0)
gamma_PIT                        = nc.createVariable('trainPIT_gamma','d',('maxtrain_days','lat','lon'),fill_value=-9999.0)


longitude.units                  = 'degrees_east'
latitude.units                   = 'degrees_north'
ECC.units			 = 'm/s'
var_verifying_obs.units          = 'm/s'
var_verifying_ensfcst.units      = 'm/s'
Tgaussian_minavgcrps.units       = 'm/s'
sqrtTgaussian_minavgcrps.units	 = 'm/s'
Tlogistic_minavgcrps.units       = 'm/s'
gamma_minavgcrps.units           = 'm/s'


longitude.long_name              = "longitude"
latitude.long_name               = "latitude"
valid_t.long_name                = "date of verification"
ECC.long_name                    = 'ECC ensemble forecasts for local noon wind speed'

var_verifying_obs.long_name      = "value of verifying observation"
var_verifying_ensfcst.long_name  = "value of verifying ensemble forecast"
Tgaussian_parreg.long_name       = "regression param of truncated gaussian based on min CRPS"
sqrtTgaussian_parreg.long_name   = "regression param of sqrt truncated gaussian based on min CRPS"
Tlogistic_parreg.long_name       = "regression param of truncated logistic based on min CRPS"
gamma_parreg.long_name           = "regression param of gamma based on CRPS"
Tgaussian_minavgcrps.long_name   = "min avg crps (over all training dates) associated with best reg param chosen for Tgaussian dist"
sqrtTgaussian_minavgcrps.long_name= "min avg crps (over all training dates) associated with best reg param chosen for sqrt Tgaussian dist"
Tlogistic_minavgcrps.long_name   = "min	avg crps (over all training dates) associated with best	reg param chosen for Tlogistic dist"
gamma_minavgcrps.long_name       = "min	avg crps (over all training dates) associated with best	reg param chosen for gamma dist"	
Tgaussian_PIT.long_name          = "PIT for Tgaussian dist, evaluated at each training observation"
sqrtTgaussian_PIT.long_name      = "PIT for sqrt Tgaussian dist, evaluated at each training observation"
Tlogistic_PIT.long_name          = "PIT	for Tlogistic dist,evaluated at each training observation"
gamma_PIT.long_name              = "PIT	for gamma dist,evaluated at each training observation"

#Assign values to variables
longitude[:]                     = lons_anl
latitude[:]                      = lats_anl
valid_t[:]                       = anl_verif_dates
ECC[:]                           = var_ensfcst_ECC

var_verifying_obs[:]             = wspd_anl_verif_tmp   
var_verifying_ensfcst[:]         = wspd_ensfcst_verif_tmp           

Tgaussian_parreg[:]              = par_reg_Tgaussian
sqrtTgaussian_parreg[:]          = par_reg_sqrtTgaussian
Tlogistic_parreg[:]              = par_reg_Tlogistic
gamma_parreg[:]                  = par_reg_gamma
Tgaussian_minavgcrps[:]          = min_avgcrps_Tgaussian
sqrtTgaussian_minavgcrps[:]      = min_avgcrps_sqrtTgaussian
Tlogistic_minavgcrps[:]          = min_avgcrps_Tlogistic
gamma_minavgcrps[:]              = min_avgcrps_gamma
Tgaussian_PIT[:]                 = PIT_Tgaussian
sqrtTgaussian_PIT[:]             = PIT_sqrtTgaussian 
Tlogistic_PIT[:]                 = PIT_Tlogistic
gamma_PIT[:]	                 = PIT_gamma

#Define attributes
creationdate                     = str(datetime.datetime.now())
nc.Conventions                   = "CF-1.6"
nc.history                       = "date_created = " + creationdate + "by Rochelle Worsnop"
nc.Createdby                     = "EMOS_wspd.py"
nc.close()

print('done making .nc file for: ' + format(iyear,'02') + format(imonth,'02') + '_' + format(ileade,'02'))







