import numpy as np


input_path_GEFSv12 = '/Projects/GEFSv12_RENEWABLES/'
input_path_ERA5 = '/Projects/era5/'

output_path_QRF_fcsts = '/volumes/DataCaddy/Joseph_stuff/Project_Renewable_energy/QRF_fcsts/windSpeed/'

bounds_area = [39, 41, -106, -104]   # small rectangle around Boulder
month = 1                                  # from 1 to 12
leads = range(3,240+1,3)                   # from 3 to 240 (Day1-10 only), by 3h interval
avail_years = range(2000,2020)             # (if we want to use all years)
nens = 50                                  # size of the QRF and climatological ensemble forecasts (can be different from the size of the raw GEFS, i.e. 5 members)

## QRF hyperparameters:
n_estimators = 1000     # Number of trees in the forest. Corresponds to 'ntree' in the R package quantregForest, where the description is: Number of trees to grow
min_samples_leaf = 20   # Minimum number of samples required to be at a leaf node. Corresponds to 'nodesize' in the R package quantregForest, where the description is: Minimum size of terminal nodes
max_features = 2        # Number of features to consider when looking for the best split. Corresponds to 'mtry' in the R package quantregForest, where the description is: Number of variables randomly sampled as candidates at each split

seed = 0                # Random seed to reproduce the experiment


for lead in leads:
    
    ## Run the script:
    exec(open("/users/jbellier/Documents/Project_Renewable_energy/QRF/QRF_windSpeed_fit_and_predict.py").read())

