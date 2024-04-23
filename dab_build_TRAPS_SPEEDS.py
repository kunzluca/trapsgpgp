# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.8
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# © 2024 Luca Kunz. Commercial use is subject to the terms of the source repository's license. All other commercial rights are reserved.

# # Compute TRAP translation speeds
#
# We load the yearly TRAPS pkl files, stack them to one file and calculate the translation speeds $c_x$ and $c_y$ of every TRAP instance.  
# Then, we export the 20-years dataframe and the yearly dataframes again. The yearly files will be further processed in the next script.
#
# Since the computation is very time-consuming, we do not apply it in the actual tracking algorithm but in this separate script.  
# Runtime ~14hours

# +
import os
import sys
import numpy as np
from scipy.interpolate import interp2d
import pandas as pd
import time
import datetime
import pickle

# import metpy.calc as mpcalc
# from metpy.units import units

from IPython.display import display, Audio
# import jupyter notebook files like regular python modules
import import_ipynb
from aa_define_classes import TRAPSdata
# -

# measure the computation time for the entire script
start_script_timer = time.perf_counter()


# # Preliminary

# ## Pickle object saving function

def save_object(obj, filename):
    with open(filename, 'wb') as outp:  # overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)


# ## Files and paths

# +
# if script is running in jupyter lab
if sys.argv[0].endswith("ipykernel_launcher.py"):
    # set the velocity product
    vel_product_ID = 1
    epsilon_ID = 1
    notebook_run = True
    # save_fig = True
    save_fig = False

    
# if script is running as python script
else:
    # read in product from bash
    vel_product_ID = int(sys.argv[1])
    # read in epsilon from bash
    epsilon_ID = int(sys.argv[2])
    notebook_run = False
    save_fig = True


vel_product_short = ['ENSRYS_24HM', 'MULTIOBS_24HI', 'MULTIOBS_24HM', 'SEALEVEL_24HI'][vel_product_ID]

vel_product_long = ['CMEMS GLOBAL_REANALYSIS_PHY_001_031 ENSEMBLE MEAN (1/4°, 24HM)', 
                    'CMEMS MULTIOBS_GLO_PHY_REP_015_004 (1/4°, 24HI)', 
                    'CMEMS MULTIOBS_GLO_PHY_REP_015_004 (1/4°, 24HM)', 
                    'CMEMS SEALEVEL_GLO_PHY_L4_MY_008_047 (1/4°, 24HI)'][vel_product_ID]


years = ['2000', '2001', '2002', '2003', '2004', '2005', '2006', '2007', '2008', '2009', 
         '2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019']

epsilon_value = [0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0][epsilon_ID]
epsilon_string = ['_e010', '_e025', '_e050', '_e075', '_e100', '_e125', '_e150', '_e175', '_e200'][epsilon_ID]
# -

# define the path to the pkl files
pkl_TRAPS_importpath = 'export_pkl/' + vel_product_short + '/20XX/'
pkl_TRAPS_exportpath = 'export_pkl/' + vel_product_short + '/'

# + [markdown] tags=[]
# # Import TRAPS DataFrames
#
# Load the yearly TRAPS dataframes and stack them together to an overall 2000-2019 dataframe.  
# This is faster than loading the original 2000-2019 dataframe.  

# +
# the lists to store the subdataframes to feed into pd.concat() after the loop
pd_TRAPS_sdfs = []

for year in years:
    
    start_timer = time.perf_counter()

    # define the pkl files to load
    pkl_TRAPS_importname = vel_product_short + epsilon_string + '_TRAPS_TRACKED_' + year + '.pkl'

    # load the pickle object for the current year
    with open(pkl_TRAPS_importpath + pkl_TRAPS_importname, 'rb') as inp:

        # cdf for current DataFrame, this will be overwritten every loop
        pd_TRAPS_sdfs.append(pickle.load(inp).pd_TRAPS_df.copy())

    stop_timer = time.perf_counter()
    print('loaded ' + pkl_TRAPS_importname + f' in: {stop_timer - start_timer:0.4f} seconds')

    
# stack the yearly dataframes
pd_TRAPS_df = pd.concat(pd_TRAPS_sdfs, copy=True)    
    
# reset the index after stacking
pd_TRAPS_df.reset_index(drop=True, inplace=True)

# save memory
del pd_TRAPS_sdfs
# -

# print check
pd_TRAPS_df

# # Calculate translation speeds
#
# We compute translation speeds for every instance of a TRAP trajectory. Therefore we choose all TRAPs that persist for at least three days and average the forward and backward shifted velocity at a current timestamp. The forward/backward shifted velocity is the distance to its succeeding/preceding position divided by the time lapsed between both instances, respectively. This way we deliberately create no velocities at the start and end of a trajectory and do not gain propagation speeds for trajectories of two days lifetime. In turn, we obtain translation speeds at individual TRAP instances which we consider more accurate than taking one one of the shifted velocities or even the full distance travelled by a TRAP divided by the respective lifetime. For this, we iterate through the unique origin IDs, extract subdataframes, calculate speeds and reassign them to the overall dataframe. We assign TRAPS with lifetime below 3 days a translation speed of nan.

# +
# introduce translation speed columns
# use np.nan for TRAPS that have no velocity information since they only existed less than 3 days
# TRAPS which persist at least 3 days will also have nan values at the start and end positions
pd_TRAPS_df['core_U'] = np.nan
pd_TRAPS_df['core_V'] = np.nan

# and get the numpy arrays
core_Us = pd_TRAPS_df.core_U.to_numpy()
core_Vs = pd_TRAPS_df.core_V.to_numpy()

# +
# columns to arrays
origin_IDs = pd_TRAPS_df.origin_ID.to_numpy()
lifetimes = pd_TRAPS_df.lifetime.to_numpy()

# get the unique set of trajectory labels
origin_IDs_unique = np.unique(origin_IDs)
number_of_originIDs = origin_IDs_unique.size
# -

start_timer = time.perf_counter()

# +
# the timedelta between positions, since TRAPS are only tracked over 
# consecutive daily timesteps we can set this a constant value
timedelta = 86400 # in seconds

# only a few iterations for test runs
originID_indices = range(100) if notebook_run else range(number_of_originIDs)

for coriginID_index in originID_indices: # iterate through every trajectory

    corigin_ID_filter = (origin_IDs==origin_IDs_unique[coriginID_index])

    # don't run this on two- or one-day TRAPS, those will remain with nan velocities
    if np.mean(lifetimes[corigin_ID_filter])<3:
        print('skipped    translation speed for originID index ' 
              + str(coriginID_index+1).zfill(len(str(number_of_originIDs))) + '/' + str(number_of_originIDs))
        continue
    
    
    # for the computation we only need a subset of the dataframe
    # this would be faster with numpy arrays but with dataframes it becomes more clear
    # only load the stuff relevant for calculations
    pd_TRAPS_cdf = pd_TRAPS_df.loc[corigin_ID_filter, ['core_lon', 'core_lat']].copy()
    
    # make extra columns with coordinates of the previous timestamp
    pd_TRAPS_cdf['previous_core_lon'] = pd_TRAPS_cdf['core_lon'].shift(1)
    pd_TRAPS_cdf['previous_core_lat'] = pd_TRAPS_cdf['core_lat'].shift(1)

    # calculate the zonal distance to the previous position, positive Eastward in meters, np.cos(radians)
    pd_TRAPS_cdf['dist_to_previous_core_lon'] = (1852*60*np.cos((pd_TRAPS_cdf.core_lat+pd_TRAPS_cdf.previous_core_lat)/2*np.pi/180)*
                                                 (pd_TRAPS_cdf.core_lon-pd_TRAPS_cdf.previous_core_lon))

    # calculate the meridional distance to the previous position, positive Northward in meters
    pd_TRAPS_cdf['dist_to_previous_core_lat'] = 1852*60*(pd_TRAPS_cdf.core_lat - pd_TRAPS_cdf.previous_core_lat)

    # calculate the forward shifted speed over ground and shift it back
    pd_TRAPS_cdf['forward_shifted_core_U'] = pd_TRAPS_cdf.dist_to_previous_core_lon / timedelta # in metres per second
    pd_TRAPS_cdf['backward_shifted_core_U'] = pd_TRAPS_cdf.forward_shifted_core_U.shift(-1)
    
    # the real mean velocity at the TRAP position is the mean of both forward and backward shifted velocities
    # this deliberately creates nan values at the start and end positions
    pd_TRAPS_cdf['core_U'] = (pd_TRAPS_cdf.forward_shifted_core_U + pd_TRAPS_cdf.backward_shifted_core_U) / 2

    
    pd_TRAPS_cdf['forward_shifted_core_V'] = pd_TRAPS_cdf.dist_to_previous_core_lat / timedelta # in metres per second
    pd_TRAPS_cdf['backward_shifted_core_V'] = pd_TRAPS_cdf.forward_shifted_core_V.shift(-1)

    pd_TRAPS_cdf['core_V'] = (pd_TRAPS_cdf.forward_shifted_core_V + pd_TRAPS_cdf.backward_shifted_core_V) / 2
    
    
    # update the respective column arrays of the overall dataframe
    core_Us[corigin_ID_filter] = pd_TRAPS_cdf.core_U.to_numpy()
    core_Vs[corigin_ID_filter] = pd_TRAPS_cdf.core_V.to_numpy()
    
    
    print('calculated translation speed for originID index ' 
          + str(coriginID_index+1).zfill(len(str(number_of_originIDs))) + '/' + str(number_of_originIDs))

    
# and finally update the overall dataframe
pd_TRAPS_df['core_U'] = core_Us
pd_TRAPS_df['core_V'] = core_Vs

# derive velocity magnitude and direction
# pd_TRAPS_df['core_speed'] = np.array(mpcalc.wind_speed(core_Us * units('m/s'), core_Vs * units('m/s')))
# pd_TRAPS_df['core_direction'] = np.array(mpcalc.wind_direction(core_Us * units('m/s'), core_Vs * units('m/s'), convention='to'))

# save memory
del pd_TRAPS_cdf
# -

stop_timer = time.perf_counter()
print(f'determined TRAP translation speeds in : {stop_timer - start_timer:0.4f} seconds')

# print check
pd_TRAPS_df

# + [markdown] tags=[]
# # Export 20-years pickle file
#
# Write the final 20-years TRAPs dataframe to one pickle file.
# -

# create the objects
TRAPS_data = TRAPSdata(vel_product_short, vel_product_long, pd_TRAPS_df)

# +
start_timer = time.perf_counter()

# build the filenames
pkl_TRAPS_exportname = vel_product_short + epsilon_string + '_TRAPS_SPEEDS_0019.pkl'

# save the object as .pkl file                
save_object(TRAPS_data, pkl_TRAPS_exportpath + pkl_TRAPS_exportname)

stop_timer = time.perf_counter()
print('saved ' + pkl_TRAPS_exportname + f' in: {stop_timer - start_timer:0.4f} seconds')

# + [markdown] tags=[]
# # Export yearly pickle files
#
# For every year, copy the respective TRAP objects to a current dataframe and export this dataframe as pkl file.  
# The yearly files are more handy for analysis.

# +
for year in years:

    start_timer = time.perf_counter()
    
    # filter dataframe for objects of the current year
    cyear_filter = (pd_TRAPS_df.time.dt.year == int(year))
    
    # extract TRAP objects of the current year to a current dataframe
    pd_TRAPS_cdf = pd_TRAPS_df[cyear_filter].copy()
    
    # reset the index to avoid errors later
    pd_TRAPS_cdf.reset_index(drop=True, inplace=True)
    
    # construct the export file name
    pkl_TRAPS_exportname = vel_product_short + epsilon_string + '_TRAPS_SPEEDS_' + year + '.pkl'
    
    # overwrite any other export path, yearly files always go here
    pkl_TRAPS_exportpath = 'export_pkl/' + vel_product_short + '/20XX/'
    
    # create the object
    TRAPS_data = TRAPSdata(vel_product_short, vel_product_long, pd_TRAPS_cdf)
    
    # save the object as .pkl file
    save_object(TRAPS_data, pkl_TRAPS_exportpath + pkl_TRAPS_exportname)


    stop_timer = time.perf_counter()
    print('saved ' + pkl_TRAPS_exportname + f' in: {stop_timer - start_timer:0.1f} seconds')
    
    
# -

# print check
pd_TRAPS_cdf

# ## End sound

# measure the computation time for the entire script
stop_script_timer = time.perf_counter()
print(f'overall computation time: {stop_script_timer - start_script_timer:0.3f} seconds')

# +
#https://gist.github.com/tamsanh/a658c1b29b8cba7d782a8b3aed685a24

framerate = 4410
play_time_seconds = 1

t = np.linspace(0, play_time_seconds, framerate*play_time_seconds)
# G-Dur
#audio_data = np.sin(2*np.pi*391*t) + np.sin(2*np.pi*493*t) + np.sin(2*np.pi*587*t)
# D-Dur
audio_data = np.sin(2*np.pi*293*t) + np.sin(2*np.pi*369*t) + np.sin(2*np.pi*440*t)
Audio(audio_data, rate=framerate, autoplay=True)
