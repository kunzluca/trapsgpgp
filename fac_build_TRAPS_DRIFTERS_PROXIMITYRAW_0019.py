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

# # Compute a drifter's distance to its closest TRAP
#
# We load the ALL GDP drifters and the TRAPS pkl files and for each drifter position, we determine the closest TRAP, compute the distance to the TRAP core and to the closest point on the TRAP curve. Then we append these distance values, the closest TRAPS origin ID and other metrics to the row of the current drifter position.
#
# The resulting dataset will allow to visualise/analyse the time evolution of drifter-TRAP distances. Since the dataframe will be processed in a follow-up script, it is exported with a 'RAW' tag.
#
# Runtime ~15000 seconds

# +
import os
import sys
import numpy as np
import pandas as pd
import time
import datetime
import pickle

# #%matplotlib widget
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as clrs

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
    rho_ID = 1
    notebook_run = True
#    save_fig = True
    save_fig = False

    
# if script is running as python script
else:
    # read in product from bash
    vel_product_ID = int(sys.argv[1])
    # read in epsilon from bash
    epsilon_ID = int(sys.argv[2])
    # read in rho from bash
    rho_ID = int(sys.argv[3])
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

rho_kmvalue = [50, 75, 100, 150, 200, 250, 300][rho_ID] # in kilometres
rho_kmstring = ['_r050', '_r075', '_r100', '_r150', '_r200', '_r250', '_r300'][rho_ID] # in kilometres

gdp_product_ID = 2 # set this a priori
gdp_product_short = ['GDP_1HI', 'GDP_6HI', 'GDP_24HI'][gdp_product_ID]

gdp_product_long = ['GLOBAL DRIFTER PROGRAM DRIFTERS 1 HOURLY',
                    'GLOBAL DRIFTER PROGRAM DRIFTERS 6 HOURLY', 
                    'GLOBAL DRIFTER PROGRAM DRIFTERS 24 HOURLY'][gdp_product_ID]
# -

# define the path to the pkl files
pkl_TRAPS_importpath = 'export_pkl/' + vel_product_short + '/20XX/'
pkl_DRIFTERS_importpath = 'export_pkl/' + gdp_product_short + '/'
pkl_exportpath = 'export_pkl/' + vel_product_short + '/'

# + [markdown] tags=[]
# # Import DataFrames
#
# Load the yearly TRAPS dataframes and stack them together to an overall 2000-2019 dataframe.  
# Also load the lighter 2000-2019 drifter dataframe.
# -

# ## Load TRAPS dataframes

start_timer = time.perf_counter()

# +
# the lists to store the subdataframes to feed into pd.concat() after the loop
pd_TRAPS_sdfs = []

for year in years:
    
    start_timer = time.perf_counter()
    
    # define the pkl files to load
    pkl_TRAPS_importname = vel_product_short + epsilon_string + '_TRAPS_GPGP_' + year + '.pkl'

    # load the pickle object for the current year
    with open(pkl_TRAPS_importpath + pkl_TRAPS_importname, 'rb') as inp:
        
        # selecting attributes saves memory and speeds up the loading
        pd_TRAPS_sdfs.append(pickle.load(inp).pd_TRAPS_df[['TRAP_ID', 'time', 
                                                           'core_lon', 'core_lat', 'core_attraction', 
                                                           'curve_lons', 'curve_lats', 'curve_attractions', 
                                                           'origin_ID', 'lifetime', 'age']].copy())
        
    stop_timer = time.perf_counter()
    print('loaded ' + pkl_TRAPS_importname + f' in: {stop_timer - start_timer:0.4f} seconds')
    
# stack the yearly dataframes
pd_TRAPS_df = pd.concat(pd_TRAPS_sdfs, copy=True)        
    
# reset the index after cropping and stacking
pd_TRAPS_df.reset_index(drop=True, inplace=True)

# save memory
del pd_TRAPS_sdfs
# -

#print check
pd_TRAPS_df

# ## Load GDP drifter data

# +
# define the pkl files to load
pkl_DRIFTERS_importname = gdp_product_short + '_ALL_0019.pkl' # contains both drogued and undrogued

# load the pickle object
with open(pkl_DRIFTERS_importpath + pkl_DRIFTERS_importname, 'rb') as inp:
    # cdf for current DataFrame, this will be overwritten every loop to save memory
    pd_DRIFTERS_df = pickle.load(inp).pd_TRAPS_df[['drifter_ID', 'time', 
                                                   'drifter_lat', 'drifter_lon', 
                                                   'drifter_U', 'drifter_V', 'drogued']].copy()
    
# reset the index after cropping
pd_DRIFTERS_df.reset_index(drop=True, inplace=True)
# -

# print check
#pd_TRAPS_df
pd_DRIFTERS_df

# change drifter IDs' data type from integer to string for compatibility with TRAP IDs
pd_DRIFTERS_df['drifter_ID'] = [str(current_ID) for current_ID in pd_DRIFTERS_df.drifter_ID]

# print check
#pd_TRAPS_df
pd_DRIFTERS_df

# # Drifter to TRAP proximity algorithm
#
# This algorithm can be seen as a simplified version of the TRAPS A to B tracking algorithm, interpreting the drifters dataframe as A dataframe and the TRAPS dataframe as B dataframe.  
# The idea is to iterate through every drifter position, i.e. row in the drifters dataframe, and search in a circle of radius of rho_kmvalue around the drifter for the closest TRAP, compute their distance and copy some of the TRAP attributes to the current drifter row.
#
# The search is carried out from the drifter perspective because there are less drifters than TRAPS and one TRAP can attract multiple drifters at a time while a drifter should approach just one TRAP at a time. The algorithm also becomes lighter by cropping the TRAPS and drifter dataframes for the subdomain above and filtering the TRAPS dataframe for the available drifter timestamps/days. This way both dataframes only contain objects in the same region and are synchronised. Beware that at drifter frequencies below 24HI multiple positions of the same drifter will be compared against the same daily TRAP.
#
# What follows is the search for the closest TRAP, iterating through the drifter dataframe:
#
# - create a rho_kmvalue (kilometres) search box around the current drifter position
# - filter the TRAPS dataframe for candidates with cores within the search box
# - compute the drifter-to-TRAPcore distance (also in kilometres) for all candidates (this is simplified and could be enhanced with distance to closest curve point instead)
# - select the closest-core candidate and check if its core distance $\leq$ rho_kmvalue (this turns the search box into a search circle)
# - record its core distance, the distance to its closest curvepoint (also kilometres), its origin ID and other metrics to the current drifter row
#
# What we get is a new drifters dataframe containing distance statistics to a drifter's closest TRAP.  
# In a next script, we can search for repetitive drifter_ID - origin_ID pairs and analyse their curve/core distance evolution.  
# Will we see a certain attraction pattern in this?

# ## Synchronise dataframes
#
# Synchronise the TRAPS dataframe to the days in the drifter dataframe.  
# As the original TRAPS dataframes usually show TRAPS on every day, TRAPS and drifter dataframes can in the end imply exactly the same set of unique timestamp days.  
# But this is not guaranteed, especially when the original TRAPS dataframe is filtered for certain cutoffs. Then it may occur, that the TRAPS set of unique timestamp days is smaller and just a subset of the set of unique drifter timestamp days. The assertion below considers this case. The algorithm further below can handle drifter occurrences on days without TRAP occurrences as it leaves the TRAPS candidates dataframe empty on such days.

start_timer = time.perf_counter()

# +
# get all timestamps, this contains duplicates, use the DatetimeIndex object type
timestamps_TRAPS = pd.to_datetime(pd_TRAPS_df.time.to_numpy(copy=True))
timestamps_DRIFTERS = pd.to_datetime(pd_DRIFTERS_df.time.to_numpy(copy=True))

# computing this once will speed up the code
trapdays = timestamps_TRAPS.date

# +
# we want to further reduce the TRAPS dataframe to the available drifter timestamps/days
drifterdays_unique = np.unique(timestamps_DRIFTERS.date) # prefer it sorted

# initialise the filter for the drifter days in the overall TRAPS dataframe
sync_timestamps_filter = (trapdays==drifterdays_unique[0])

# loop through all unique drifter timestamps and build the filter
for current_drifterday in drifterdays_unique: 
    sync_timestamps_filter += (trapdays==current_drifterday)

# build the synchronised TRAPS dataframe
pd_TRAPSSYNC_df = pd_TRAPS_df[sync_timestamps_filter].copy()

# reset index to avoid indexing errors later
pd_TRAPSSYNC_df.reset_index(drop=True, inplace=True)

# also get a synched timestamp array, this could also have been achieved with the filter itself
timestamps_TRAPSSYNC = pd.to_datetime(pd_TRAPSSYNC_df.time.to_numpy(copy=True))

trapssyncdays = timestamps_TRAPSSYNC.date
trapssyncdays_unique = np.unique(trapssyncdays)

# save memory
del pd_TRAPS_df
# -

# assert that the unique trapssyncdays are at least a subset of the unique drifterdays
assert np.all(np.isin(trapssyncdays_unique, drifterdays_unique)), 'dataframe synchronisation failed'

stop_timer = time.perf_counter()
print(f'synchronised dataframes in: {stop_timer - start_timer:0.4f} seconds')

# print check
pd_DRIFTERS_df
drifterdays_unique
pd_TRAPSSYNC_df
timestamps_TRAPSSYNC
trapssyncdays
trapssyncdays_unique

# ## Introduce columns for closest TRAP
#
# Introduce columns to the drifters dataframe into which the closest TRAP metrics will be copied.

# +
# this is also the size of the column arrays later
number_of_drifters = pd_DRIFTERS_df.index.size

# use default values which will be easy to differentiate from actual values and initialises the right dtype
pd_DRIFTERS_df['TRAP_ID'] = '' # together with the timestamp this is a unique TRAP identifier
pd_DRIFTERS_df['distance_to_core'] = 999. # in kilometres
pd_DRIFTERS_df['core_lon'] = 999.
pd_DRIFTERS_df['core_lat'] = 999.
pd_DRIFTERS_df['core_attraction'] = 999.
pd_DRIFTERS_df['distance_to_curve'] = 999. # in kilometres
pd_DRIFTERS_df['closest_curvepoint_index'] = 999
pd_DRIFTERS_df['on_curve_25km'] = False
pd_DRIFTERS_df['on_curve_50km'] = False
pd_DRIFTERS_df['curve_lons'] = [np.array([]) for i in range(number_of_drifters)]
pd_DRIFTERS_df['curve_lats'] = [np.array([]) for i in range(number_of_drifters)]
pd_DRIFTERS_df['curve_attractions'] = [np.array([]) for i in range(number_of_drifters)]
pd_DRIFTERS_df['lifetime'] = 999
pd_DRIFTERS_df['age'] = 999
# -

# print check
pd_DRIFTERS_df

# + [markdown] tags=[]
# ## Columns to arrays
#
# The algorithm will work on individual drifter objects but has to write into some columns of the overall drifter dataframe. It is faster to write upon the array version of these columns and to update the respective drifter dataframe columns at the end.
# -

# the default columns of the drifter dataframe which will be modified throughout the algorithm
TRAP_IDs = pd_DRIFTERS_df.TRAP_ID.to_numpy(copy=True)
distance_to_cores = pd_DRIFTERS_df.distance_to_core.to_numpy(copy=True)
core_lons = pd_DRIFTERS_df.core_lon.to_numpy(copy=True)
core_lats = pd_DRIFTERS_df.core_lat.to_numpy(copy=True)
core_attractions = pd_DRIFTERS_df.core_attraction.to_numpy(copy=True)
distance_to_curves = pd_DRIFTERS_df.distance_to_curve.to_numpy(copy=True)
closest_curvepoint_indices = pd_DRIFTERS_df.closest_curvepoint_index.to_numpy(copy=True)
on_curve_25kms = pd_DRIFTERS_df.on_curve_25km.to_numpy(copy=True)
on_curve_50kms = pd_DRIFTERS_df.on_curve_50km.to_numpy(copy=True)
curve_lons = pd_DRIFTERS_df.curve_lons.to_numpy(copy=True)
curve_lats = pd_DRIFTERS_df.curve_lats.to_numpy(copy=True)
curve_attractions = pd_DRIFTERS_df.curve_attractions.to_numpy(copy=True)
lifetimes = pd_DRIFTERS_df.lifetime.to_numpy(copy=True)
ages = pd_DRIFTERS_df.age.to_numpy(copy=True)

# + [markdown] tags=[]
# ## Determine closest TRAP
#
# *Reminder*  
# The following is very similar to the TRAPS tracking algorithm, interpreting the drifter dataframe as some A dataframe and the TRAPSSYNC dataframe as some B dataframe. Find TRAPS in the B dataset that might represent a neighbour of a given A drifter. Evaluate B TRAP candidates and select the closest one to the A drifter.  
#
# - create a rho_kmvalue (kilometres) search box around the current drifter position
# - filter the TRAPS dataframe for candidates with cores within the search box
# - compute the drifter-to-TRAPcore distance (also in kilometres) for all candidates (this is simplified and could be enhanced with distance to closest curve point instead)
# - select the closest-core candidate and check if its core distance $\leq$ rho_kmvalue (this turns the search box into a search circle)
# - record its core distance, the distance to its closest curvepoint (also kilometres), its origin ID and other metrics to the current drifter row
#
# Here, it is also defined that one A drifter can only be mapped to one single B TRAP. But this is not vice versa: One B TRAP can be mapped to multiple A drifters. If no appropriate B TRAP cancandidates can be found, A drifter won't get appended any TRAPS data.
# -

start_timer = time.perf_counter()

# the boundaries of the approximate velocity domain
wbound_AVD = -160
ebound_AVD = -125
sbound_AVD = 22.5
nbound_AVD = 42.5

# +
# rho_kmvalue defines the size of the search box/circle around a given drifter,
# this parameter was parsed at the beginning and is in kilometres
# but since we have coordinates in °E/°N, we have to do some transformations

# note that zonal distances that are static in arclength space squeeze in 
# kilometre space when moving to higher latitudes, or in other words, distances that
# are static in kilometre space squeeze in arclength space when moving to lower latitudes
# at 42.5°N, 50km zonal distane transform to 0.61° zonal distance
# at 22.5°N, 50km zonal distane transform to 0.49° zonal distance

# for a rough first-hand cropping of the current TRAPs dataframe, we create a search box
# that guarantees to capture all neighbourhing TRAPs that are within a rho_kmvalue radius around a drifter
# therefore we convert the rho_kmvalue to arclength at the latitude of the northern boundary 
# which will capture 50km distances at any latitude below the northern boundary
# and since there are no TRAPs to detect beyond the northern boundary, we do not have to consider the 
# squeezing of a search box beyond it

# in numbers: the arclength distance at 42.5°N corresponding to 50km zonal distance is the upper 
# all-inclusive limit for a search box that shall capture all 50km distances within the study domain
# np.cos(radians)
rho_degvalue = rho_kmvalue * 1000 / (1852*60*np.cos(nbound_AVD * np.pi/180))

# +
# shortcut for testruns
row_indices=range(2) if notebook_run else range(number_of_drifters)

# index to iterate through the different rows, i.e. the individual drifter positions
for row_index in row_indices:

    # .iloc[] is integer-location based and will ignore the index label
    # .loc[] is label-location based
    # the following gives a series
    current_drifter = pd_DRIFTERS_df.iloc[row_index].copy()

    # filter the TRAPS dataframe for the day of the current drifter
    current_day_filter = (trapssyncdays==current_drifter.time.date())
    
    # all TRAPS on the day of the current drifter
    # using the already synchronised TRAPSSYNC dataframe will speed this up
    pd_TRAPS_cdf = pd_TRAPSSYNC_df[current_day_filter].copy()
    
    
    # the bounds of the rough search box that guarantees to capture all
    # (if present) 50km distance candidates
    wbound_rho = current_drifter.drifter_lon - rho_degvalue
    ebound_rho = current_drifter.drifter_lon + rho_degvalue
    sbound_rho = current_drifter.drifter_lat - rho_degvalue
    nbound_rho = current_drifter.drifter_lat + rho_degvalue

    # filter the TRAPS dataset for the rho box of the current drifter
    current_rho_filter = ((pd_TRAPS_cdf.core_lon.to_numpy() >= wbound_rho) & 
                          (pd_TRAPS_cdf.core_lon.to_numpy() <= ebound_rho) & 
                          (pd_TRAPS_cdf.core_lat.to_numpy() >= sbound_rho) & 
                          (pd_TRAPS_cdf.core_lat.to_numpy() <= nbound_rho))


    # get all TRAP candidates for which cores lie within the rho box
    TRAP_candidates = pd_TRAPS_cdf[current_rho_filter].copy()


    ############################################
    # Determine closest TRAP in kilometres
    ############################################


    # if there are no candidates at all, i.e. no TRAPS in the rho box of the current drifter, 
    # the candidates dataframe is empty and one can simply set the closest TRAP to None
    if TRAP_candidates.empty:     
        closest_TRAP = None

    # TRAP candidates exist
    else:
        candidates_core_lons = TRAP_candidates.core_lon.to_numpy()
        candidates_core_lats = TRAP_candidates.core_lat.to_numpy()

        # the distance between the current drifter and the TRAP candidates' cores in arclength
        core_distances_lon = candidates_core_lons-current_drifter.drifter_lon
        core_distances_lat = candidates_core_lats-current_drifter.drifter_lat
        
        # which needs to be converted in metres using the flat surface formula, np.cos(radians)
        core_distances_zonal = (1852*60*np.cos((candidates_core_lats+current_drifter.drifter_lat)/2 
                                               * np.pi/180) * core_distances_lon)
        
        core_distances_meridional = 1852*60*(core_distances_lat)

        # this is in kilometres now
        core_distances = ((core_distances_zonal**2 + core_distances_meridional**2)**0.5) / 1000
        TRAP_candidates['core_distance'] = core_distances
        
        # sort ascending the candidates by their core distance
        # ignore the very low probability of having two cores at equal distance
        TRAP_candidates.sort_values(by=['core_distance'], ascending=True, inplace=True)

        # check now if the top candidate really is within a search circle of rho_kmvalue radius
        if TRAP_candidates.iloc[0].core_distance <= rho_kmvalue:
            closest_TRAP = TRAP_candidates.iloc[0].copy() # .iloc[0] gives a pandas series
            
            # getting the curve coordinates of the closest TRAP
            closest_curve_lons = closest_TRAP.curve_lons
            closest_curve_lats = closest_TRAP.curve_lats

            # the distance between the current drifter and every curvepoint in arclength
            curve_distances_lon = closest_curve_lons-current_drifter.drifter_lon
            curve_distances_lat = closest_curve_lats-current_drifter.drifter_lat

            # which needs to be converted in metres using the flat surface formula, np.cos(radians)
            curve_distances_zonal = (1852*60*np.cos((closest_curve_lats+current_drifter.drifter_lat)/2 
                                                   * np.pi/180) * curve_distances_lon)

            curve_distances_meridional = 1852*60*(curve_distances_lat)

            # this is in kilometres now
            curve_distances = ((curve_distances_zonal**2 + curve_distances_meridional**2)**0.5) / 1000
            
            # find the index of the minimum distance
            closest_curvepoint_index = np.argmin(curve_distances)
            
            # assign as new attributes to the TRAP object, it's handy
            # even though it's a bit contradicting since these are kind of self-referencing attributes
            closest_TRAP['distance_to_curve'] = curve_distances[closest_curvepoint_index]
            closest_TRAP['closest_curvepoint_index'] = closest_curvepoint_index
            closest_TRAP['on_curve_25km'] = True if curve_distances[closest_curvepoint_index] <= 25 else False
            closest_TRAP['on_curve_50km'] = True if curve_distances[closest_curvepoint_index] <= 50 else False
            
            
        else:
            closest_TRAP = None

            
    ############################################
    # Copy TRAP metrics to drifter column arrays
    ############################################

    if type(closest_TRAP)==pd.core.series.Series: # only True if there is a closest TRAP, otherwise None
        
        # these column arrays will update the drifter dataframe at the end
        TRAP_IDs[row_index] = closest_TRAP.origin_ID # here origin ID is renamed to TRAP ID for clarity
        distance_to_cores[row_index] = closest_TRAP.core_distance
        core_lons[row_index] = closest_TRAP.core_lon
        core_lats[row_index] = closest_TRAP.core_lat
        core_attractions[row_index] = closest_TRAP.core_attraction
        
        distance_to_curves[row_index] = closest_TRAP.distance_to_curve        
        closest_curvepoint_indices[row_index] = closest_TRAP.closest_curvepoint_index
        on_curve_25kms[row_index] = closest_TRAP.on_curve_25km
        on_curve_50kms[row_index] = closest_TRAP.on_curve_50km
        curve_lons[row_index] = closest_TRAP.curve_lons
        curve_lats[row_index] = closest_TRAP.curve_lats
        curve_attractions[row_index] = closest_TRAP.curve_attractions

        lifetimes[row_index] = closest_TRAP.lifetime
        ages[row_index] = closest_TRAP.age
    
    
    # show progress in terminal
    if not notebook_run:
        print('checked ' + vel_product_short + ' TRAPS proximity to drifter position ' 
              + str(row_index+1).zfill(len(str(number_of_drifters))) + '/' + str(number_of_drifters))

    
# after looping through all drifters, update the overall drifter dataframe 
pd_DRIFTERS_df['TRAP_ID'] = TRAP_IDs
pd_DRIFTERS_df['distance_to_core'] = distance_to_cores
pd_DRIFTERS_df['core_lon'] = core_lons
pd_DRIFTERS_df['core_lat'] = core_lats
pd_DRIFTERS_df['core_attraction'] = core_attractions
pd_DRIFTERS_df['distance_to_curve'] = distance_to_curves
pd_DRIFTERS_df['closest_curvepoint_index'] = closest_curvepoint_indices
pd_DRIFTERS_df['on_curve_25km'] = on_curve_25kms
pd_DRIFTERS_df['on_curve_50km'] = on_curve_50kms
pd_DRIFTERS_df['curve_lons'] = curve_lons
pd_DRIFTERS_df['curve_lats'] = curve_lats
pd_DRIFTERS_df['curve_attractions'] = curve_attractions
pd_DRIFTERS_df['lifetime'] = lifetimes
pd_DRIFTERS_df['age'] = ages
# -

stop_timer = time.perf_counter()
print(f'searched closest TRAPS in: {stop_timer - start_timer:0.4f} seconds')

# print check
pd_DRIFTERS_df

# # Seperate dataframe into drogued and undrogued drifters

# +
# create two seperate dataframes
pd_DROGUED_df = pd_DRIFTERS_df[pd_DRIFTERS_df.drogued].copy()
pd_UNDROGUED_df = pd_DRIFTERS_df[~pd_DRIFTERS_df.drogued].copy()

# reset the index after cropping
pd_DROGUED_df.reset_index(drop=True, inplace=True)
pd_UNDROGUED_df.reset_index(drop=True, inplace=True)

# assert that dataframes only contain what they display
assert np.all(pd_DROGUED_df.drogued), 'drogued dataframe contains undrogued drifters'
assert not np.any(pd_UNDROGUED_df.drogued), 'undrogued dataframe contains drogued drifters'
# -

# print check
pd_DRIFTERS_df
#pd_DROGUED_df
#pd_UNDROGUED_df

# + [markdown] tags=[]
# # Export pickle objects
# -

# create the object, using the former TRAPS object
#DRIFTERS_data = TRAPSdata(vel_product_short, vel_product_long, pd_DRIFTERS_df)
DROGUED_data = TRAPSdata(vel_product_short, vel_product_long, pd_DROGUED_df)
UNDROGUED_data = TRAPSdata(vel_product_short, vel_product_long, pd_UNDROGUED_df)

# +
# save the object as .pkl file
start_timer = time.perf_counter()

pkl_DROGUED_exportname = vel_product_short + epsilon_string + '_' + gdp_product_short
pkl_DROGUED_exportname += '_DROGUED_TRAPS_DRIFTERS_PROXIMITYRAW_0019' + rho_kmstring + '.pkl'

pkl_UNDROGUED_exportname = vel_product_short + epsilon_string + '_' + gdp_product_short
pkl_UNDROGUED_exportname += '_UNDROGUED_TRAPS_DRIFTERS_PROXIMITYRAW_0019' + rho_kmstring + '.pkl'

# save the object as .pkl file                
save_object(DROGUED_data, pkl_exportpath + pkl_DROGUED_exportname)
save_object(UNDROGUED_data, pkl_exportpath + pkl_UNDROGUED_exportname)


stop_timer = time.perf_counter()

print(f'finished export in: {stop_timer - start_timer:0.1f} seconds')
# -

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
