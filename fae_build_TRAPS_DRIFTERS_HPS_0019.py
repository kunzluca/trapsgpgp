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

# # Produce a statistics dataframe about (hyperbolic) drifter-TRAP pairs
#
# We first load the PROXIMITY, the PTRT and the TRAPS GPGP pkl files. Then we group all drifter rows by their pair ID in order to obtain a dataframe with one drifter-TRAP pair per row. Each pair will be assigned a range of old and new attributes like e.g. the lifetime of the associated TRAP, the TRAP age at first encounter, an indicator if measurements of the vorticity curve are available for all pair instances as well as arrays of the core attractions and quadrupole orders a pair involves. With this new dataframe we can analyse the characteristics of drifter-TRAP pairs from many perspectives and produce statistics. Since, among other things, we are looking for the conditions that cause hyperbolic drifter motion, we call this final dataframe **hyperbolic pair statistics (HPS)**.

# +
import os
import sys
import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import pearsonr
import pandas as pd
import time
import datetime
import pickle

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
    drogue_ID = 1
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
    # read in drogue state from bash
    drogue_ID = int(sys.argv[4])
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

gdp_product_ID = 2 # set this apriori
gdp_product_short = ['GDP_1HI', 'GDP_6HI', 'GDP_24HI'][gdp_product_ID]

drogue_state = ['_ALL', '_DROGUED', '_UNDROGUED'][drogue_ID]
# -

# define the path to the pkl files
pkl_PROXIMITY_importpath = 'export_pkl/' + vel_product_short + '/'
pkl_VIRCLES_importpath = 'export_pkl/' + vel_product_short + '/20XX/'
pkl_HPS_exportpath = 'export_pkl/' + vel_product_short + '/'

# + [markdown] tags=[]
# # Import PROXIMITY dataframe

# +
# define the pkl files to load
pkl_PROXIMITY_importname = vel_product_short + epsilon_string + '_' + gdp_product_short
pkl_PROXIMITY_importname += drogue_state + '_TRAPS_DRIFTERS_PROXIMITY_0019' + rho_kmstring + '.pkl'

# load the pickle object
with open(pkl_PROXIMITY_importpath + pkl_PROXIMITY_importname, 'rb') as inp:
    # cdf for current DataFrame, this will be overwritten every loop to save memory
    pd_PROXIMITY_df = pickle.load(inp).pd_TRAPS_df.copy()
    
# reset the index after loading and cropping
pd_PROXIMITY_df.reset_index(drop=True, inplace=True)
# -

# print check
pd_PROXIMITY_df

# + [markdown] tags=[]
# # Import PTRT dataframe

# +
# define the pkl files to load
pkl_PTRT_importname = vel_product_short + epsilon_string + '_' + gdp_product_short
pkl_PTRT_importname +=  drogue_state + '_TRAPS_DRIFTERS_PTRT_0019' + rho_kmstring + '.pkl'

# load the pickle object
with open(pkl_PROXIMITY_importpath + pkl_PTRT_importname, 'rb') as inp:
    # cdf for current DataFrame, this will be overwritten every loop to save memory
    pd_PTRT_df = pickle.load(inp).pd_TRAPS_df.copy()
        
# reset the index after loading
pd_PTRT_df.reset_index(drop=True, inplace=True)
# -

# extract origin IDs from pair IDs
pd_PTRT_df.insert(1, 'origin_ID', [cpair_ID[-16:] for cpair_ID in pd_PTRT_df.pair_ID])

# print check
pd_PTRT_df

# + [markdown] tags=[]
# # Import TRAPS GPGP DataFrame
#
# Load the yearly TRAPS GPGP dataframes and stack them together.  
# Since we actually do this to get the attributes related to the vorticity circle, we flag all related variables `_VIRCLE` for clarity.

# + tags=[]
# the lists to store the subdataframes to feed into pd.concat() after the loop
pd_VIRCLES_sdfs = []

for year in years:
    
    start_timer = time.perf_counter()
    
    # define the pkl files to load
    pkl_VIRCLES_importname = vel_product_short + epsilon_string + '_TRAPS_GPGP_' + year + '.pkl'

    # load the pickle objects
    with open(pkl_VIRCLES_importpath + pkl_VIRCLES_importname, 'rb') as inp:
        
        # selecting attributes saves memory and speeds up the loading
        pd_VIRCLES_sdfs.append(pickle.load(inp).pd_TRAPS_df[['time', 'origin_ID', 
                                                             'vircle', 'phase_shift', 
                                                             'pattern_flag_NBV', 'configu_flag_NBV', 'qorder_flag_NBV']].copy())
    
    
    stop_timer = time.perf_counter()
    print('loaded ' + pkl_VIRCLES_importname + f' in: {stop_timer - start_timer:0.1f} seconds')
    
# stack the yearly dataframes
pd_VIRCLES_df = pd.concat(pd_VIRCLES_sdfs, copy=True)

# reset the index after stacking
pd_VIRCLES_df.reset_index(drop=True, inplace=True)

# save memory
del pd_VIRCLES_sdfs
# -

# print check
pd_VIRCLES_df

# ## Reduce VIRCLES dataframe to PROXIMITY originIDs and timestamp dates
#
# From the VIRCLES dataframe, we only need the TRAP instances that also appear in the PROXIMITY dataframe.

# +
# get all timestamps, this contains duplicates, use the DatetimeIndex object type
timestamps_PROXIMITY = pd.to_datetime(pd_PROXIMITY_df.time.to_numpy(copy=True))
timestamp_dates_PROXIMITY = timestamps_PROXIMITY.date
timestamp_dates_PROXIMITY_unique = np.unique(timestamps_PROXIMITY.date)

# and get all TRAPS IDs that occur in the PROXIMITY dataframe
origin_IDs_PROXIMITY = pd_PROXIMITY_df.TRAP_ID.to_numpy(copy=True)
origin_IDs_PROXIMITY_unique = np.unique(origin_IDs_PROXIMITY)

# further below we assert that the PTRT and PROXIMITY dataframes contain the same set of origin IDs
# -

start_timer = time.perf_counter()

# +
# building a filter using dictionaries is thousand times faster than adding boolean arrays
originIDs_filter_DICT = {}
timestampdates_filter_DICT = {}

for coriginID in origin_IDs_PROXIMITY_unique:
    originIDs_filter_DICT[coriginID] = True

for cdate in timestamp_dates_PROXIMITY_unique:
    timestampdates_filter_DICT[cdate] = True
    
# use the pandas map() function to build the filter
# assigns True to every PTRT origin ID in the VIRCLES dataframe and NaN(-> False) to all others
originIDs_filter = pd_VIRCLES_df.origin_ID.map(originIDs_filter_DICT).fillna(False).to_numpy(copy=True)
timestampdates_filter = pd_VIRCLES_df.time.dt.date.map(timestampdates_filter_DICT).fillna(False).to_numpy(copy=True)

# crop dataframe
pd_VIRCLES_df = pd_VIRCLES_df[(originIDs_filter & timestampdates_filter)].copy()

# reset the index after cropping
pd_VIRCLES_df.reset_index(drop=True, inplace=True)

# +
origin_IDs_PTRT = pd_PTRT_df.origin_ID.to_numpy(copy=True)

# assert that all dataframes bear the same origin IDs
# mind that one origin ID can appear multiple times in the PTRT dataframe!
assert np.all(np.unique(origin_IDs_PTRT)==np.unique(pd_PROXIMITY_df.TRAP_ID)), 'PTRT and PROXIMITY origin IDs do not coincide'
assert np.all(np.unique(origin_IDs_PTRT)==np.unique(pd_VIRCLES_df.origin_ID)), 'PTRT and TRAPS origin IDs do not coincide'
# -

stop_timer = time.perf_counter()
print(f'task time: {stop_timer - start_timer:0.4f} seconds')

# print check
pd_VIRCLES_df

# # Enhance proximity dataframe
#
# Enhance the PROXIMITY dataframe with characteristics from the TRAPS VIRCLES dataframe.

# ## Introduce columns for additional TRAP characteristics

# use this to initialise column arrays of the right size
pd_PROXIMITY_df['vircle'] = False
pd_PROXIMITY_df['phase_shift'] = np.nan
pd_PROXIMITY_df['pattern_flag_NBV'] = np.nan
pd_PROXIMITY_df['configu_flag_NBV'] = ''
pd_PROXIMITY_df['qorder_flag_NBV'] = np.nan

# ## Columns to arrays

# +
# get all timestamps, this contains duplicates, use the DatetimeIndex object type
timestamps_VIRCLES = pd.to_datetime(pd_VIRCLES_df.time.to_numpy(copy=True))
timestamps_PROXIMITY = pd.to_datetime(pd_PROXIMITY_df.time.to_numpy(copy=True))

timestamp_dates_VIRCLES = timestamps_VIRCLES.date
timestamp_dates_PROXIMITY = timestamps_PROXIMITY.date

# +
origin_IDs_VIRCLES = pd_VIRCLES_df.origin_ID.to_numpy(copy=True)
vircles_VIRCLES = pd_VIRCLES_df.vircle.to_numpy(copy=True)
phase_shifts_VIRCLES = pd_VIRCLES_df.phase_shift.to_numpy(copy=True)
pattern_flags_VIRCLES = pd_VIRCLES_df.pattern_flag_NBV.to_numpy(copy=True)
configu_flags_VIRCLES = pd_VIRCLES_df.configu_flag_NBV.to_numpy(copy=True)
qorder_flags_VIRCLES = pd_VIRCLES_df.qorder_flag_NBV.to_numpy(copy=True)

origin_IDs_PROXIMITY = pd_PROXIMITY_df.TRAP_ID.to_numpy(copy=True)
vircles_PROXIMITY = pd_PROXIMITY_df.vircle.to_numpy(copy=True)
phase_shifts_PROXIMITY = pd_PROXIMITY_df.phase_shift.to_numpy(copy=True)
pattern_flags_PROXIMITY = pd_PROXIMITY_df.pattern_flag_NBV.to_numpy(copy=True)
configu_flags_PROXIMITY = pd_PROXIMITY_df.configu_flag_NBV.to_numpy(copy=True)
qorder_flags_PROXIMITY = pd_PROXIMITY_df.qorder_flag_NBV.to_numpy(copy=True)

# we will need the following PROXIMITY columns later
pair_IDs_PROXIMITY = pd_PROXIMITY_df.pair_ID.to_numpy(copy=True)
TRAP_lifetimes_PROXIMITY = pd_PROXIMITY_df.lifetime.to_numpy(copy=True)
TRAP_ages_PROXIMITY = pd_PROXIMITY_df.age.to_numpy(copy=True)
TRAP_core_attractions_PROXIMITY = pd_PROXIMITY_df.core_attraction.to_numpy(copy=True)
core_distances_PROXIMITY = pd_PROXIMITY_df.distance_to_core.to_numpy(copy=True)
curve_distances_PROXIMITY = pd_PROXIMITY_df.distance_to_curve.to_numpy(copy=True)
proximity_times_PROXIMITY = pd_PROXIMITY_df.proximity_time.to_numpy(copy=True)
proximity_ages_PROXIMITY = pd_PROXIMITY_df.proximity_age.to_numpy(copy=True)
# -

# ## Transfer data from VIRCLES to PROXIMITY
#
# Assign data from the VIRCLES to the PROXIMITY dataframe.  
# This takes a lot of time!

start_timer = time.perf_counter()

# +
number_of_drifter_positions = timestamps_PROXIMITY.size

# shortcut for test runs
row_indices=range(100) if notebook_run else range(number_of_drifter_positions)

for row_index in row_indices: # iterate through all drifter positions
        
    # this filter is unique
    # as we also want to compare 1HI and 6HI drifter data against 24HI TRAPS data, 
    # we have to compare dates instead of the full timestamps
    #closest_TRAP_filter = ((timestamp_dates_VIRCLES==timestamp_dates_PROXIMITY[row_index]) & 
    #                       (origin_IDs_VIRCLES==origin_IDs_PROXIMITY[row_index]))

    # this is faster
    closest_TRAP_index = np.argmax((timestamp_dates_VIRCLES==timestamp_dates_PROXIMITY[row_index]) & 
                                   (origin_IDs_VIRCLES==origin_IDs_PROXIMITY[row_index]))
    
    vircles_PROXIMITY[row_index] = vircles_VIRCLES[closest_TRAP_index]
    phase_shifts_PROXIMITY[row_index] = phase_shifts_VIRCLES[closest_TRAP_index]
    pattern_flags_PROXIMITY[row_index] = pattern_flags_VIRCLES[closest_TRAP_index]
    configu_flags_PROXIMITY[row_index] = configu_flags_VIRCLES[closest_TRAP_index]
    qorder_flags_PROXIMITY[row_index] = qorder_flags_VIRCLES[closest_TRAP_index]
    
    # show progress in terminal
    if not notebook_run:
        print('enhanced proximity dataframe at drifter position ' 
              + str(row_index).zfill(len(str(number_of_drifter_positions-1))) + '/' + str(number_of_drifter_positions-1))

    
# update the overall dataframe
pd_PROXIMITY_df['vircle'] = vircles_PROXIMITY
pd_PROXIMITY_df['phase_shift'] = phase_shifts_PROXIMITY
pd_PROXIMITY_df['pattern_flag_NBV'] = pattern_flags_PROXIMITY
pd_PROXIMITY_df['configu_flag_NBV'] = configu_flags_PROXIMITY
pd_PROXIMITY_df['qorder_flag_NBV'] = qorder_flags_PROXIMITY
# -

stop_timer = time.perf_counter()
print(f'task time: {stop_timer - start_timer:0.4f} seconds')

# print check
pd_PROXIMITY_df.columns

# # Determine reachable pairs
#
# We analyse for every drifter-TRAP pair if the given TRAP remains long enough to be reached by the drifter before it dissipates.  
# To approximate this, we divide the initial drifter distance to the TRAP core by a drifter velocity of 17 km/day, i.e. the average drifter
# speed in our dataset. If the resulting number of days is less or equal than the remaining lifetime of the TRAP at first encounter, the drifter-TRAP pair is considered *reachable*.  
# Remaining lifetime = lifetime - age. This automatically sorts out all ephemeral TRAPs with lifetime=1 day.
#
# From this we may derive how many drifter days are related to reachable drifter-TRAP pairs, i.e. are spent within a reachbale distance around a sufficiently persistent
# TRAP. From these measures we get an impression of how available drifters are to TRAP attraction.

# +
# define remaining TRAP lifetime
TRAP_remaining_lifetimes_PROXIMITY = TRAP_lifetimes_PROXIMITY - TRAP_ages_PROXIMITY

pd_PROXIMITY_df['remaining_lifetime'] = TRAP_remaining_lifetimes_PROXIMITY

# assert that there are no negative values
assert np.all(TRAP_remaining_lifetimes_PROXIMITY>=0), 'found negative remaining lifetime'

# +
# and define the days a drifter would need on average to overcome the current distance to core or curve
# we compute this for all drifter positions, however later we will only use this value at first encounter of a drifter-TRAP pair
days_to_reach_core_PROXIMITY = core_distances_PROXIMITY / 17 # 17 km/day is the average drifter speed in our dataset
days_to_reach_curve_PROXIMITY = curve_distances_PROXIMITY / 17 # 17 km/day is the average drifter speed in our dataset

pd_PROXIMITY_df['days_to_reach_core'] = days_to_reach_core_PROXIMITY
pd_PROXIMITY_df['days_to_reach_curve'] = days_to_reach_curve_PROXIMITY

# +
# now define if for a given drifter position, the closest TRAP remains long enough to be reached,
# i.e. determine if the days to reach core OR curve are less equal than the remaining lifetime
# if this is given, the drifter-TRAP pair instance is reachable
# again, apply this to all positions even though only the value at first encounter will be needed later
reachable_core_filter = (days_to_reach_core_PROXIMITY<=TRAP_remaining_lifetimes_PROXIMITY)
reachable_curve_filter = (days_to_reach_curve_PROXIMITY<=TRAP_remaining_lifetimes_PROXIMITY)

reachables_PROXIMITY = (reachable_core_filter | reachable_curve_filter)

pd_PROXIMITY_df['reachable'] = reachables_PROXIMITY
# -

# print check
pd_PROXIMITY_df[['lifetime', 'age', 'remaining_lifetime', 'days_to_reach_core', 'days_to_reach_curve', 'reachable']]

# # Create hyperbolic pair statistics dataframe
#
# Now take the PTRT and for every drifter-TRAP pair ID add metrics like 
# - the TRAP lifetime  
# - the minimum TRAP age  
# - the reachable flag at first encounter  
# - an indicator if vircles are always available
# - an array of the core attractions they involve  
# - (an array of the vorticity patterns they involve)
# - (an array of the vorticity configurations they involve)
# - an array of the vorticity quadrupole orders they involve
# - an array of the phase shifts of the individual TRAP instances
#

# prefer to create a new dataframe and keep the original PTRT one
pd_HPS_df = pd_PTRT_df.copy()

# we define the unit of the retention time as days
pd_HPS_df.rename(columns={'counts': 'retention_time'}, inplace=True)

# ## Introduce columns for additional TRAP characteristics
#
# For some attributes, already create the columns.  
# We can only make statements about vorticity patterns during a proximity time when there have been vircles computed at every TRAP instance of a drifter-TRAP pair.

# use this to initialise column arrays of the right size
pd_HPS_df['TRAP_lifetime'] = 999
pd_HPS_df['minimum_TRAP_age'] = 999
pd_HPS_df['reachable'] = False
pd_HPS_df['vircles_always_available'] = False
# more other columns will be created at runtime

# print check
pd_HPS_df

# ## Columns to arrays

# +
pair_IDs_HPS = pd_HPS_df.pair_ID.to_numpy(copy=True)
retention_times_HPS = pd_HPS_df.retention_time.to_numpy(copy=True)

# new columns
TRAP_lifetimes_HPS = pd_HPS_df.TRAP_lifetime.to_numpy(copy=True)
minimum_TRAP_ages_HPS = pd_HPS_df.minimum_TRAP_age.to_numpy(copy=True)
reachables_HPS = pd_HPS_df.reachable.to_numpy(copy=True)
vircles_always_availables_HPS = pd_HPS_df.vircles_always_available.to_numpy(copy=True)

# and initiate simple lists to collect more data
# involved_patterns = []
# involved_configus = []
involved_attractions = []
involved_qorders = []
involved_phase_shifts = []
# -

# ## Transfer data from PROXIMITY to HPS

# + tags=[]
# assign statistics from the proximity to the HPS dataframe
for ix in range(pair_IDs_HPS.size): # iterate through all drifter-TRAP pairs
    
    # current pair ID filter that fits the PROXIMITY dataframe
    cpairID_filter = (pair_IDs_PROXIMITY==pair_IDs_HPS[ix])
    
    # assert that the current pair instances are sorted in time
    assert np.all(timestamps_PROXIMITY[cpairID_filter]==np.sort(timestamps_PROXIMITY[cpairID_filter])), 'unsorted pair instances'    
    
    # extract data from the column arrays of the PROXIMITY dataframe
    TRAP_lifetimes_HPS[ix] = np.min(TRAP_lifetimes_PROXIMITY[cpairID_filter]) # max() should give the same
    minimum_TRAP_ages_HPS[ix] = np.min(TRAP_ages_PROXIMITY[cpairID_filter])
    reachables_HPS[ix] = reachables_PROXIMITY[cpairID_filter][0] # the flag at first encounter
    
    # this checks at every drifter timestep which can be smaller than the 24H TRAPS timestep
    vircles_always_availables_HPS[ix] = True if np.all(vircles_PROXIMITY[cpairID_filter]) else False    

    # involved_patterns.append(pattern_flags_PROXIMITY[cpairID_filter])
    # involved_configus.append(configu_flags_PROXIMITY[cpairID_filter])
    involved_attractions.append(TRAP_core_attractions_PROXIMITY[cpairID_filter])
    involved_qorders.append(qorder_flags_PROXIMITY[cpairID_filter])
    involved_phase_shifts.append(phase_shifts_PROXIMITY[cpairID_filter])
    

# update the overall dataframe
pd_HPS_df['TRAP_lifetime'] = TRAP_lifetimes_HPS
pd_HPS_df['minimum_TRAP_age'] = minimum_TRAP_ages_HPS
pd_HPS_df['reachable'] = reachables_HPS
pd_HPS_df['vircles_always_available'] = vircles_always_availables_HPS
# pd_HPS_df['involved_patterns'] = involved_patterns
# pd_HPS_df['involved_configus'] = involved_configus
pd_HPS_df['involved_attractions'] = involved_attractions
pd_HPS_df['involved_qorders'] = involved_qorders
pd_HPS_df['involved_phase_shifts'] = involved_phase_shifts
# -

# print check
# MIND THAT WE HAVE ONLY TRANSFERED DATA FOR 100 DRIFTERS
pd_HPS_df[vircles_always_availables_HPS].involved_qorders
sum(pd_HPS_df.reachable)/pd_HPS_df.size
pd_HPS_df

# ## Normalise retention ages
#
# This can be done super easily when knowing the total retention time.

# +
# normalisation is relative since the first age is always 1 day
# nage = (age - 1 day) / (lifetime - 1 day), division by zero is prevented by only taking 3 instance pairs
retention_nages_HPS = [(np.arange(retention_time)/(retention_time-1)) if retention_time>2 else np.array([]) for retention_time in retention_times_HPS]

pd_HPS_df['retention_nages'] = retention_nages_HPS

# +
# print check
# pd_HPS_df
# retention_nages_HPS
# -

# ## Get the original core and curve distances

# +
original_core_distances = []
original_curve_distances = []

for ix in range(pair_IDs_HPS.size): # iterate through all drifter-TRAP pairs
               
    # current pair ID filter that fits the PROXIMITY dataframe
    cpairID_filter = (pair_IDs_PROXIMITY==pair_IDs_HPS[ix])
    
    # append the characteristics of the current pair
    original_core_distances.append(core_distances_PROXIMITY[cpairID_filter])
    original_curve_distances.append(curve_distances_PROXIMITY[cpairID_filter])
            
        
# update the overall dataframe
pd_HPS_df['original_core_distances'] = original_core_distances
pd_HPS_df['original_curve_distances'] = original_curve_distances

# +
# print check
# pd_HPS_df
# original_curve_distances

# + [markdown] tags=[]
# # Export the 20-years pickle file
# -

# create the object, using the former TRAPS object
HPS_data = TRAPSdata(vel_product_short, vel_product_long, pd_HPS_df)

# +
# save the object as .pkl file
start_timer = time.perf_counter()

pkl_HPS_exportname = vel_product_short + epsilon_string + '_' + gdp_product_short + drogue_state
pkl_HPS_exportname += '_TRAPS_DRIFTERS_HPS_0019' + rho_kmstring + '.pkl'

# save the object as .pkl file                
save_object(HPS_data, pkl_HPS_exportpath + pkl_HPS_exportname)

stop_timer = time.perf_counter()

print('saved ' + pkl_HPS_exportname + f' in: {stop_timer - start_timer:0.1f} seconds')
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
