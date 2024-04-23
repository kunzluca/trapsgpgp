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

# TRAPs tracking algorithm
# ==
#
# Track instantaneous TRAP detections through the dataframes of consecutive snapshots and for every TRAP trajectory, determine the lifetime and strongest $s_1$ attraction rate, i.e. peak attraction. For every instance of a trajectory, also determine the current age. Propagation speeds of TRAPs are computed in a next script due to its long runtime.
#
# If this script is excecuted as python file from terminal, the algorithm runs across the full 20-years dataset of TRAPS (runtime ~10000 seconds). If it's excecuted as jupyter notebook, it runs a over one year to facility learning and debugging (runtime ~770 seconds).
#
# Large epsilon values slow down the code.

# +
import os
import sys
import numpy as np
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
                    'SEALEVEL_GLO_PHY_L4_NRT_OBSERVATIONS_008_046'][vel_product_ID]

years = ['2000', '2001', '2002', '2003', '2004', '2005', '2006', '2007', '2008', '2009', 
         '2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019']

# only use a the 2000 data for test runs
if notebook_run: years = years[:1]

# we use an epsilon value of 0.25 throughout the study
epsilon_value = [0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0][epsilon_ID] # defines the search area for a future candidate
epsilon_string = ['_e010', '_e025', '_e050', '_e075', '_e100', '_e125', '_e150', '_e175', '_e200'][epsilon_ID]
# -

# define paths to files
pkl_importpath = 'export_pkl/' + vel_product_short + '/20XX/'

# # Import DataFrames

# ## Load TRAPS data
#
# Load the yearly TRAPS UNTRACKED dataframes and concatenate them to one 20-years dataframe.

# + tags=[]
# the lists to store the subdataframes to feed into pd.concat() after the loop
pd_TRAPS_sdfs = []

for year in years:
    
    start_timer = time.perf_counter()
    
    # define the pkl files to load
    pkl_TRAPS_importname = vel_product_short + '_TRAPS_UNTRACKED_' + year + '.pkl'
    
    # load the pickle objects
    with open(pkl_importpath + pkl_TRAPS_importname, 'rb') as inp:
        # cdf for current DataFrame
        pd_TRAPS_sdfs.append(pickle.load(inp).pd_TRAPS_df.copy())
    
    stop_timer = time.perf_counter()
    print('loaded ' + pkl_TRAPS_importname + f' in: {stop_timer - start_timer:0.1f} seconds')

              
# stack the yearly dataframes
pd_TRAPS_df = pd.concat(pd_TRAPS_sdfs, copy=True)

# reset the index after stacking
pd_TRAPS_df.reset_index(drop=True, inplace=True)

# save memory
del pd_TRAPS_sdfs
# -

# print check
pd_TRAPS_df

# # Crop velocity domain
#
# The original velocity domain considers all grid points of a given velocity product and slightly varies from product to product around the boundaries -160°E, -125°E, 22.5°N, 42.5°N which were chosen for the netCDF velocity data download. Call these the approximated velocity domain boundaries (AVD).
#
# But to have really similar domains among both velocity products SEALEVEL and MULTIOBS, we focus on the coinciding grid points of the SEALEVEL and the MULTIOBS velocity fields. After looking at the source netCDF files the limits of this coincidence velocity domain (CVD) could be determined to occur at 0.125° inside of the above AVD boundaries (-159.875°E and so on..).
#
# Now we crop the TRAPS dataset to this CVD, only allowing TRAP objects for which cores lie within the CVD boundaries. This prevents the tracking algorithm from working beyond the CVD boundaries. The bias circles approach introduced below would not suffice to prevent trajectory building beyond the CVD since the bias circle criterion can actually be met from both sides of the CVD boundaries. Cropping off all TRAP objects on/beyond the CVD boundaries finally makes the bias circle approach work only from the inner of the CVD, building trajectories only within the CVD. 
#
# Generally speaking, we just harmonise the SEALEVEL and MULTIOBS datasets to exactly the same velocity domain and directly remove TRAP objects which we won't look at anyways.
# This may slightly impact on the e.g. lifetime histograms later since the number of 1-day-living origin IDs will be reduced as many of them are living on/beyond the CVD boundaries.
#
# Further, the epsilon domain (ED) is introduced as the domain limited by the boundaries *epsilon* degrees within these CVD boundaries and will consequently be the same among both velocity products. The ED boundaries will later become important to speed up the bias circles part of the algorithm.
#
# Beware that the later HD boundaries need to be far enough away from the CVD boundaries for curvepoint histograms not to be biased by the TRAPS removal as TRAP curves can overlap into histogram grid cells. Everything should be fine when choosing HD boundaries >0.5 degrees away from the CVD boundaries as TRAP branches are set to this as maximum length.

# +
# the bounds set during the netCDF velocity field download
wbound_AVD = -160
ebound_AVD = -125
sbound_AVD = 22.5
nbound_AVD = 42.5

# the boundaries of the coincidence velocity grid
wbound_CVD = wbound_AVD + 0.125
ebound_CVD = ebound_AVD - 0.125
sbound_CVD = sbound_AVD + 0.125
nbound_CVD = nbound_AVD - 0.125

# the boundaries of the epsilon domain
wbound_ED = wbound_CVD + epsilon_value
ebound_ED = ebound_CVD - epsilon_value
sbound_ED = sbound_CVD + epsilon_value
nbound_ED = nbound_CVD - epsilon_value

# +
# get the core coordinates from the original dataset
core_lons = pd_TRAPS_df.core_lon.to_numpy()
core_lats = pd_TRAPS_df.core_lat.to_numpy()

print('original number of TRAP objects: ', str(pd_TRAPS_df.index.size))

# +
# get all TRAP cores WITHIN the CVD boundaries
# we also decide to exclude the ones on the boundaries since these won't build trajectories by construction
# and only lead to 1-day-living origin IDS, but the impact should be negligible
# by this we also omit cores computed upon the velocity grid limits which always seem a little suspicious to me
cvd_traps_filter = ((core_lons>wbound_CVD) & (core_lons<ebound_CVD) & 
                    (core_lats>sbound_CVD) & (core_lats<nbound_CVD))

# crop the velocity domain, i.e. filter for TRAP objects only within the CVD 
pd_TRAPS_df = pd_TRAPS_df[cvd_traps_filter].copy()

print('removed ' + str(sum(~cvd_traps_filter)) + ' TRAP objects on/beyond the CVD boundaries')

# +
# reset the index after cropping rows
pd_TRAPS_df.reset_index(drop=True, inplace=True)

# save memory
del cvd_traps_filter
# -

# print check
pd_TRAPS_df.index.size

# # Introduce origin ID and memory columns

# +
# all we need is a unique identifier which allows us to mark the same TRAP over several snapshots
# let's relate this identifier to some properties the TRAP was having when it first emerged
# introduce its column here on the yearly level, use strings to work with one specific datatype
pd_TRAPS_df['origin_ID'] = '' # empty string as default value

# column to save moments of potential TRAP divisions, for each row initialise an empty list
# lists contain rejected B TRAP candidates
#pd_TRAPS_df['division_memory'] = [[] for i in range(pd_TRAPS_df.index.size)]

# column to save moments of potential TRAP fusions, for each row initialise an empty list
# lists contain rejected A TRAP candidates
#pd_TRAPS_df['fusion_memory'] = [[] for i in range(pd_TRAPS_df.index.size)]
# -

# print check
pd_TRAPS_df

# # A to B tracking algorithm

start_timer = time.perf_counter()

# ## Get unique snapshots
#
# Get the timestamps of all snapshots.  
# Since one timestamp occurs for every individual TRAP on a snapshot, one has to filter duplicates.

# pandas uniques are returned in order of appearance
snapshots = pd.to_datetime(pd_TRAPS_df.time.unique())

# print check
snapshots[0]

# + [markdown] tags=[]
# ## Columns to arrays
#
# The algorithm will work on individual TRAP objects and subdataframes of the source dataframe but also on full columns of the source dataframe. In case of a 20-years source dataframe, these full columns may contain millions of rows. To speed up the algorithm, these columns are converted to numpy arrays which are preferred since computing on arrays is much faster than computing on dataframe columns of large datasets.
#
# In the end, the algorithm will work on TRAP objects, subdataframes and these arrays, but will only update the full source dataframe instead of computing upon it.

# +
# TRAP_ID and time are static columns, won't be changed throughout the algorithm
# origin_ID column will be modified throughout the algorithm
TRAP_IDS = pd_TRAPS_df.TRAP_ID.to_numpy()
timestamps = pd.to_datetime(pd_TRAPS_df.time.to_numpy()) # better use this DatetimeIndex format
origin_IDs = pd_TRAPS_df.origin_ID.to_numpy() # object dtype
#division_memories = pd_TRAPS_df.division_memory.to_numpy()
#fusion_memories = pd_TRAPS_df.fusion_memory.to_numpy()

# update the coordinate arrays, overwrite the uncropped ones
core_lons = pd_TRAPS_df.core_lon.to_numpy()
core_lats = pd_TRAPS_df.core_lat.to_numpy()

# +
# print check
#TRAP_IDS
#origin_IDs
#timestamps
#snapshots
#(timestamps==snapshots[0]) # automatically returns numpy array
#timestamps.strftime('%Y%m%d%H%M') + ' ' 
#division_memories
#fusion_memories

# + [markdown] tags=[]
# ## Select snapshots A and B
#
# Iterate over all snapshots of a given year.  
# If tracking across multiple years, the source dataframe has to contain all these years to handle transitions from year to year.
#
# ## Search B TRAP candidates to match A TRAP
#
# Find TRAPS in the B dataset that might represent a future version of a given A TRAP.  
# If no appropriate B TRAP cancandidates can be found, A TRAP has its last occurrence in the A snapshot and will not pass on its origin ID.
#
# ## Determine B TRAP
#
# Evaluate B TRAP candidates and select the most reasonable one to represent A TRAP in the future.  
# This is where the selection criterion is defined and the algorithm can be improved.  
# First, filter TRAPS for the ones in the epsilon-box around the A TRAP.  
# Second, sort the candidates by their core distances to the A TRAP(, core attraction and the mean attraction along the curve).  
# Third, select the candidate at the top position of the sorting as future B TRAP.
#
# Here, it is also defined that one A TRAP can only be mapped to one single B TRAP. If in the B snapshot e.g. three TRAPS emerge around the position of A TRAP, only one of them is considered the evolution of A TRAP while the other two are considered new TRAPS.  
#
# Vice versa, if in the A snapshot multiple close TRAPS occur and seem to coincide into one B TRAP, this B TRAP may only get passed on the origin ID of the most similar A TRAP, using the same similarity criterion as before, just backwards checking. The other two A TRAPS then have to end in the A snapshot. This operation is always performed when a current B TRAP candidate already bears an origin ID from some previous A TRAP. Hence, always two A TRAPS will be compared against the criterion. The unsuccessfull A TRAP which does not pass on its origin ID will pass its origin ID into a fusion memory column to be able to analyse TRAP fusions later.
#
# ## Prevent B TRAP bias
#
# As soon as an A TRAP is located within epsilon degrees from one of the velocity domain (VD) boundaries, a biased tracking of the future B TRAP is possible. In this situation, the epsilon search box will touch or cross the VD boundaries and may not capture the full set of potential B TRAP candidates since on or beyond the VD boundaries no TRAPS are computed by the matlab algorithm. As a consequence, the tracking algorithm might pick a B TRAP candidate which is the top candidate of the incomplete set of B TRAP candidates but actually isn't the future version of the A TRAP and thus might lead to a wrong trajectory estimation.  
#
# To prevent this, we introduce bias circles for which the radii indicate the A TRAP's distance to all four sides of the velocity domain. An unbiased B TRAP candidate must then lie within all four bias circles, i.e. its distance to the A TRAP must be smaller than all of the bias circle radii. This is almost always the case and can only be broken within epsilon degrees from the VD boundaries. Note that we only consider core distances since the tracking algorithm will only track TRAP cores.
#
# If the distance between B TRAP and A TRAP cores is greater than any of the bias circle radii, the real B TRAP candidate might hide behind a VD boundary but cannot be seen by the algorithm since it wasn't computed at all. The instead-chosen B TRAP candidate then might actually be a biased one and should be ignored. In this case, we just don't assign a B TRAP to the current A TRAP and let its origin ID end in the current A snapshot.
#
# The epsilon domain (ED) is defined above to be limited by the boundaries *epsilon* degrees within these CVD boundaries and will is the same among both velocity products. Tracking biases may then only occur if an A TRAP is on or beyond an ED boundary.
#
#
# ## Assign the unique origin ID
#
# Some new TRAP which is just emerging in the A snapshot will have no origin ID yet, the origin ID will be NaN.  
# But if a TRAP in A has already existed in pervious snapshots, its origin ID has been set by the previous iterations.  
# In the latter case, the A TRAP's origin ID remains unchanged and will only be passed on to the - if existing - B TRAP candidate.  
# This way, some TRAP's origin ID can propagate through several snapshots and mark a long-living TRAP.
#
# The origin ID of a new TRAP is composed of its creation timestamp and the TRAP ID at this timestamp since together they build a unique identifier for any TRAP.
#
# Assign the origin IDs both to the individual objects A TRAP and B TRAP and to the A and B dataframe via updating the origin ID column. It is faster to update the subdataframes within the row iterations and the source dataframe only at the end of every timestep.
#
# Use the division and fusion memories to register incidences, always assign to the B_TRAP to register the ones that 'didn't make it to the trajectory'.
#
# For the last snapshot we can't create an A dataframe since it would have no B dataframe. This means the algorithm does not run on the last snapshot, which isn't necessary at all because TRAPS from the last snapshot have no future to propagate and be tracked into.  
# However, we also want all TRAPS from the last snapshot to bear an origin ID, i.e. either their own timestamp + TRAP ID if they emerge during the last snapshot or the origin ID from previous iterations if they stem from another TRAP. 
# The latter is achieved by the algorithm itself, the first we can construct for all B TRAPS that after the last time loop have no origin ID assigned yet.

# +
# flag TRAPS within the epsilon domain, these are only cores WITHIN the ED boundaries and not on the boundaries
# bias circles will then be checked for TRAPS with FALSE ED_TRAP flags only
ed_traps_filter = ((core_lons>wbound_ED) & (core_lons<ebound_ED) & 
                   (core_lats>sbound_ED) & (core_lats<nbound_ED))

# assign flags to dataframe
pd_TRAPS_df.insert(5, 'ED_TRAP', ed_traps_filter)

# save memory
del core_lons, core_lats

# +
####################################
# Select snapshots A and B
####################################

# iterate through [:-1] since we don't run the algorithm on the last snapshot
number_of_snapshots = snapshots[:-1].size
for snapshot_index_A in range(number_of_snapshots):

    snapshot_index_B = snapshot_index_A + 1
        
    
    ######################################
    # Select snapshots A and B
    ######################################
    
    
    # filters for TRAPS occuring on the snapshot dates
    time_filter_A = (timestamps==snapshots[snapshot_index_A]) # automatically returns numpy array
    time_filter_B = (timestamps==snapshots[snapshot_index_B]) # automatically returns numpy array

    # dataframes of the A and B snapshots
    pd_TRAPS_A_df = pd_TRAPS_df[time_filter_A].copy()
    pd_TRAPS_B_df = pd_TRAPS_df[time_filter_B].copy()

    # reset the index otherwise it would have jumps and might throw errors later
    pd_TRAPS_A_df.reset_index(drop=True, inplace=True)
    pd_TRAPS_B_df.reset_index(drop=True, inplace=True)

    # get column arrays of the snapshot dataframes
    TRAP_IDS_A = pd_TRAPS_A_df.TRAP_ID.to_numpy()
    TRAP_IDS_B = pd_TRAPS_B_df.TRAP_ID.to_numpy()
    origin_IDs_A = pd_TRAPS_A_df.origin_ID.to_numpy()
    origin_IDs_B = pd_TRAPS_B_df.origin_ID.to_numpy()
    
    ################################################
    # Search B TRAP candidates to match A TRAP
    ################################################


    # index to iterate through the different rows, i.e. the individual A TRAPS since there's one TRAP per row
    # row_index_A and TRAP_ID are not necessarily the same due to TRAPS removal in previous processing steps
    for row_index_A in range(pd_TRAPS_A_df.index.size):

        # .iloc[] is integer-location based and will ignore the index label
        # .loc[] is label-location based
        # the following gives a series
        A_TRAP = pd_TRAPS_A_df.iloc[row_index_A].copy()

        # assert that the current TRAP ID is below 999 otherwise one has to change the padding of the origin ID
        assert A_TRAP.TRAP_ID <= 999, 'padding of the origin ID will be insufficient'

        # epsilon defines the size of the search box around a given TRAP, in degrees
        # this parameter was parsed at the beginning and can be used for a senstivity study

        # the bounds of the epsilon search box
        wepsilon_bound = A_TRAP.core_lon - epsilon_value # West
        eepsilon_bound = A_TRAP.core_lon + epsilon_value # East
        sepsilon_bound = A_TRAP.core_lat - epsilon_value # South
        nepsilon_bound = A_TRAP.core_lat + epsilon_value # North

        # filter the B dataset for the epsilon box of the current A TRAP
        current_epsilon_filter = ((pd_TRAPS_B_df.core_lon.to_numpy() > wepsilon_bound) & (pd_TRAPS_B_df.core_lon.to_numpy() < eepsilon_bound) & 
                                  (pd_TRAPS_B_df.core_lat.to_numpy() > sepsilon_bound) & (pd_TRAPS_B_df.core_lat.to_numpy() < nepsilon_bound))

        # get all B TRAPS for which cores lie within the epsilon box
        B_TRAP_candidates = pd_TRAPS_B_df[current_epsilon_filter].copy()

        
        ############################################
        # Determine B TRAP
        ############################################


        # if there are no candidates at all, i.e. no B TRAPS in the epsilon box of A TRAP, the candidates dataframe is empty
        # and one can simply set B TRAP to None
        if B_TRAP_candidates.empty:     
            B_TRAP = None

        # B TRAP candidates exist
        else:
            candidates_core_lons = B_TRAP_candidates.core_lon.to_numpy()
            candidates_core_lats = B_TRAP_candidates.core_lat.to_numpy()
            candidates_core_attractions = B_TRAP_candidates.core_attraction.to_numpy()

            # the distance between the A TRAP core and some B TRAP candidate's core
            core_distances = ((candidates_core_lons-A_TRAP.core_lon)**2 + (candidates_core_lats-A_TRAP.core_lat)**2)**0.5
            B_TRAP_candidates['core_distance'] = core_distances

            # the difference in core attraction rates, since attraction values are negative use abs()
            core_attraction_differences = abs(candidates_core_attractions - A_TRAP.core_attraction)
            B_TRAP_candidates['core_attraction_difference'] = core_attraction_differences
            
            
            ###############################################################################
            # TEMPLATE FOR A DEVIATION SCORE
            #candidates_mean_curve_attractions = np.array([curve_attractions.mean() for curve_attractions in B_TRAP_candidates.curve_attractions])
            # pandas mean() automatically skips NaN values
            # the difference in mean curve attraction rate
            #mean_curve_attraction_differences = abs(candidates_mean_curve_attractions - A_TRAP.curve_attractions.mean())
            #B_TRAP_candidates['mean_curve_attraction_difference'] = mean_curve_attraction_differences            
            # introduce a deviation score to equally weight between all 3 criteria
            # normalise the values by the worst deviation in the candidates dataframe
            #core_distances_HAT = core_distances/core_distances.max()
            #core_attraction_differences_HAT = core_attraction_differences/core_attraction_differences.max()
            #mean_curve_attraction_differences_HAT = mean_curve_attraction_differences/mean_curve_attraction_differences.max()
            # and get the average normalised deviation
            #deviation_scores = (core_distances_HAT + core_attraction_differences_HAT + mean_curve_attraction_differences_HAT)/3
            #B_TRAP_candidates['deviation_score'] = deviation_scores
            #B_TRAP = B_TRAP_candidates.nsmallest(1, ['deviation_score'], keep='first').iloc[0]
            ###############################################################################

            
            # select the most reasonable B TRAP candidate
            # MAKE IT SIMPLE 
            # and sort ascending the candidates' by their core distance
            # and secondly for the rare case of equally close candidates sort ascending by core attraction difference
            B_TRAP_candidates.sort_values(by=['core_distance','core_attraction_difference'], inplace=True)

            # select the top candidate
            B_TRAP_candidate = B_TRAP_candidates.iloc[0].copy() # .iloc[0] gives a pandas series
            
            
            #################################
            # CHECK B TRAP CANDIDATE FOR BIAS
            #################################
            
            # as soon as an A TRAP is within the ED, its B TRAP candidate will always lie 
            # within all four bias circles by construction and we don't need to compute them explicitly
            if A_TRAP.ED_TRAP:
                candidate_within_bias_circles = True

            # but for TRAPS close to a VD boundary, we have to consider the bias circle radii
            else:
                wbound_bias_radius = abs(A_TRAP.core_lon - wbound_CVD)
                ebound_bias_radius = abs(A_TRAP.core_lon - ebound_CVD)
                sbound_bias_radius = abs(A_TRAP.core_lat - sbound_CVD)
                nbound_bias_radius = abs(A_TRAP.core_lat - nbound_CVD)

                # check if candidate lies within all four bias circles
                candidate_within_bias_circles = ((B_TRAP_candidate.core_distance < wbound_bias_radius) & 
                                                 (B_TRAP_candidate.core_distance < ebound_bias_radius) & 
                                                 (B_TRAP_candidate.core_distance < sbound_bias_radius) & 
                                                 (B_TRAP_candidate.core_distance < nbound_bias_radius)) 
            
            
            # if the distance between B TRAP and A TRAP cores is smaller than all four bias circle radii
            # the current B TRAP candidate has no bias and can be assigned as the future B TRAP
            if candidate_within_bias_circles:
                # select the candidate
                B_TRAP = B_TRAP_candidate.copy()

                # this filter for the candidate's position in the B dataframe will often be needed in the following
                B_TRAP_filter = (TRAP_IDS_B==B_TRAP.TRAP_ID) # since this is at one timestamp, it gives one specific element
                # this index indicates the row number in the B dataframe where the current B TRAP occurs
                B_TRAP_iloc_index = [index for index, value in enumerate(B_TRAP_filter) if value][0]
                
                # save the other candidates to the division memory
                for ix in range(B_TRAP_candidates.index.size): 
                    # we don't save the top candidate
                    if ix==0: continue
                    
                    # save the rejected candidates, i.e. the members of a potential TRAP division
                    # to clearly separate their ID from origin IDs, use another delimiter symbol D for division
                    cocandidate = B_TRAP_candidates.iloc[ix]
                    cocandidate_ID = cocandidate.time.strftime('%Y%m%d%H%M') + 'D' + str(cocandidate.TRAP_ID).zfill(3)
                    # directly update the subdataframe's division memory column if there is an entry
                    # appending elements to a list within a dataframe cell only works in this unconventional way of indexing
                    #pd_TRAPS_B_df.iloc[B_TRAP_iloc_index].division_memory.append(cocandidate_ID)
            
            # but if the distance between B TRAP and A TRAP cores is greater than any of the bias circle radii
            # the current B TRAP candidate (and all other candidates as well) might be biased and shall be irgnored                        
            else:
                B_TRAP = None


                
        #####################################
        # Assign the unique origin ID
        #####################################

        # the origin IDs of the subdataframes need to be updated every iteration
        # because an earlier A TRAP may have mapped to the same B TRAP to which the current A TRAP is mapping
        # and this double mapping needs to be resolved by comparing both A candidates now backwards against the criterion, 
        # for this we need the subdataframes with the most recent version of the origin ID column

        # if there is no origin ID yet, build and assign it
        if A_TRAP.origin_ID=='':

            # write to the individual object
            A_TRAP['origin_ID'] = A_TRAP.time.strftime('%Y%m%d%H%M') + ' ' + str(A_TRAP.TRAP_ID).zfill(3) # pad with zeros

            # write to the origin ID subarray
            origin_IDs_A[(TRAP_IDS_A==A_TRAP.TRAP_ID)] = A_TRAP.origin_ID # since this is at one timestamp, it gives one specific element
            pd_TRAPS_A_df['origin_ID'] = origin_IDs_A # update the subdataframe
            

        # if a consecutive B TRAP was found, it is assigned the origin ID of the current A TRAP
        # this also allows to propagate an origin ID from previous snapshots
        if type(B_TRAP)==pd.core.series.Series: # only True if there is a B TRAP candidate, otherwise None
                        
            # if there is no origin ID yet, just assign it
            if B_TRAP.origin_ID=='':
                # write to the individual object
                B_TRAP['origin_ID'] = A_TRAP.origin_ID
            
            # if there is already an origin ID because some previous A_TRAP was also mapped to this B_TRAP, 
            # compare both A TRAP candidates
            else:
                # choose the already mapped A TRAP via origin ID and the current A TRAP via TRAP ID
                A_TRAP_candidates = pd_TRAPS_A_df[(origin_IDs_A==B_TRAP.origin_ID) | (TRAP_IDS_A==A_TRAP.TRAP_ID)].copy()
                
                # assert that there are always only two A_TRAP candidates by construction
                assert A_TRAP_candidates.index.size==2, 'more than two recursive A TRAP candidates'
                
                ############################################
                # Select the most reasonable A TRAP
                ############################################

                # now apply the candidate selection criterion backwards from B to A
                # simply interchange 'B' and 'A' letters and overwrite arrays from the previous B candidates selection
                candidates_core_lons = A_TRAP_candidates.core_lon.to_numpy()
                candidates_core_lats = A_TRAP_candidates.core_lat.to_numpy()
                candidates_core_attractions = A_TRAP_candidates.core_attraction.to_numpy()

                # the distance between the B TRAP core and some A TRAP candidate's core
                core_distances = ((candidates_core_lons-B_TRAP.core_lon)**2 + (candidates_core_lats-B_TRAP.core_lat)**2)**0.5
                A_TRAP_candidates['core_distance'] = core_distances

                # the difference in core attraction rates, since attraction values are negative use abs()
                core_attraction_differences = abs(candidates_core_attractions - B_TRAP.core_attraction)
                A_TRAP_candidates['core_attraction_difference'] = core_attraction_differences
                
                
                ###############################################################################
                # TEMPLATE FOR A DEVIATION SCORE                
                #candidates_mean_curve_attractions = np.array([curve_attractions.mean() for curve_attractions in A_TRAP_candidates.curve_attractions])
                # the difference in mean curve attraction rate
                #mean_curve_attraction_differences = abs(candidates_mean_curve_attractions - B_TRAP.curve_attractions.mean())
                #A_TRAP_candidates['mean_curve_attraction_difference'] = mean_curve_attraction_differences
                # the deviation score
                # normalise the values by the worst deviation in the candidates dataframe
                #core_distances_HAT = core_distances/core_distances.max()
                #core_attraction_differences_HAT = core_attraction_differences/core_attraction_differences.max()
                #mean_curve_attraction_differences_HAT = mean_curve_attraction_differences/mean_curve_attraction_differences.max()
                # and get the average normalised deviation
                #deviation_scores = (core_distances_HAT + core_attraction_differences_HAT + mean_curve_attraction_differences_HAT)/3
                #A_TRAP_candidates['deviation_score'] = deviation_scores
                #origin_A_TRAP = A_TRAP_candidates.nsmallest(1, ['deviation_score'], keep='first').iloc[0]                
                ###############################################################################

                # sort and extract the most reasonable A TRAP candidate to be the origin of the current B TRAP
                A_TRAP_candidates.sort_values(by=['core_distance','core_attraction_difference'], inplace=True)

                # select the candidate
                origin_A_TRAP = A_TRAP_candidates.iloc[0]
                
                # assign the most reasonable origin ID
                B_TRAP['origin_ID'] = origin_A_TRAP.origin_ID

                # save the rejected candidate, i.e. the member of a potential TRAP fusion
                # to clearly separate its ID from origin IDs, use another delimiter symbol F for fusion
                cocandidate = A_TRAP_candidates.iloc[1]
                cocandidate_ID = cocandidate.time.strftime('%Y%m%d%H%M') + 'F' + str(cocandidate.TRAP_ID).zfill(3)
                # directly update the subdataframe's fusion memory column if there is an entry
                # appending elements to a list within a dataframe cell only works in this unconventional way of indexing
                #pd_TRAPS_B_df.iloc[B_TRAP_iloc_index].fusion_memory.append(cocandidate_ID)

                
            # write to the origin ID subarray
            origin_IDs_B[B_TRAP_filter] = B_TRAP.origin_ID
            pd_TRAPS_B_df['origin_ID'] = origin_IDs_B # update the subdataframe

            
    # after iterating through the A dataframe, update the full origin IDs array using the latest subarrays
    origin_IDs[time_filter_A] = origin_IDs_A
    origin_IDs[time_filter_B] = origin_IDs_B
    
    # also update the full memory arrays using the latest subarrays, memories were only created on the B snapshot
    #division_memories[time_filter_B] = pd_TRAPS_B_df.division_memory.to_numpy()
    #fusion_memories[time_filter_B] = pd_TRAPS_B_df.fusion_memory.to_numpy()
    
    # after one time loop finally update the source dataframe 
    # since the next A and B dataframes will be extracted from this
    pd_TRAPS_df['origin_ID'] = origin_IDs
    # this is not necessary for the memory arrays since their creation in the B snapshot does not
    # depend on memory entries in the A snapshot

    print('finished snapshot ' 
          + str(snapshot_index_A+1).zfill(len(str(number_of_snapshots))) + '/' + str(number_of_snapshots))

    
# as explained above, in the last time iteration there are B TRAPS that have not been assigned an origin ID yet, 
# these are all TRAPS that newly emerged in the last snapshot for which the origin ID needs to be constructed manually
lastnew_filter = (time_filter_B & (origin_IDs==''))

origin_IDs[lastnew_filter] = snapshots[snapshot_index_B].strftime('%Y%m%d%H%M') + ' ' # the last snapshot timestamp
origin_IDs[lastnew_filter] = origin_IDs[lastnew_filter] + [str(trapID).zfill(3) for trapID in TRAP_IDS[lastnew_filter]]

# assert that at the end, every TRAP has some origin ID
assert np.all(~(origin_IDs=='')), 'end of algorithm: found TRAP without origin ID'
# assert that at the end, every TRAP has exactly one origin ID
assert np.all([len(origin_ID)==16 for origin_ID in origin_IDs]), 'end of algorithm: found TRAP with corrupted origin ID'

# update source dataframe one last time
pd_TRAPS_df['origin_ID'] = origin_IDs
# and finally assign the finished memories
#pd_TRAPS_df['division_memory'] = division_memories
#pd_TRAPS_df['fusion_memory'] = fusion_memories

# -

stop_timer = time.perf_counter()
print(f'tracked TRAPs in: {stop_timer - start_timer:0.4f} seconds')

# +
# print check
#pd_TRAPS_A_df
#pd_TRAPS_B_df
#A_TRAP
#B_TRAP
#pd_TRAPS_df[time_filter_A | time_filter_B]
#pd_TRAPS_df[time_filter_B]
#pd_TRAPS_df.origin_ID.iloc[-1]
pd_TRAPS_df

#origin_IDs[-1].split('/')

#B_TRAP_candidates#.curve_lons.iloc[1]#[:A_TRAP.curve_lons.size]
#A_TRAP.curve_lons

#np.array([curve_attractions.mean() for curve_attractions in B_TRAP_candidates.curve_attractions])
#A_TRAP.curve_attractions.mean()
#B_TRAP_candidates

# + [markdown] tags=[]
# # Determine TRAP lifetimes and ages
#
# First, determine the lifetime of TRAPS by counting the occurrence of individual origin IDs in the dataframe.  
# This gives a persistency ranking table of origin IDs sorted by descending lifetime which shall be exported as **TRAPSVD PRT**.  
#
# Then add a lifetime column and an age column to the overall dataframe.
# -

start_timer = time.perf_counter()

# + [markdown] tags=[]
# ## Build persistency ranking table
#
# Determine the lifetime of TRAPS by counting the occurrence of individual origin IDs.  
# The tracking algorithm was designed in a way that origin IDs can only last over consecutive days without gaps.  
# Thus the number of occurrences for a given origin ID describes for how many days in a row it has persisted.  
# We also want to assign an age to every TRAP object, i.e. track the aging of a given origin ID.
# -

# to be on the safe side, reset the index of the dataframe
pd_TRAPS_df.reset_index(drop=True, inplace=True)

# +
# dataframe containing all origin IDs and their counts, i.e. lifetimes
pd_PRT_df = pd_TRAPS_df.value_counts(subset=['origin_ID']).reset_index(drop=False, inplace=False)
pd_PRT_df.rename(columns={0: 'lifetime'}, inplace=True)

# turn into arrays for later assertions
origin_IDs_PRT = pd_PRT_df.origin_ID.to_numpy()
lifetimes_PRT = pd_PRT_df.lifetime.to_numpy()

# for each origin ID, we want a LIST of ages a TRAP goes through until it reaches its lifetime
# we want lists since later we will pop() ages from these
pd_PRT_df['ages'] = [list(np.arange(lifetime)+1) for lifetime in lifetimes_PRT]
# -

# print check
pd_PRT_df

# + [markdown] tags=[]
# ## Assign lifetimes and ages to dataframe
#
# Assign all lifetimes and ages to the respective origin IDs in the overall dataframe.  
# Use the pandas mapping function and dictionaries for this since it is very very million times faster than working and assigning using boolean arrays!
#
# For this we want to turn the persistency ranking table into a dictionary which maps origin IDs to their lifetimes and age lists.
# -

# prepare the ranking dataframe
pd_PRT_MAPPING_df = pd_PRT_df.set_index('origin_ID').T.copy()

# convert the mapping dataframe into a dictionary which maps origin IDs to lifetimes and age lists
PRT_MAPPING_DICTS = pd_PRT_MAPPING_df.to_dict('records') # gives one dictionary per row, i.e. one for lifetime and one for ages

# get the individual dictionaries that map origin IDs to lifetimes or the age lists
LIFETIMES_DICT = PRT_MAPPING_DICTS[0]
AGES_DICT = PRT_MAPPING_DICTS[1]

# +
# print check
#pd_PRT_df
#pd_PRT_MAPPING_df
#PRT_MAPPING_DICTS
#LIFETIMES_DICT
#AGES_DICT
#pd_TRAPS_df

# +
# map() substitutes each value in the series with another value that is derived from the dictionary and returns a series
# the returned series represents the new lifetime column
# this approach is a million times faster than assigning through arrays!
pd_TRAPS_df['lifetime'] = pd_TRAPS_df.origin_ID.map(LIFETIMES_DICT)

lifetimes = pd_TRAPS_df.lifetime.to_numpy()

# assert that every TRAP has a lifetime value greater than 0
assert np.all(lifetimes>0), 'found TRAP without lifetime'

# -

# Now assign one age value to each occurence of a given origin ID such that an origin ID's age increases with time.  
# Since TRAP objects are ordered chronologically within the dataframe, we can go through the dataframe row by row and look up the respective age list for the current origin ID and pop() the first value.  
# The succeeding occurence of the current origin ID will then get age+1 as this is the next first value of the age list.  
# This is the fastest way of assigning ages I could find.

# +
# initiate the array for the age column, reuse the previous origin IDs array
ages = np.zeros(origin_IDs.size).astype(int)

# assign through dictionaries and save hours instead of using boolean arrays!
for ix in range(origin_IDs.size):
    
    # as an age value is removed from its age list after assignement 
    # the next call of the same origin ID will get an increment of the recently assigned age
    ages[ix] = AGES_DICT[origin_IDs[ix]].pop(0) 
    
# this shrinks the lists within the dictionary every iteration
# until finally, all lists have to be empty since all ages must have been assigned
assert not any(AGES_DICT.values()), 'found remaining age values in ages dictionary'

# assert that the number of 'full-aged' TRAPS and the number of unique origin IDs are coherent
assert sum(lifetimes==ages)==origin_IDs_PRT.size, 'found more/less full aged TRAPS than unique origin IDs'

# assign the new age column to the overall dataframe
pd_TRAPS_df['age'] = ages

# remove the age column again from the persistency ranking table since it refers to the same age lists in the memory 
# and is also affected by pop(), bearing empty lists only
del pd_PRT_df['ages']

# -

stop_timer = time.perf_counter()
print(f'determined TRAP lifetimes and ages in : {stop_timer - start_timer:0.4f} seconds')

# print check
#pd_TRAPS_df.lifetime.hist()
#pd_TRAPS_df.age.max()
#pd_TRAPS_df.head(50)
pd_TRAPS_df#[origin_IDs==origin_IDs[1]]

# + [markdown] tags=[]
# # Determine TRAP attraction metrics
#
# For every origin ID/trajectory, determine the minimum (i.e. strongest, the peak) attraction rate along the trajectory, the cumulative attraction rate along the trajectory and the mean attraction rate over the trajectory lifetime. Add these metrics to the overall dataframe and define three new rankings: 
# - the peak attraction ranking table (**ARTP**)
# - the cumulative attraction ranking table (**ARTC**)
# - and the mean attraction ranking table (**ARTM**)
#
# All three ranking tables are sorted ascending such that the origin ID with the strongest, i.e. the most negative attraction rate value is at the first position.   
# These ranking tables will be exported unsorted along with the sorted persistency ranking table within the pkl file **TRAPSVD RMT**.
#
# EVERYTHING RELATES TO CORE ATTRACTION ONLY.
#
# The groupby() approach is the fastest option to use for this task:
# 1) get a subdataframe that only contains the origin_ID and core_attraction columns
# 2) groupby() the origin ID label which for every unique origin ID determines a list of all associated core_attraction values
# 3) apply the python min(), sum() or mean() functions on each of these lists, resulting in the peak, cumulative or mean attraction rate along an origin ID, respectively
# 4) rename the resulting columns accordingly
# 5) sort ascending every ranking table
# 6) transpose the dataframe to bring it into the required form for a mapping dataframe
# 7) create the mapping dictionary
# 8) map() the origin IDs in the overall dataframe to their respective attraction value using the respective mapping dictionary
# -

start_timer = time.perf_counter()

# +
# build the ranking dataframes, dropna=True by default so we don't have to take care of NaN values entering a function
# groupby() sets the label as index
pd_ARTP_df = pd_TRAPS_df[['origin_ID', 'core_attraction']].groupby('origin_ID').min().rename(columns={'core_attraction': 's1_peak'})
pd_ARTP_df.sort_values(by=['s1_peak'], inplace=True)

#pd_ARTC_df = pd_TRAPS_df[['origin_ID', 'core_attraction']].groupby('origin_ID').sum().rename(columns={'core_attraction': 's1_cumulative'})
#pd_ARTC_df.sort_values(by=['s1_cumulative'], inplace=True)

#pd_ARTM_df = pd_TRAPS_df[['origin_ID', 'core_attraction']].groupby('origin_ID').mean().rename(columns={'core_attraction': 's1_mean'})
#pd_ARTM_df.sort_values(by=['s1_mean'], inplace=True)

# +
# build the mapping dataframes, could also be merged into one dataframe
pd_ARTP_MAPPING_df = pd_ARTP_df.T.copy()
#pd_ARTC_MAPPING_df = pd_ARTC_df.T.copy()
#pd_ARTM_MAPPING_df = pd_ARTM_df.T.copy()

# and the mapping dictionaries
S1PEAKS_DICT = pd_ARTP_MAPPING_df.to_dict('records')[0] # gives one dictionary per row
#S1CUMULATIVES_DICT = pd_ARTC_MAPPING_df.to_dict('records')[0]
#S1MEANS_DICT = pd_ARTM_MAPPING_df.to_dict('records')[0]
# -

stop_timer = time.perf_counter()
print(f'determined TRAP attraction metrics in : {stop_timer - start_timer:0.4f} seconds')

# +
# print check
#pd_ARTP_df
#pd_ARTC_df
#pd_ARTM_df

#pd_ARTP_MAPPING_df
#pd_ARTC_MAPPING_df
#pd_ARTM_MAPPING_df

#S1PEAKS_DICT
#S1CUMULATIVES_DICT
#S1MEANS_DICT
# -

# map() substitutes each value in the series with another value that is derived from the dictionary and returns a series
# the returned series represents the new s1 metric column for the overall dataframe, first save it as a numpy array
track_s1_peaks = pd_TRAPS_df.origin_ID.map(S1PEAKS_DICT).to_numpy()
#track_s1_cumulatives = pd_TRAPS_df.origin_ID.map(S1CUMULATIVES_DICT).to_numpy()
#track_s1_means = pd_TRAPS_df.origin_ID.map(S1MEANS_DICT).to_numpy()

# +
# assign s1 metrics to dataframe
pd_TRAPS_df['track_s1_peak'] = track_s1_peaks
#pd_TRAPS_df['track_s1_cumulative'] = track_s1_cumulatives
#pd_TRAPS_df['track_s1_mean'] = track_s1_means

# assert that all attraction metrics are negative
assert np.all(track_s1_peaks<0), 'found non-negative s1 metric'
#assert np.all(track_s1_cumulatives<0), 'found non-negative s1 metric'
#assert np.all(track_s1_means<0), 'found non-negative s1 metric'
# -

# print check
pd_TRAPS_df

# # Build an overall rankings table
#
# Build one table for all the different kinds of ranking metrics: lifetime, s1 peak, s1 cumulative, s1 mean.  
# Here, every unique origin ID is assigned one value for each ranking metric, defining a tracked TRAP's trajectory properties in terms of lifetime and attraction rate.
#
# Export this ranking metrics table sorted by lifetime as the pkl file **TRAPS RMT**.
#

# +
# returns a copy
# since in the PRT origin_ID is a column and not the index like for the other ranking tables, 
# origin_ID will also become a column in RANKINGS dataframe
pd_RANKINGS_df = pd_PRT_df.merge(pd_ARTP_df, how='outer', on='origin_ID')
#pd_RANKINGS_df = pd_RANKINGS_df.merge(pd_ARTC_df, how='outer', on='origin_ID')
#pd_RANKINGS_df = pd_RANKINGS_df.merge(pd_ARTM_df, how='outer', on='origin_ID')

# sort by descending lifetime
pd_RANKINGS_df.sort_values(by=['lifetime'], ascending=False, inplace=True)

# +
# save memory before writing pkl files
#del pd_PRT_df, pd_ARTP_df, pd_ARTC_df, pd_ARTM_df
# -

# print check
#pd_TRAPS_df
pd_RANKINGS_df

# + [markdown] tags=[]
# # Export 20-years pickle files
#
# Write the full 20-years TRAPs dataframe and the ranking metrics tables to one pickle file each for publication.  
# -

# create the objects
TRAPS_data = TRAPSdata(vel_product_short, vel_product_long, pd_TRAPS_df)
RANKINGS_data = TRAPSdata(vel_product_short, vel_product_long, pd_RANKINGS_df)

# +
start_timer = time.perf_counter()

# build the filenames
pkl_TRAPS_exportname = vel_product_short + epsilon_string + '_TRAPS_TRACKED_0019.pkl'
pkl_RANKINGS_exportname = vel_product_short + epsilon_string + '_TRAPS_TRACKED_0019_RMT.pkl'

# 20-years files always go here
pkl_exportpath = 'export_pkl/' + vel_product_short + '/'

# save the object as .pkl file                
save_object(TRAPS_data, pkl_exportpath + pkl_TRAPS_exportname)
save_object(RANKINGS_data, pkl_exportpath + pkl_RANKINGS_exportname)

stop_timer = time.perf_counter()
print('saved ' + pkl_TRAPS_exportname + f' in: {stop_timer - start_timer:0.4f} seconds')
print('saved ' + pkl_RANKINGS_exportname + f' in: {stop_timer - start_timer:0.4f} seconds')

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
    pkl_exportname = vel_product_short + epsilon_string + '_TRAPS_TRACKED_' + year + '.pkl'
    
    # overwrite any other export path, yearly files always go here
    pkl_exportpath = 'export_pkl/' + vel_product_short + '/20XX/'
    
    # create the object
    TRAPS_data = TRAPSdata(vel_product_short, vel_product_long, pd_TRAPS_cdf)
    
    # save the object as .pkl file
    save_object(TRAPS_data, pkl_exportpath + pkl_exportname)


    stop_timer = time.perf_counter()
    print('saved ' + pkl_exportname + f' in: {stop_timer - start_timer:0.1f} seconds')
    
    
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
