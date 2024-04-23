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

# Building the final yearly TRAPS dataframes
# ==
#
# Load the TRAPSINTERPOL pickle objects for a given product and year and interpolate s1 values to the equidistant curve points. Then export the new dataframe as yearly TRAPS_UNTRACKED dataframe.

# +
import os
import sys
import numpy as np
from scipy.interpolate import interp2d
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
    year_ID = 0
    notebook_run = True
    # save_fig = True
    save_fig = False

    
# if script is running as python script
else:
    # read in product from bash
    vel_product_ID = int(sys.argv[1])
    # read in year from bash
    year_ID = int(sys.argv[2])
    notebook_run = False
    save_fig = True


vel_product_short = ['ENSRYS_24HM', 'MULTIOBS_24HI', 'MULTIOBS_24HM', 'SEALEVEL_24HI'][vel_product_ID]

vel_product_long = ['CMEMS GLOBAL_REANALYSIS_PHY_001_031 ENSEMBLE MEAN (1/4°, 24HM)', 
                    'CMEMS MULTIOBS_GLO_PHY_REP_015_004 (1/4°, 24HI)', 
                    'CMEMS MULTIOBS_GLO_PHY_REP_015_004 (1/4°, 24HM)', 
                    'SEALEVEL_GLO_PHY_L4_NRT_OBSERVATIONS_008_046'][vel_product_ID]

years = ['2000', '2001', '2002', '2003', '2004', '2005', '2006', '2007', '2008', '2009', 
         '2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019']

year = years[year_ID]

# -

# define paths to files
pkl_importpath = 'export_pkl/' + vel_product_short + '/20XX/'
pkl_exportpath = 'export_pkl/' + vel_product_short + '/20XX/'

# # Import DataFrames
#
# Load the TRAPSINTERPOL_ARANGE pkl file.  

start_timer = time.perf_counter()

# +
pkl_importname = vel_product_short + '_TRAPS_INTERPOL_' + year + '.pkl'
    
# load the pickle object for the current year
with open(pkl_importpath + pkl_importname, 'rb') as inp:
    # VD for velocity domain
    pd_TRAPS_df = pickle.load(inp).pd_TRAPS_df.copy()

pd_TRAPS_df.rename(columns={'interpol_arange_curve_lons': 'curve_lons', 
                            'interpol_arange_curve_lats': 'curve_lats'}, inplace=True)

# to be on the safe side, always reset the index after loading
pd_TRAPS_df.reset_index(drop=True, inplace=True)
# -

stop_timer = time.perf_counter()
if notebook_run: print(f'task time: {stop_timer - start_timer:0.4f} seconds')

# print check
pd_TRAPS_df

# + [markdown] tags=[]
# # Interpolate s1-values to curve points
#
# We want to know the attraction rate not only at the core but also at every point along the curve.  
# For this, we have to load the full s1 field at a given timestamp and interpolate s1-values to the curve points.  
# Since multiple TRAPS exist at one timestamp, the current s1 field will be interpolated to the points of multiple curves. 
#
# We do this before the filtering for the histogram domain since afterwards one would have to handle various complex NaN cases. It's worth the extra computational cost of interpolating to curves which lie outside the histogram domain and will be removed later.
# -

# ## Get all timestamps

# np.unique() sorts the output, DataFrame.unique() keeps the order of appearance
timestamps = pd.to_datetime(pd_TRAPS_df.time.unique()) # this gives a Timestamp object

# print check
timestamps
timestamps[0].strftime('%Y%m%d%H%M')
#timestrings

# ## Load grid coordinates
#
# Load the longitude and latitude data defining the grid of the current object.  
# Since for one model, the grid structure is the same for all years, it's sufficient to load the 2000 one.

# paths
grid_path = 'export_csv_velocities/' + vel_product_short + '/2000/'
s1_path = 'export_matlab/s1/' + vel_product_short + '/' + year + '/'

# +
pd_LONS_df = pd.read_csv(grid_path + vel_product_short + '_LONS.csv', header=None)
pd_LATS_df = pd.read_csv(grid_path + vel_product_short + '_LATS.csv', header=None)

# get the array of LON and LAT values
lons = pd_LONS_df.iloc[0,:].to_numpy() # copy from first row
lats = pd_LATS_df.iloc[:,0].to_numpy() # copy from first column

# assert that the array values are ascending
assert np.all(lons==np.sort(lons)), 'unsorted lons after loading'
assert np.all(lats==np.sort(lats)), 'unsorted lats after loading'

# +
# print check
#pd_LONS_df
#pd_LATS_df
#lons
#lats
# -

# ## Interpolate the s1 field

start_timer = time.perf_counter()

# append a new column to the dataframe, specifying the attraction rates along all curves
pd_TRAPS_df['curve_attractions'] = np.nan

#for timestamp in timestamps[:1]:
for timestamp in timestamps:
    
    #======================================
    # SELECT TRAPS OF THE CURRENT TIMESTAMP
    #======================================

    # boolean array
    time_filter = (pd_TRAPS_df.time==timestamp).to_numpy()
        
    # coordinates and attraction rates for the curves of the current timestamp, ccurve for current curve
    ccurve_lons = pd_TRAPS_df[time_filter].curve_lons.to_numpy()
    ccurve_lats = pd_TRAPS_df[time_filter].curve_lats.to_numpy()
    ccurve_attractions = []
    
    # assert same number of TRAPS
    assert ccurve_lons.size==ccurve_lats.size, 'different number of curves in lon and lat columns'
    
    
    #===========================================
    # LOAD THE S1 FIELD OF THE CURRENT TIMESTAMP
    #===========================================
    
    current_s1_file = vel_product_short + '_s1_' + timestamp.strftime('%Y%m%d%H%M') + '.csv'
    
    current_s1_field = pd.read_csv(s1_path + current_s1_file, header=None).to_numpy()
    
    # assert that the s1 field has the right shape
    assert current_s1_field.shape[0]==lats.size, 's1 field with wrong size in y'
    assert current_s1_field.shape[1]==lons.size, 's1 field with wrong size in x'
    
    #==========================================
    # DEFINE THE CURRENT INTERPOLATION FUNCTION
    #==========================================

    # interpolation function at the current timestamp-specific s1 field
    # apply cubic interpolation since the original MATLAB code uses cubic spline interpolation
    # linear interpolation would cause some s1 values below 0.3 * core_attraction
    interpol_func = interp2d(x=lons, y=lats, z=current_s1_field, kind='cubic', bounds_error=True)
    
    #=============================================
    # INTERPOLATE S1 VALUES TO PRESENT TRAP CURVES
    #=============================================

    for curve_index in range(ccurve_lons.size): # itterate through all TRAP curves at the current timestamp
                            
        # we have multiple points with two coordinates for which we want to interpolate s1 values
        # if we pass the two coordinate arrays straight to interp2d, it will give a 2D array with s1 values 
        # at all coordinate combinations, not just at coordinate pairs (these would lie on the diagonal)
        # thus we apply the function individually for every coordinate tuple

        # interpolate the s1 values to every individual point of the current TRAP curve
        s1_values = np.array([float(interpol_func(lon, lat)) for lon, lat 
                              in zip(ccurve_lons[curve_index], ccurve_lats[curve_index])])


        # assert same number of attraction values and curve points
        assert s1_values.size==ccurve_lons[curve_index].size, 'numbers of s1 values and curvepoints mismatch'
                        
        ccurve_attractions.append(s1_values) # list of ragged arrays, one per TRAP curve

    # Creating an ndarray from ragged nested sequences must be specified with 'dtype=object'
    ccurve_attractions = np.array(ccurve_attractions, dtype=object)
        
    # assert same number of TRAP curves in attraction and coordinate arrays
    assert ccurve_attractions.size==ccurve_lons.size, 'different number of TRAP curves in attraction and coordinate columns'
        
    # load the curve attractions of the current timestamp into the overall dataframes
    pd_TRAPS_df.loc[time_filter, 'curve_attractions'] = ccurve_attractions
        

stop_timer = time.perf_counter()
if notebook_run: print(f'task time: {stop_timer - start_timer:0.4f} seconds')

# print check
#current_s1_field.shape
#s1_values
#ccurve_attractions
#ccurve_lons
pd_TRAPS_df

# + [markdown] tags=[]
# # Review s1 values
#
# Check again if s1 is negative at a TRAP core and if the related curvepoints bear an attraction rate of at least 30% of the attraction at the core.  
# One would expect all curvepoints to fulfill this criterion since the coarse 1/12° curves we look at were interpolated upon the TRUNCATED 1/40° CURVES which all fulfilled this assertion.  
# However, there might occur small numerical errors, this is why we round the fraction curvepoint_attraction/core_attraction to two decimals.  
# If finally the critetion should not be met, we'd have to compute something new.
#
# This is also a good test if we have set the right dimensions and fields in the s1 interpolation process.  
# A systematically wrong interpolation setup would not make it through the full dataset without violating the below assertion once. Note that already a linear instead of a cubic interpolation can't make it through the following loop.

# +
core_attraction = pd_TRAPS_df.core_attraction.to_numpy()
curve_attractions = pd_TRAPS_df.curve_attractions.to_numpy()

for ix in range(core_attraction.size): # iterate through all TRAPS
                
    # assert that cores are still attracting
    assert core_attraction[ix] < 0, 'found a repelling core'    
    
    for curvepoint_attraction in curve_attractions[ix]:
        # assert that curvepoints bear at least 30% of attraction at the core
        assert np.around(curvepoint_attraction/core_attraction[ix], decimals=2) >= 0.3, 'insufficient s1 value at curvepoint'
    
# -

# print check
pd_TRAPS_df

# # Export pickle objects

# create the object
TRAPS_data = TRAPSdata(vel_product_short, vel_product_long, pd_TRAPS_df)

# +
# save the object as .pkl file
start_timer = time.perf_counter()

pkl_exportname = vel_product_short + '_TRAPS_UNTRACKED_' + year + '.pkl'

save_object(TRAPS_data, pkl_exportpath + pkl_exportname)

stop_timer = time.perf_counter()
print('saved ' + pkl_exportname + f' in: {stop_timer - start_timer:0.4f} seconds')

# + [markdown] tags=[]
# # End sound
# -

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
