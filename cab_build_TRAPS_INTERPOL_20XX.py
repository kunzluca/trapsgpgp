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

# Building the TRAPS data object
# ==
#
# Build the overall TRAPS dataframe for every individual year, objectify and save it.  
# Also save a reduced version of the TRAPS dataframe which only contains cores and raw/interpolated curves.  
# These reduced TRAPS dataframes will be easier to load in later analysis.

# +
import os
import sys
import numpy as np
from scipy.interpolate import interp1d
import pandas as pd
import time
import datetime
import os
import pickle

from IPython.display import display, Audio
# import jupyter notebook files like regular python modules
import import_ipynb
from aa_define_classes import interpol_along_curve, TRAPSdata
# -

# measure the computation time for the entire script
start_script_timer = time.perf_counter()


# # Preliminary

# ## Pickle object saving function

def save_object(obj, filename):
    with open(filename, 'wb') as outp:  # overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)


# + [markdown] tags=[]
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
xTC_csvpath = 'export_csv_TRAPS/' + vel_product_short + '/' + year + '/pd_xTC_' + year + '_df.csv'
yTC_csvpath = 'export_csv_TRAPS/' + vel_product_short + '/' + year + '/pd_yTC_' + year + '_df.csv'
pxt_csvpath = 'export_csv_TRAPS/' + vel_product_short + '/' + year + '/pd_pxt_' + year + '_df.csv'
pyt_csvpath = 'export_csv_TRAPS/' + vel_product_short + '/' + year + '/pd_pyt_' + year + '_df.csv'
s1TC_csvpath = 'export_csv_TRAPS/' + vel_product_short + '/' + year + '/pd_s1TC_' + year + '_df.csv'

pkl_exportpath = 'export_pkl/' + vel_product_short + '/20XX/'

# # Import DataFrames
#
# Read the data from the respective csv files and put it into a pandas DataFrame.

start_timer = time.perf_counter()

# +
# xTC: x-component trap cores, yTC: y-component trap cores - vector
pd_xTC_df = pd.read_csv(xTC_csvpath)
pd_yTC_df = pd.read_csv(yTC_csvpath)

# pxt: x-coordinates of TRAPs, pyt: y-coordinates of TRAPs - size: [#TRAPs, #points along a TRAP]
# coordinates with Nan indicate regions of the tensor lines that do not satisfy the desired attraction properties
pd_pxt_df = pd.read_csv(pxt_csvpath)
pd_pyt_df = pd.read_csv(pyt_csvpath)

# s1TC: attraction rate at trap cores - vector
pd_s1TC_df = pd.read_csv(s1TC_csvpath)
# -

stop_timer = time.perf_counter()
print(f'task time: {stop_timer - start_timer:0.4f} seconds')

# ## Assertions
#
# Check for integrity of the different files.

# +
# assert that coordinate arrays are of same shape
assert pd_xTC_df.shape==pd_yTC_df.shape, 'TRAP cores: different number of x- and y-coordinates'
assert pd_pxt_df.shape==pd_pyt_df.shape, 'TRAP curves: different number of x- and y-coordinates'

# assert that number of TRAP cores equals number of TRAP curves and number of attraction rates
assert pd_xTC_df.shape[0]==pd_pxt_df.shape[0], 'mismatch number of TRAP cores and curves'
assert pd_xTC_df.shape==pd_s1TC_df.shape, 'mismatch number of TRAP cores and number of attraction rates'

# assert that data is given for exactly the same timestamps
assert pd_xTC_df.time.to_list()==pd_yTC_df.time.to_list()==pd_pxt_df.time.to_list(), 'mismatching timestamps'
assert pd_pxt_df.time.to_list()==pd_pyt_df.time.to_list()==pd_s1TC_df.time.to_list(), 'mismatching timestamps'

# define one array for all unique timestamps, call them timestrings since these are no datetime objects yet
#timestrings = np.unique(pd_xTC_df.time)
# -

# print check
pd_xTC_df
#pd_yTC_df
#pd_pxt_df
#pd_pyt_df.shape
#pd_s1TC_df
#timestrings

# + [markdown] tags=[]
# # TRAPS DataFrame
# Put cores and tensorlines into one DataFrame.  
# It's faster to loop through arrays than through an entire DataFrame, thus get the arrays first, assign them to a DataFrame column and keep them for later loops.
# -

start_timer = time.perf_counter()

pd_TRAPS_df = pd_xTC_df.rename(columns={'0': 'core_lon'}) # returns a copy
pd_TRAPS_df['core_lat'] = pd_yTC_df['0'].copy()
pd_TRAPS_df['core_attraction'] = pd_s1TC_df['0'].copy()

# +
# aggregate lons & lats of a TRAP curve into one array each
# we don't want TRAP_ID or time to show up in the aggregated lists, thus drop these columns first
# raw_ for the original TRAP curve, more explaination below
raw_curve_lons = pd_pxt_df.drop(columns=['TRAP_ID', 'time']).agg(func=np.array, axis=1).to_numpy() # this is an array of arrays
raw_curve_lats = pd_pyt_df.drop(columns=['TRAP_ID', 'time']).agg(func=np.array, axis=1).to_numpy() # this is an array of arrays

pd_TRAPS_df['raw_curve_lons'] = raw_curve_lons
pd_TRAPS_df['raw_curve_lats'] = raw_curve_lats

# + [markdown] tags=[]
# ## Timestrings to timestamps
#
# Convert the timestrings to datetime objects.
# -

pd_TRAPS_df['time'] = pd.to_datetime(pd_TRAPS_df.time, format='%Y%m%d%H%M')

stop_timer = time.perf_counter()
print(f'task time: {stop_timer - start_timer:0.4f} seconds')

# print check
pd_TRAPS_df#.time[0]
#pd_TRAPS_df.raw_curve_lons[927]
#pd_TRAPS_df.raw_curve_lats[927]


# cleanup and save memory
del pd_xTC_df, pd_yTC_df, pd_pxt_df, pd_pyt_df, pd_s1TC_df
del raw_curve_lons, raw_curve_lats

# # Remove non-attracting TRAPs
#
# For some reason, there are a few TRAP cores that are not attracting.  
# This error must stem from the MATLAB computation and needs to be corrected here.  
# Otherwise, the attracting/neutral TRAP corres will mess up with the following analysis.  
# Separate these attracting/neutral TRAPs from the DataFrame and save them for review if needed.  
# Move on with the clean DataFrame.

# +
pd_ERRORTRAPS_df = pd_TRAPS_df[pd_TRAPS_df.core_attraction >= 0].copy()
pd_TRAPS_df = pd_TRAPS_df[pd_TRAPS_df.core_attraction < 0].copy()

# reset the index otherwise it would have jumps and throw errors later
pd_TRAPS_df.reset_index(drop=True, inplace=True)
# -

# print check
pd_ERRORTRAPS_df
#pd_TRAPS_df

# +
# reload the raw arrays since the removal of non-attracting TRAPs has reset the DataFrame indices
# use these arrays later for loops and calculations
core_lon = pd_TRAPS_df.core_lon.to_numpy()
core_lat = pd_TRAPS_df.core_lat.to_numpy()

raw_curve_lons = pd_TRAPS_df.raw_curve_lons.to_numpy()
raw_curve_lats = pd_TRAPS_df.raw_curve_lats.to_numpy()

# + [markdown] tags=[]
# # Truncate TRAP curves to continuous ones
#
# Arrays for TRAP curves contain NaN values due to insufficient attraction rate at respective points. This may occur anywhere along the TRAP and thus cause discontinuous TRAPs. We want continuous TRAPs with no gaps. Truncating the TRAP curves to where the first discontinuities occur both on the left and right side of the core position will give us the inner continuous version of a TRAP curve.
#
# As a next step one could interpolate to equal distances between curve points and by this remove further discontinuities there, this is done at the end of the script.
#
# To truncate, we need to start at the array position/curve point closest to the TRAP core and then move both to left and right, checking the array elements for NaN values and slice the array there.
#
# The array index of the curve point closest to the TRAP core can be calculated for arbitrary TRAP core positions along the array using the lower code in comments. First applications revealed that this is actually always the same index for any TRAP and can be infered from the way TRAPs were constructed with lower and upper branches in the MATLAB code:
#
# Upper and lower branch each contain NumPointsOnCurve/2 points. They are connected by overlaping in one point, which is the position of the TRAP core. Thus the curve consists of NumPointsOnCurve-1 points. And the TRAP core is at point number NumPointsOnCurve/2. In terms of a numpy array, this relates to the core at index NumPointsOnCurve/2-1:
#
# # + NumPointsOnCurve-1 = raw_curve_lons.size
# # + core_index = (raw_curve_lons.size+1)/2 - 1
#
#
#

# + tags=[]
# gives the array index of the point closest to the TRAP core
# for the unlikely case of two points being equally close to the TRAP core, 
# it will return the index of the point that occurs first in the array
# which is okay, since we only need some point to start
# for a given TRAP, e.g. 
#ix = 540
#core_index = np.where(pd_TRAPS_df.loc[ix].curve_core_distances==np.nanmin(pd_TRAPS_df.loc[ix].curve_core_distances))[0][0]

# +
# infer the original MATLAB parameter NumPointsOnCurve, call it num_points_on_curve
# but due to overlap of the branches, the actual number is actually one point less, call it actual_points_on_curve
# pd_pxt_df has same shape as pd_pyt_df as asserted previously
actual_points_on_curve = pd_TRAPS_df.loc[0].raw_curve_lons.size
num_points_on_curve = actual_points_on_curve + 1

core_index = int(num_points_on_curve/2 - 1)
core_index

# +
# use **0.5 instead of np.sqrt since the latter throws the error
# loop of ufunc does not support argument 0 of type numpy.ndarray which has no callable sqrt method
raw_curve_core_distances = ((raw_curve_lons-core_lon)**2 + (raw_curve_lats-core_lat)**2)**0.5

for ix in pd_TRAPS_df.index:
    # assert that for every curve the curve_core_distance at the core_index is always zero
    # this would also highlight nan values at the core position
    assert raw_curve_core_distances[ix][core_index]==0, 'non-zero curve-core distance at core index'

# assign to DataFrame
pd_TRAPS_df['raw_curve_core_distances'] = raw_curve_core_distances

# +
# the maximum index of the original, raw arrays
max_index = actual_points_on_curve - 1

# initialise the new arrays of the truncated curves
trunc_curve_lons = []
trunc_curve_lats = []

for ix in pd_TRAPS_df.index: # iterates through all TRAPs

    left_index = core_index # scan elements to the left of the 1D array
    right_index = core_index # scan elements to the right of the 1D array
    
    # in the following use curve_core_distances since this naturally bears nan if only one of both coordinates is nan, 
    # so this already works as a combined filter
    current_array = raw_curve_core_distances[ix]
    
    # as soon as we are at index 0 or encounter a nan value at the preceding index, we are at the left end of the TRAP
    # and slice everything before the index position
    while (left_index > 0 and ~np.isnan(current_array[left_index-1])): left_index -= 1

    # as soon as we are at max_index+1 or encounter a nan value at the current index, we are at the right end of the TRAP
    # and slice off everything beyond the index position, including the index position, considering array slicing notion
    while (right_index <= max_index and ~np.isnan(current_array[right_index])): right_index += 1

    # these indices now work on every curve coordinate array since they all have the same shape
    trunc_curve_lons.append(raw_curve_lons[ix][left_index:right_index])
    trunc_curve_lats.append(raw_curve_lats[ix][left_index:right_index])


# assert that the truncated lons/lats arrays contain no NaNs
assert np.all(~np.isnan(np.concatenate(trunc_curve_lons))), 'NaN values in truncated lons arrays'
assert np.all(~np.isnan(np.concatenate(trunc_curve_lats))), 'NaN values in truncated lats arrays'
    
# assign the truncated curves to the DataFrame
pd_TRAPS_df['trunc_curve_lons'] = trunc_curve_lons
pd_TRAPS_df['trunc_curve_lats'] = trunc_curve_lats

# -

# print check
#type(pd_TRAPS_df.loc[0].trunc_curve_lats)
#pd_TRAPS_df#.trunc_curve_lons
#pd_TRAPS_df.trunc_curve_lats
pd_TRAPS_df

# # Estimate truncation
#
# Find out, how many of the dataset's raw TRAP points were removed since they belonged to discontinuous parts of the TRAP curve, i.e. behind a nan-gap in the TRAP.  Nan points/points of insufficient attraction rate are not counted in.  
#
# By how much is the number of TRAP points reduced when looking only at truncated TRAPs?

# +
all_lons_raw = np.concatenate(raw_curve_lons) # contains NaNs
all_lons_raw = all_lons_raw[~np.isnan(all_lons_raw)] # remove NaNs

all_lons_trunc = np.concatenate(trunc_curve_lons) # contains no NaNs

num_points_raw = all_lons_raw.size
num_points_trunc = all_lons_trunc.size

print(num_points_raw, ' TRAP points before truncation')
print(num_points_trunc, ' TRAP points after truncation')
print()
print('number of TRAP points reduced by ', (1 - num_points_trunc/num_points_raw)*100, ' %')

# + [markdown] tags=[]
# # Interpolate to equal distances
#
# After truncation, TRAP curves bear no more NaN values but points on the truncated curve can still be unequally distributed, leading to TRAP points with variable distance between one another.  
#
# We want to interpolate these points to equidistant points along a TRAP curve such that TRAP curves show no more gaps, i.e. are finally continuous in both senses. This will allow for a correct counting of TRAP curve occurences in the histograms later.
#
# The interpolation can be performed using np.linspace() or np.arange() for the creation of the interpolation points. Both methods lead to different distributions of segment sizes, we choose to use the np.arange() results.
# -

start_timer = time.perf_counter()

# ## Interpolate using np.linspace()

# +
# initialise the new arrays of the interpolated curves
interpol_linspace_curve_lons = []
interpol_linspace_curve_lats = []
interpol_arange_curve_lons = []
interpol_arange_curve_lats = []


for ix in pd_TRAPS_df.index: # iterate through all TRAPs

    # get the points of the truncated version of the current TRAP curve
    xs=trunc_curve_lons[ix]
    ys=trunc_curve_lats[ix]
    
    # interpolate using np.linspace()
    interpolated_lons, interpolated_lats = interpol_along_curve(xs=xs, ys=ys, interpolation_mode='LINSPACE')            
    
    interpol_linspace_curve_lons.append(interpolated_lons)
    interpol_linspace_curve_lats.append(interpolated_lats)

    # interpolate using np.arange()
    interpolated_lons, interpolated_lats = interpol_along_curve(xs=xs, ys=ys, interpolation_mode='ARANGE')            
    
    interpol_arange_curve_lons.append(interpolated_lons)
    interpol_arange_curve_lats.append(interpolated_lats)

# assign the interpolated curves to the DataFrame
pd_TRAPS_df['interpol_linspace_curve_lons'] = interpol_linspace_curve_lons
pd_TRAPS_df['interpol_linspace_curve_lats'] = interpol_linspace_curve_lats
pd_TRAPS_df['interpol_arange_curve_lons'] = interpol_arange_curve_lons
pd_TRAPS_df['interpol_arange_curve_lats'] = interpol_arange_curve_lats
# -

stop_timer = time.perf_counter()
print(f'task time: {stop_timer - start_timer:0.4f} seconds')

# print check
pd_TRAPS_df

# # Export sub dataframes
#
# Extract sub dataframes from the overall pd_TRAPS_df and export these as pickle files.  
# We want a reduced version of the TRAPS dataframe which only contains cores and certain kind of curves since this will be easier to save and load in later analysis.

# ## Export raw TRAPS
#
#

# +
#pd_TRAPSRAW_df = pd_TRAPS_df[['TRAP_ID', 'time', 
#                              'core_lon', 'core_lat', 'core_attraction', 
#                              'raw_curve_lons', 'raw_curve_lats']].copy()

# +
# print check
#pd_TRAPSRAW_df

# +
# create the object
#TRAPS_data = TRAPSdata(vel_product_short, vel_product_long, pd_TRAPSRAW_df)

# +
# save the object as .pkl file
#start_timer = time.perf_counter()

#pkl_filename = vel_product_short + '_TRAPSRAW_' + year + '.pkl'

#save_object(TRAPS_data, pkl_exportpath + pkl_filename)

#stop_timer = time.perf_counter()
#print()
#print(f'task time: {stop_timer - start_timer:0.4f} seconds')

# +
# cleanup and save memory
#del TRAPS_data, pd_TRAPSRAW_df
# -

# ## Export trunc TRAPS
#
#

# +
#pd_TRAPSTRUNC_df = pd_TRAPS_df[['TRAP_ID', 'time', 
#                                'core_lon', 'core_lat', 'core_attraction', 
#                                'trunc_curve_lons', 'trunc_curve_lats']].copy()

# +
# print check
#pd_TRAPSTRUNC_df

# +
# create the object
#TRAPS_data = TRAPSdata(vel_product_short, vel_product_long, pd_TRAPSTRUNC_df)

# +
# save the object as .pkl file
#start_timer = time.perf_counter()

#pkl_filename = vel_product_short + '_TRAPSTRUNC_' + year + '.pkl'

#save_object(TRAPS_data, pkl_exportpath + pkl_filename)

#stop_timer = time.perf_counter()
#print()
#print(f'task time: {stop_timer - start_timer:0.4f} seconds')

# +
# cleanup and save memory
#del TRAPS_data, pd_TRAPSTRUNC_df
# -

# ## Export TRAPS interpolated with np.arange()

pd_TRAPSINTERPOL_df = pd_TRAPS_df[['TRAP_ID', 'time', 
                                   'core_lon', 'core_lat', 'core_attraction', 
                                   'interpol_arange_curve_lons', 'interpol_arange_curve_lats']].copy()

# print check
pd_TRAPSINTERPOL_df

# create the object
TRAPS_data = TRAPSdata(vel_product_short, vel_product_long, pd_TRAPSINTERPOL_df)

# +
# save the object as .pkl file
start_timer = time.perf_counter()

# TRAPSINTERPOL for interpolated TRAPS
pkl_exportname = vel_product_short + '_TRAPS_INTERPOL_' + year + '.pkl'

save_object(TRAPS_data, pkl_exportpath + pkl_exportname)

stop_timer = time.perf_counter()
print('saved ' + pkl_exportname + f' in: {stop_timer - start_timer:0.4f} seconds')
# -

# cleanup and save memory
del TRAPS_data, pd_TRAPSINTERPOL_df

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
