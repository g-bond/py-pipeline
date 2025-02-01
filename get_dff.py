# getDFF
# processing script for extracting traces from calcium movies
# presupposes registration, denoising optional
import os
import sys
import code
from glob import glob

import tifffile
import roifile

import math
import h5py
import numpy as np
from scipy.signal import medfilt, find_peaks

import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt

from tqdm import tqdm

from img_utils import *


save_location = '/mnt/md0/'
data_type = 'BRUKER'
date = '11042024'
file_num = 39
stim_file= -1
optical_zoom = 2
dur_resp  = 2.5 # seconds

# Uncomment as each is verified in behavior
do_neuropil = True
do_cascade = True
cascade_model = ''
is_2p_opto = False
# is_voltage = False
# extract_hotspot = False

# 30 frames added to movie start at .h5 conversion.
#   If DeepInterpolation is not run, these must be accounted for.
#   DeepInterpolation removes these when run.
is_deep_interp = False

frame_period = 0.033
chnk = int(1e3) # Number of frames to process at a time

# Cascade filenames here ~~~~~~~~~~~~~~~~~~~~~~
cascade_dir = '/home/schollab-dion/Documents/Cascade-1.0.0'
cascade_model_dir = os.path.join(cascade_dir, 'Pretrained_models')
assert os.path.isdir(os.path.join(cascade_model_dir, cascade_model)),  \
    f'Specified cascade model {cascade_model} not found.' \
    'Install, or check if a valid pretrained cascade model was specified.'
sys.path.append(cascade_dir)
from cascade2p.cascade import predict as cascade_predict

folder_list = get_target_folders_v2(save_location+data_type+'/', date,file_num, 'TSeries')

target_folder = folder_list[0]
os.chdir(target_folder)
print(f'Starting CE creation for {target_folder}')

## ROI file handling
try:
    roi_list = roifile.roiread('RoiSet.zip')
except FileNotFoundError:
    print(f'Halting. No ROI file for {target_folder}')
    sys.exit(1) # kill
else:
    num_cells = len(roi_list)
    located_dend_roi = any(roi.roitype == 5 for roi in roi_list)

## Grabbing .h5 processed calcium movie handling - 
try:
    if is_deep_interp:
        h = h5py.File('inference_results.h5', 'r')
    else:
        h = h5py.File('registered.h5', 'r')
except FileNotFoundError:
    print(f'Halting. Processed calcium movie file not found. \n Check destination and/or "is_deep_interp" flag.')
    sys.exit(1) # kill

dat_name = [key for key in h.keys()][0]
num_frames, size_x, size_y = h[dat_name].shape

## Grabbing cell masks -
x = np.linspace(0, size_x-1, size_x)
y = np.linspace(0, size_y-1, size_y)
x, y = np.meshgrid(x, y)
mask2d = np.zeros((num_cells, size_x, size_y))
neuropil_mask = np.zeros((size_x,size_y))

for cc in tqdm(range(num_cells), desc="getting masks", ncols=75):
    nm_coord = roi_list[cc].coordinates()
    if roi_list[cc].roitype == 5:       
        mask2d[cc,:,:] = gen_polyline_roi(nm_coord=nm_coord, d_width=roi_list[cc].stroke_width)
    else:
        mask2d[cc,:,:] = in_polygon(x, y, nm_coord[:,0], nm_coord[:,1])
    neuropil_mask += mask2d[cc,:,:]


## Using cell masks to extract raw cell traces -
raw_cell_traces = np.zeros((num_frames, num_cells))
raw_neuropil    = np.zeros((num_frames))

for f_i in tqdm(range(math.ceil(num_frames / chnk)), desc="Extracting...", ncols=75):
    start = int(f_i * chnk)
    stop = min(int((f_i + 1) * chnk), num_frames)
    imgstack = h[dat_name][start:stop, :, :]
    
    for cc in range(num_cells):
        nz = np.nonzero(mask2d[cc,:,:])
        raw_cell_traces[start:stop, cc] = np.mean(imgstack[:,nz[0], nz[1]], axis=1)
    
    # Need to get nz_neuropil as well
    if do_neuropil:
        nz = np.nonzero(neuropil_mask)
        raw_neuropil[start:stop] = np.mean(imgstack[:,nz[0], nz[1]], axis=1)


# Because of immutability of NumPy arrays, have to perform a deep copy and temporarily duplicate in memory.
if not is_deep_interp:
    raw_cell_traces_ = raw_cell_traces[30:, :]
    raw_neuropil_ = raw_neuropil[30:]
    raw_cell_traces = raw_cell_traces_
    raw_neuropil = raw_neuropil_
    del raw_neuropil_
    del raw_cell_traces_
    num_frames = num_frames - 30

## Getting dF/F from raw traces
dff = np.zeros((num_frames, num_cells))
for cc in tqdm(range(num_cells), desc="Getting dF/F per cell...", ncols=75):
    dff[:,cc] = filter_baseline_dF_comp(raw_cell_traces[:,cc], 99*4+1)

if do_neuropil:
    dff_neuropil = filter_baseline_dF_comp(raw_neuropil, 99*4+1)

if stim_file > -1:
    print('Grabbing two-photon frametimes')
    if 'BRUKER' in data_type:
        xml_dest = os.path.join(target_folder, os.path.basename(target_folder))+'.xml'
        frame_triggers = read_xml_file(xml_dest)
        frame_triggers = frame_triggers * 1e4 # Convert seconds to 10kHz sampling
        if frame_triggers[0] > 340:
            print('First 2p frame was dropped!')
        
        frame_triggers = replace_missing_frame_triggers(frame_triggers)

        psychopy_loc = data_type + '_PSYCHOPY'
        voltage_files = glob('*.csv') # already in datadir
        assert len(voltage_files) == 1, f'Unique, singular voltage file not found. Check {target_folder}'
        vrec = genfromtxt_with_progress(voltage_files[0], delimiter=',', skip_header=1) # skip one row. uses np.genfromtxt


    elif 'SCANIMAGE' in data_type:
        print('scanimage two-photon timeframes must be implemented')
        sys.exit(1)

    print(f'Num frametriggers detected: {frame_triggers.shape[1]}')
    print('Getting stimulus times and syncing with two-photon frame times...')
    stim_triggers = medfilt(vrec[:,2],101)
    stim_triggers[stim_triggers < 0] = 0
    stim_triggers = np.diff(stim_triggers)

    stim_on, _  = find_peaks(stim_triggers, distance=1e3, height=(max(stim_triggers) - max(stim_triggers)*0.9))
    stim_off, _ = find_peaks(stim_triggers, distance=1e3, height=(max(stim_triggers) - max(stim_triggers)*0.9))
    
    if is_2p_opto:
        print('Processing 2pOpto triggers...')
        stim_triggers = medfilt(vrec[:,2],51)
        stim_triggers[stim_triggers < 0] = 0
        photostim_triggers = find_peaks(stim_triggers, distance=1e4, height=(max(stim_triggers) - max(stim_triggers)*0.9))
    
    psychopy_file_str = os.path.join(psychopy_loc,'-'.join([date[4:], date[0:2], date[2:4]]))
    os.chdir(save_location+psychopy_file_str)
    
    psychopy_file = np.genfromtxt('T'+'{:03d}'.format(stim_file)+'.txt')
    
    if not is_2p_opto:
        stim_id = psychopy_file[:,0]
        unique_stims = np.unique(stim_id)
        stim_properties = psychopy_file[:,1:]
    else:
        print('Need to test this with a 2p opto file')
        sys.exit(1)

    assert len(stim_on) == len(stim_id),'Mismatch between stimulus onset times and number of stimulus IDs.'

    
    stim_on_2p_frame = np.zeros(len(stim_id))

    for s in range(len(stim_id)):
        stim_on_2p_frame[s] = np.argmin(abs(stim_on[s] - frame_triggers))

    ## Target Stim 2p Frame synchronization
    #if is_2p_opto:
else:
    print('No stimulus triggers recorded for this dataset.')


# This isn't matching the same regression slopes as MATLAB.
#   Will need to make sure the algorithm matches the same
#   weighting choices that MATLAB does.
#if do_neuropil:
#    neuropil_subtraction()

#if do_cascade:
    #cascade_results = cascade_predict('Universal_30Hz_smoothing100ms', dff.T, model_folder=cascade_model_dir)
#    spike_inference = cascade_predict('Global_EXC_30Hz_smoothing50ms_causalkernel', dff.T, model_folder=cascade_model_dir).T



###  gen_stim_cyc([dur_resp, 0, 0], ) ---------------
# SO much global information is used for calculating stimulus cycles...
# Do it here first, then figure out how you're going to pass that info
#   via reading off the .h5 (may have concurrent read access problems)
if stim_file > 0:
    pre = 0
    slag = 0

    n_trials = int(np.floor(len(stim_id) / len(unique_stims)))
    stim_dur = int(np.round(dur_resp / frame_period))
    cyc_pre  = int(np.round(pre     / frame_period))
    cyc_slag = int(np.round(slag   / frame_period))

    cyc         = np.zeros((num_cells, len(unique_stims), n_trials, stim_dur + pre))
    cyc_spk_inf = np.zeros((num_cells, len(unique_stims), n_trials, stim_dur + pre))

    for cc in range(num_cells):
        trial_list = np.zeros(len(unique_stims))

        for ii in range(len(stim_on_2p_frame)):
            # The original MATLAB for this one is confusing to me
            #   if sum(trialList==ntrials)~=uniqStims
            if True:
                tt = np.arange(stim_on_2p_frame[ii]- cyc_pre + 1 + cyc_slag, stim_on_2p_frame[ii] + stim_dur + cyc_slag + 1)
                tt = tt.astype('int')
                ind = np.argwhere(unique_stims == stim_id[ii])[0][0] # <- horrendous notation
                trial_list[ind] += 1

                if tt[-1] < dff.shape[0]: # Don't want to continue beyond the end of our timeseries
                    f = dff[tt,cc]
                    # Isn't the same check as finding if the ce struct has this field.
                    if do_cascade:
                        s = spike_inference[tt,cc]
                else:
                    f = np.empty((76,))
                    s = np.empty((76,))
                    f[:] = np.nan
                    s[:] = np.nan

                cyc[cc, ind, int(trial_list[ind]), :] = f
                if do_cascade:
                    cyc_spk_inf[cc, ind, int(trial_list[ind]), :] = s

### End genstimcyc -------------------------


### Dendrite Subtraction -------------------
dend_sub_flag = 1

is_dendrite = [roi.roitype == 5 for roi in roi_list]
is_dendrite = np.where(is_dendrite)[0]

dend_count = 0

for cc in range(num_cells):
    
    if roi_list[cc].roitype == 5:
        dend_count += 1
    
    elif roi_list[cc].roitype == 7:
        sp_dff = dff[:,cc]
        sp_dff = sp_dff[0:int(np.round(len(sp_dff)*0.9))] # Kick out the last 10%
        sp_dff[np.isinf(sp_dff)] = 0
        sp_dff_sub = sp_dff[sp_dff < np.nanmedian(sp_dff) + np.abs(np.min(sp_dff))]
        
        noise_m  = np.nanmedian(sp_dff_sub)
        noise_sd = np.nanstd(sp_dff_sub)

        if dend_sub_flag == 1:
            disp('find robustfit py equiv')
            # slope = robustfit( dff[:,is_dendrite[dend_count]], dff[:,cc])
        elif dend_sub_flag == 1 and cyc:
            sp_cyc = cyc[cc,:,:,:]
            sp_cyc[np.isnan[sp_cyc]] = 0
            # check indexing of cyc material here
            # slope = robustfit (cyc[is_dendrite[dend_count],:,:,:], sp_cyc))
        else:
            print('Not implemented')
            # slope = robustfit (dff[:,is_dendrite[dend_count]])
        #r = dff[cc,:] - slope 
        r_sp = dff[:,cc]
        r_dn = dff[:,is_dendrite[dend_count]]
        r_sp = r_sp - np.multiply(slope[2], noise_sd)
        r_sp[r_sp < (-1*noise_sd)] = (-1 * noise_sd)
        r_sp[np.isinf(r_sp)] = 0
        r_dn[np.isinf(r_dn)] = 0
        
        # dffRes structure
        r_sp[r_sp <= 0] = np.nan
        r_dn[r_dn <= 0] = np.nan

        # r = corrcoef(r_sp, r_dn, 'rows', 'pairwise')
    else:
        #cycRes
        #slope
        #corr
        print('Not implemented.')

# End Dendrite Subtraction

## Extract basic responses from cyc or cyc_res
if stim_file > -1 and np.floor(len(stim_id) / len(unique_stims)) > 2:
    for cc in range(num_cells):
        # This is not the same way as saying something's a spine.
        if roi_list[cc].roitype == 7 and dend_count > 0:
            resp, resps, resp_err = compute_peak_resp(cyc)
        else:
            resp, resps, resp_err = compute_peak_resp(cyc_res)

        # W/in loop, collect resp, resps, resp_err into .h5


## Saving - 
code.interact(local=dict(globals(), **locals())) 


sys.exit(0) # Signal correct exit