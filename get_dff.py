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

from tqdm import tqdm

from img_utils import *

save_location = '/mnt/md0/'
data_type = 'BRUKER'
date = '01072025'
file_num = 1
stim_file= 1
optical_zoom = 2
dur_resp  = 2.5 # seconds

# Uncomment as each is verified in behavior
do_neuropil = True
do_cascade = True
is_2p_opto = False
# is_voltage = False
# extract_hotspot = False

# 30 frames added to movie start at .h5 conversion.
#   If DeepInterpolation is not run, these must be accounted for.
#   DeepInterpolation removes these when run.
is_deep_interp = False

chnk = int(1e3) # Number of frames to process at a time

# Cascade filenames here ~~~~~~~~~~~~~~~~~~~~~~
cascade_dir = '/home/schollab-dion/Documents/Cascade-1.0.0'
cascade_model_dir = os.path.join(cascade_dir, 'Pretrained_models')
sys.path.append(cascade_dir)
from cascade2p.cascade import predict as cascade_predict

folder_list = get_target_folders_v2(save_location+data_type+'/', date,file_num, 'TSeries')
# Improvement would be handling more than one directory at a time,
#       but our workflow almost always goes session by session
#       for data checking and validation reasons.
target_folder = folder_list[0]
os.chdir(target_folder)
print(f'Starting CE creation for {target_folder}')

## ROI file handling
try:
    roi_list = roifile.roiread('RoiSet.zip')
    #code.interact(local=dict(globals(), **locals())) 
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

# Because of immutability of NumPy arrays...
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
        vrec = np.genfromtxt(voltage_files[0], delimiter=',', skip_header=1) # skip one row


    elif 'SCANIMAGE' in data_type:
        print('scanimage two-photon timeframes must be implemented')
        sys.exit(1)

    print(f'Num frametriggers detected: {frame_triggers.shape[0]}')
    print('Getting stimulus times and syncing with two-photon frame times...')
    stim_triggers = medfilt(vrec[:,2],101)
    stim_triggers[stim_triggers < 0] = 0
    stim_triggers = np.diff(stim_triggers)

    stim_on, _ = find_peaks(stim_triggers, distance=1e3, height=(max(stim_triggers) - max(stim_triggers)*0.9))
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
    else:
        print('Need to test this with a 2p opto file')
        sys.exit(1)

    assert stim_on.shape[0] == stim_id.shape[0],'Mismatch between stimulus onset times and number of stimulus IDs.'

    stim_on_2p_frame = np.zeros(stim_id.shape[0])
    #for s in range(stim_id.shape[0]):
    #    stim = stim_id[s]
    #    frame_loc = min(abs(stim - frame_triggers))
    #    stim_on_2p_frame[s] = frame_loc

    ## Target Stim 2p Frame synchronization
    #if is_2p_opto:
else:
    print('No stimulus triggers recorded for this dataset.')

#if do_neuropil:
#    neuropil_subtraction()


if do_cascade:
    cascade_results = cascade_predict('Universal_30Hz_smoothing100ms', dff.T, model_folder=cascade_model_dir)
#gen_stim_cyc()

#dendrite_subtraction()

## Extract some basic responses from cyc or cyc_res

## Saving - 
code.interact(local=dict(globals(), **locals())) 


### Confirmations that signal has been acquired correctly.
### - masks are in right locations
### - raw signal looks correct
### - dff/filter_baseline_dF_comp results are correct
### - neuropil signal looks correct

# Want to signal correct execution, but does it matter?
sys.exit(0)