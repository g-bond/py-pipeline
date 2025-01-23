# getDFF
# processing script for extracting traces from calcium movies
# presupposes registration, denoising optional
import os
import code
from glob import glob

import tifffile
import roifile

import math
import h5py
import numpy as np

from tqdm import tqdm

from img_utils import in_polygon, filter_baseline_dF_comp, get_target_folders_v2, read_xml_file

save_location = '/mnt/md0/'
data_type = 'BRUKER'
date = '01072025'
file_num = 4
stim_file= 4
optical_zoom = 2
dur_resp  = 2.5 # seconds

# Uncomment as each is verified in behavior
# do_neuropil = False
# do_cascade = False
# is_2p_opto = False
# is_voltage = False
# extract_hotspot = False

# 30 frames added to movie start at .h5 conversion.
#   If DeepInterpolation is not run, these must be accounted for.
#   DeepInterpolation removes these when run.
is_deep_interp = False

chnk = int(1e4) # Number of frames to process at a time

# Cascade filenames here ~~~~~~~~~~~~~~~~~~~~~~


folder_list = get_target_folders_v2(save_location+data_type+'/', date,file_num, 'TSeries')
# Improvement would be handling more than one directory at a time,
#       but our workflow almost always goes session by session
#       for data checking and validation reasons.
target_folder = folder_list[0]
os.chdir(target_folder)

## ROI file handling
try:
    roi_list = roifile.roiread('RoiSet.zip')
except FileNotFoundError:
    print(f'No ROI file for {target_folder}')

#code.interact(local=dict(globals(), **locals())) 
num_cells = len(roi_list)
located_dend_roi = any(roi.roitype == 5 for roi in roi_list)

## Grabbing .h5 processed calcium movie handling - 
try:
    if is_deep_interp:
        h = h5py.File('inference_results.h5', 'r')
    else:
        h = h5py.File('registered.h5', 'r')
except FileNotFoundError:
    print(f'Processed file not found. \n Check destination or "is_deep_interp" flag.')

dat_name = [key for key in h.keys()][0]
num_frames, sizeX, sizeY = h[dat_name].shape


## Grabbing cell masks -
x = np.linspace(0, sizeX-1, sizeX)
y = np.linspace(0, sizeY-1, sizeY)
x, y = np.meshgrid(x, y)
mask2d = np.zeros((num_cells, sizeX, sizeY))

for cc in tqdm(range(num_cells), desc="getting masks", ncols=75):
    nmCoord = roi_list[cc].coordinates()
    mask2d[cc,:,:] = in_polygon(x, y, nmCoord[:,0], nmCoord[:,1])

## Using cell masks to extract raw cell traces -
raw_cell_traces = np.zeros((num_frames, num_cells))
dff = np.zeros((num_frames, num_cells))

for f_i in tqdm(range(math.ceil(num_frames / chnk)), desc="Extracting...", ncols=75):
    start = int(f_i * chnk)
    stop = min(int((f_i + 1) * chnk), num_frames)
    imgstack = h[dat_name][start:stop, :, :]

    for cc in range(num_cells):
        nz = np.nonzero(mask2d[cc,:,:])
        raw_cell_traces[start:stop, cc] = np.mean(imgstack[:,nz[0], nz[1]], axis=1)

print("Calculating dff for all cells")
for cc in range(num_cells):
    dff[:,cc] = filter_baseline_dF_comp(raw_cell_traces[:,cc], 99*4+1)
print("done")

if stim_file > -1:
    print('Grabbing two-photon frametimes')
    if 'BRUKER' in data_type:
        code.interact(local=dict(globals(), **locals()))
        xml_dest = os.path.join(target_folder, os.path.basename(target_folder))+'.xml'
        frame_triggers = read_xml_file(xml_dest)
        frame_triggers = frame_triggers * 1e4 # Convert seconds to 10kHz sampling
        if frame_triggers[0] > 340:
            print('First 2p frame was dropped!')
        
        #frame_triggers = replace_missing_frame_triggers(frame_triggers)

        psychopy_loc = data_type + '_PSYCHOPY'
        voltage_files = glob('*.csv') # already in datadir
        assert len(voltage_files) == 1, f'Unique, singular voltage file not found. Check {target_folder}'
    
        #vrec = csvread(voltage_files[0])


    elif 'SCANIMAGE' in data_type:
        print('scanimage two-photon timeframes must be implemented')

    print(f'Num frametriggers detected: {frame_triggers.shape[0]}')
    print('Getting stimulus times and syncing with two-photon frame times...')
    #stim_triggers = medfile1(vrec[],101)
    stim_triggers[stim_triggers < 0] = 0
    stim_triggers = np.diff(stim_triggers)
    
    # findpeaks for stimOn
    # findpeaks for stimOff

    if is_2p_opto:
        print('Processing 2pOpto triggers...')
        #stim_triggers = medfilt1(vrec[],101)
        stim_triggers[stim_triggers < 0] = 0
        #photostim_triggers = findpeaks(stim_triggers, ...)
    
    os.chdir(psychopy_loc)
    #psychopy_file = readmatrix()
    
    ## getting stimstr - 

    #if not is_2p_opto:
    #
    #else:
    #

    assert stim_on.shape[0] == stim_id.shape[0],'Mismatch between stimulus onset times and number of stimulus IDs.'

    stim_on_2p_frame = np.zeros(stim_id.shape[0])
    for s in range(stim_id.shape[0]):
        stim = stim_id[s]
        frame_loc = min(abs(stim - frame_triggers))
        stim_on_2p_frame[s] = frame_loc

    ## Target Stim 2p Frame synchronization
    #if is_2p_opto:
else:
    print('No stimulus triggers recorded for this dataset.')

#if do_neuropil:
#    neuropil_subtraction()


#if do_cascade:
#   ...

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
