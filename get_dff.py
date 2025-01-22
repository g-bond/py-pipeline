# getDFF
# processing script for extracting traces from calcium movies
# presupposes registration, denoising optional
import os
from glob import glob

import tifffile
import roifile

import math
import h5py
import numpy as np

from tqdm import tqdm

from img_utils import in_polygon, filter_baseline_dF_comp, get_target_folders_v2

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

# Cascade filenames here

folder_list = get_target_folders_v2(save_location, data_type,file_num, 'TSeries')
target_folder = folder_list[0]
os.chdir(target_folder)

# ROI file handling
try:
    roi_list = roifile.roiread('RoiSet.zip')
except FileNotFoundError:
    print(f'No ROI file for {target_folder}')

num_cells = len(roi_list)
located_dend_roi = False
for i in range(num_cells):
    print('asdf')
    if 'PolyLine':
        located_dend_roi = True
        break

# .h5 processed calcium movie handling
## don't use max bytes, use is_deep_interp instead
try:
    if is_deep_interp:
        h = h5py.File('inference_results.h5', 'r')
    else:
        h = h5py.File('registered.h5', 'r')
except FileNotFoundError:
    print(f'Processed file not found. \n Check destination or "is_deep_interp" flag.')

datName = [key for key in h.keys()][0]
totalFrames, sizeX, sizeY = h[datName].shape

roi_list = roifile.roiread("RoiSet.zip")
numCells = len(roi_list)

x = np.linspace(0, sizeX-1, sizeX)
y = np.linspace(0, sizeY-1, sizeY)
x, y = np.meshgrid(x, y)
mask2d = np.zeros((numCells, sizeX, sizeY))
for cc in tqdm(range(numCells), desc="getting masks", ncols=75):
    nmCoord = roi_list[cc].coordinates()
    mask2d[cc,:,:] = in_polygon(x, y, nmCoord[:,0], nmCoord[:,1])
raw_cell_traces = np.zeros((totalFrames, numCells))
dff = np.zeros((totalFrames, numCells))
for f_i in tqdm(range(math.ceil(totalFrames / chnk)), desc="Extracting...", ncols=75):
    start = int(f_i * chnk)
    stop = min(int((f_i + 1) * chnk), totalFrames)
    imgstack = h[datName][start:stop, :, :]
    for cc in range(numCells):
        nz = np.nonzero(mask2d[cc,:,:])
        raw_cell_traces[start:stop, cc] = np.mean(imgstack[:,nz[0], nz[1]], axis=1)

print("Calculating dff for all cells")
for cc in range(numCells):
    dff[:,cc] = filter_baseline_dF_comp(raw_cell_traces[:,cc], 99*4+1)
print("done")