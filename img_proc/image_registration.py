import os
import os.path
import cv2
import math
import h5py
import glob
import numpy as np
from datetime import datetime

import code
    
try:
    cv2.setNumThreads(0)
except:
    pass

try:
    if __IPYTHON__:
        # this is used for debugging purposes only. allows to reload classes
        # when changed
        get_ipython().magic('load_ext autoreload')
        get_ipython().magic('autoreload 2')
except NameError:
    pass


import caiman as cm
from caiman.motion_correction import MotionCorrect
from caiman.source_extraction.cnmf import cnmf as cnmf
from caiman.source_extraction.cnmf import params as params
from caiman.utils.utils import download_demo
from caiman.summary_images import local_correlations_movie_offline

import tifffile

global mc

def register_one_session(parent_dir, mc_dict, keep_memmap, save_sample, sample_name):
    #pass  # For compatibility between running under Spyder and the CLI
    fnames = sorted(glob.glob(os.path.join(parent_dir, "*registered.h5")))
    #fnames = sorted(glob.glob(os.path.join(parent_dir, '*tif')))

    mc_dict['fnames'] = fnames
    mc_dict['upsample_factor_grid'] = 8 # Attempting to fix subpixel registration issue

    opts = params.CNMFParams(params_dict=mc_dict)

# %% start a cluster for parallel processing
    c, dview, n_processes = cm.cluster.setup_cluster(
        backend='local', n_processes=None, single_thread=False)
    mc = MotionCorrect(fnames, dview=dview, **opts.get_group('motion'))

    mc.motion_correct(save_movie=True)

 # Storing results in HDF5 for DeepInterpolation
    if mc_dict['pw_rigid']:
        numframes = len(mc.x_shifts_els)
    else:
        numframes = len(mc.shifts_rig)

    #os.replace(fnames[0], parent_dir + "//registered.h5")
    #datafile = h5py.File(parent_dir + '//registered.h5', 'w')
    os.replace(fnames[0], os.path.join(parent_dir, 'registered.h5'))
    datafile = h5py.File(os.path.join(parent_dir, 'registered.h5'), 'w')
    datafile.create_dataset("mov", (numframes, 512, 512))
    #datafile.create_dataset("mov", (numframes, 128, 128))
    #code.interact(local=dict(globals(), **locals())) 
    
    # Behavior changed since last stable
    #fnames_new = glob.glob(parent_dir + "//*.mmap")
    fnames_new = mc.mmap_file
    frames_written = 0
    
    for i in range(0, math.floor(numframes / 1000)):
        mov = cm.load(fnames_new[0])
        temp_data = np.array(mov[frames_written:frames_written + 1000, :, :])
        datafile["mov"][frames_written:frames_written + 1000, :, :] = temp_data
        frames_written += 1000
        del mov
    
    # handling last point
    if numframes > frames_written:
        mov = cm.load(fnames_new[0])
        temp_data = np.array(mov[frames_written:mov.shape[0], :, :])
        datafile["mov"][frames_written:mov.shape[0], :, :] = temp_data
        del mov
        
    del temp_data        
    cm.stop_server(dview=dview)

    if not keep_memmap:
        for i in range(0, len(fnames_new)):
            os.remove(fnames_new[i])

    if save_sample:
        if not os.path.isdir(os.path.join(parent_dir, "samples")):
            os.makedirs(os.path.join(parent_dir, "samples"))
        #with tifffile.TiffWriter(parent_dir+"//samples//"+sample_name, bigtiff=False, imagej=False) as tif:
        with tifffile.TiffWriter(os.path.join(parent_dir, 'samples', sample_name), bigtiff=False, imagej=False) as tif:
            for i in range(0, min(2000, numframes)):
                curfr = datafile["mov"][i,:,:].astype(np.int16)
                tif.save(curfr)
                
    datafile.close()


def register_bulk(sessions_to_run, sparse=False):
    """
    Function to run CaImAn's motion correction on prepared .h5 stack.

    Parameters:
        sessions_to_run (list): List of strings for each file directory to find data in.
        sparse (bool): True runs piecewise-nonrigid registration (NoRMCorre) after rigid registration.
                       False runs only rigid.
    Returns:
        Nothing. Does disk operations, creates new .h5 for registered calcium movie.
    """
    
    ### dataset dependent parameters
    fr = 30             # imaging rate in frames per second
    decay_time = 1      # Recommended by Ben on March 20th
    
    ### Based on Ben's recommendations
    dxy = (1.0, 1.0)
    max_shift_um = (32, 32)
    patch_motion_um = (64., 64.)
    max_shifts = [int(a/b) for a, b in zip(max_shift_um, dxy)]
    strides = tuple([int(a/b) for a, b in zip(patch_motion_um, dxy)])
    overlaps = (32, 32)
    max_deviation_rigid = 3
    #code.interact(local=locals())

    mc_dict = {
        'fr': fr,
        'decay_time': decay_time,
        'dxy': dxy,
        'pw_rigid': False,
        'max_shifts': max_shifts,
        'strides': strides,
        'overlaps': overlaps,
        'max_deviation_rigid': max_deviation_rigid,
        'border_nan': 'copy',
        'nonneg_movie': False,
        'use_cuda': False,
        'niter_rig': 5
    }

    time_deltas = []
    for i in range(0,len(sessions_to_run)):
        print(sessions_to_run[i])
        t_start = datetime.now()
        #stacks_to_hdf5(sessions_to_run[i], delete_tiffs=True)
        register_one_session(sessions_to_run[i], mc_dict, keep_memmap=False,save_sample=True, sample_name="01_rigid.tif")
        if sparse:
            mc_dict['pw_rigid'] = True
            register_one_session(sessions_to_run[i], mc_dict, keep_memmap=False, save_sample=True, sample_name="02_nonrigid.tif")