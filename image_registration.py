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





def stacks_to_hdf5(parent_dir, delete_tiffs):
    """
    Convert tiff/ome.tiff files to 

    TODO:
    - Check that loading the first ome.tiff file with this method only loads
        one frame. Remember that sometimes loading the first of an ome.tiff
        series for some software causes it to load the whole series.
        Flag for suppressing this behavior.
    - 'fnames' and 'f_out' should have platform-agnostic path concatenation
        rather than escaped slashes ("//"). 
        
        e.g. import os
        os.path.join(os.path.curdir, 'file.name')
    """
    
    fnames = sorted(glob.glob(parent_dir + "//.tif"))
    first_fname_handle = tifffile.TiffFile(fnames[0])

    stack_size = len(first_fname_handle.pages)
    img_width, img_height = first_fname_handle.pages[0]
    
    if stack_size == 1: # Single frames per file in old acquisitions
        first_30 = np.zeros((30, img_width, img_height))
        last_30  = np.zeros((30, img_width, img_height))
        last_30_fnames = fnames[-30:]
        for i in range(30):
            first_30[i,:,:] = tifffile.imread(fnames[i], is_ome=False))
            last_30[i,:,:]  = tifffile.imread(last_30_fnames, is_ome=False)
        num_frames = len(fnames) + 60
        
    else: # Stacks of frames per file.
        # All except last stack ought to be same size.
        for i in range(1, len(fnames - 1)):
            tif_handle = tifffile.TiffFile(fnames[i])
            num_pages_in_tif = len(tif_handle.pages)
            assert num_pages_in_tif == stack_size, \
                "Stack sizes inconsistent : first stack is of size {}, \
                but file {} has {} pages. Data may be malformed. \
                Investigate {}".format(stack_size, fname[i], num_pages_in_tif, parent_dir)
        mov = tifffile.imread(fnames[i])
        first_30 = mov[0:30,:,:]
        last_stack = tifffile.imread(fnames[-1])
        last_30 = last_stack[-30:,:,:]
        num_frames = (stack_size * (len(fnames) - 1)) + \
                        last_stack.shape[0] + 60
    
    f_out = h5py.File(parent_dir + "//unregistered.h5", "w")
    f_out.create_dataset("mov", (num_frames, img_width, img_height))

    # First and last 30 frames removed by DeepInterpolation moving window
    f_out["mov"][0:30,:,:] = np.flip(first_30, axis=0)
    
    # Adding the acquisition itself in the middle
    for i in range(len(fnames) - 1):
        index = 30 + (i * stack_size)
        mov = tifffile.imread(fnames[i], is_ome=False)
        f_out["mov"][index:index + stack_size,:,:] = mov
        
    # Add last 30 flipped, updating index with size of last loaded frame file.
    index = index + mov.shape[0] if stack_size == 1 else index + 1
    f_out["mov"][index:,:,:] = np.flip(last_30, axis=0)
    f_out.close() # Done.
                      
def deinterleave_movies(parent_dir, scope_format):
    """
    Splits tiff stacks where two channels have been interleaved.
    Matches how our SCANIMAGE acquisitions are done for 2ch.

    TODO : 
        - escaped slashes aren't system-agnostic. Change this w/ os.path?
    """
    is_bruker = 'BRUKER' in parent_dir
    is_scanimage = 'SCANIMAGE' in parent_dir

    fnames = glob.glob(parent_dir + '//*.tif')
    assert len(fnames) > 0, "No valid tif files found in {}".format(parent_dir)
    
    os.mkdir(parent_dir + '//ch1')
    os.mkdir(parent_dir + '//ch2')

    if scope_format == "BRUKER":
        # Separate files for each
        ch_1_fnames = [f for f in fnames if 'Ch1' in f]
        ch_2_fnames = [f for f in fnames if 'Ch2' in f]
        # Not presuming same number for both, but there usually should be.
        for i in range(len(ch_1_fnames)):
            shutil.move(ch_1_fnames[i], "//".join([parent_dir, 'ch1', f"{i:05d}", ".tif"]))
        for i in range(len(ch_2_fnames)):
            shutil.move(ch_2_fnames[i], "//".join([parent_dir, 'ch2', f"{i:05d}", ".tif"]))
            
    elif scope_format == "SCANIMAGE":
        # Interleaved frames in the same file. Read, split, write separate channels.
        for i in range(len(fnames)):
            mov = tifffile.imread(fnames[i])
            ch_1 = mov[::2,:,:] # Even frames, including first
            ch_2 = mov[1::2,:,:]# Odd frames
            tifffile.imwrite('//'.join([parent_dir, 'ch1', f"{i:05d}", "*.tif"]), ch_1)
            tifffile.imwrite('//'.join([parent_dir, 'ch2', f"{i:05d}", "*.tif"]), ch_2)
            
    else:
        assert False, "{} is improper scope format. \
                Give either BRUKER or SCANIMAGE as second argument.".format(scope_format)
        
        



