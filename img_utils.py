# A module to handle common formats for calcium movies.
import os
import sys
import xml.etree.ElementTree as ET
import code

from glob import glob
import h5py
import tifffile
import numpy as np

from matplotlib import path

import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt

from scipy import signal
from statistics import median
from skimage.draw import polygon2mask

from tqdm import tqdm

def gen_polyline_roi(nm_coord, d_width=10.0, size_x=512, size_y=512):
    '''
    Create a mask for dendrite ROI
    Parameters:
        nm_coord (np.array): 2D NumPy array listing x and y coordinates indicating member pixels.
        d_width (float): Width to draw mask for
        size_x (int): Size of mask to use in first dimension.
        size_y (int): Size of mask to use in second dimension.
    Returns:
        polyline_mask (np.array): 2D NumPy array encoding mask for polyline
    '''
    x = nm_coord[:,1] # These are reversed from MATLAB
    y = nm_coord[:,0]

    r = np.sqrt( (x[1:] - x[0:-1])**2 + (y[1:] - y[0:-1])**2 )

    delta_x = np.multiply((d_width / 2) / r, (y[:-1] - y[1:]))
    delta_y = np.multiply((d_width / 2) / r, (x[1:] - x[:-1]))

    new_x = x + np.append(delta_x, delta_x[-1])
    new_y = y + np.append(delta_y, delta_y[-1])

    delta_x = np.multiply((-d_width / 2) / r, (y[:-1] - y[1:]))
    delta_y = np.multiply((-d_width / 2) / r, (x[1:] - x[:-1]))

    new_x = np.append(new_x, np.flip(x + (np.append(delta_x, delta_x[-1]))))
    new_y = np.append(new_y, np.flip(y + (np.append(delta_y, delta_y[-1]))))

    new_x = np.append(new_x, new_x[0])
    new_y = np.append(new_y, new_y[0])

    polyline_mask = polygon2mask((size_x, size_y), np.array([new_x, new_y]).T)
    return polyline_mask.T

def in_polygon(xq, yq, xv, yv):
    shape = xq.shape
    xq = xq.reshape(-1)
    yq = yq.reshape(-1)
    xv = xv.reshape(-1)
    yv = yv.reshape(-1)
    q = [(xq[i], yq[i]) for i in range(xq.shape[0])]
    p = path.Path([(xv[i], yv[i]) for i in range(xv.shape[0])])
    return p.contains_points(q).reshape(shape)

def get_h5_size(h5_path):
    '''
    Just return the size of the given .h5 file.
    
    Parameters:
        h5_path(str): Path to h5 file.
    Returns:
        dims (tuple or None): Dimensions of the file.
            If file does not exist, return None.
    '''
    assert h5_path.endswith(('.h5', '.hdf5')), f"{h5_path} does not end with .h5 or .hdf5."
    try:
        with h5py.File(h5_path, 'r') as f:
            key = 'mov' if 'mov' in f.keys() else 'data'
            return f[key].shape
    except FileNotFoundError:
        print("Cannot give size for f{h5_path} because it wasn't found.")
        return None
    

def prctfilt1(x, n=3, blksz=1000, dim=None):
    """
    One-dimensional percentile filter with support for chunking and multidimensional arrays.
    Equivalent to the MATLAB custom prctfilt1 function.
    
    Parameters:
    x : array_like
        Input signal or array
    n : int, optional
        Window size (default = 3)
    blksz : int, optional
        Block size for chunking. If None, processes all data at once.
    dim : int, optional
        Dimension along which to apply the filter. If None, applies along the first non-singleton dimension.
        
    Returns:
    y : ndarray
        Filtered signal or array
    """
    x = np.asarray(x)
    
    # Handle dimension argument
    if dim is None:
        # Find first non-singleton dimension
        dim = 0
        while dim < x.ndim and x.shape[dim] == 1:
            dim += 1
        if dim == x.ndim:
            return x.copy()  # All dimensions are singleton
            
    # Move the desired dimension to the front
    if dim != 0:
        perm = np.arange(x.ndim)
        perm[0], perm[dim] = dim, 0
        x = np.transpose(x, perm)
        restore_perm = np.argsort(perm)
    
    # Set default block size if not specified
    if blksz is None:
        blksz = x.shape[0]
    
    # Determine padding size
    if n % 2 == 1:  # odd
        m = (n - 1) // 2
    else:  # even
        m = n // 2
    
    # Create output array
    y = np.zeros_like(x)
    
    # Process each "column" (i.e., along all other dimensions)
    for idx in np.ndindex(x.shape[1:]):
        # Extract the vector to filter
        vector = x[(slice(None),) + idx]
        
        # Pad the vector
        padded = np.zeros(len(vector) + 2*m)
        padded[m:m+len(vector)] = vector
        
        # Process in chunks to save memory
        for i in range(0, len(vector), blksz):
            end_idx = min(i + blksz, len(vector))
            chunk_size = end_idx - i
            
            # Create window slices for this chunk
            windows = np.zeros((n, chunk_size))
            for j in range(n):
                windows[j] = padded[i+j:i+j+chunk_size]
            
            # Calculate 20th percentile for each window
            y[(slice(i, end_idx),) + idx] = np.percentile(windows, 20, axis=0)
    
    # Restore original dimensions if needed
    if dim != 0:
        y = np.transpose(y, restore_perm)
        
    return y

def filter_baseline_dF_comp(raw, pts = 99):
    F_temp = raw
    F_temp = np.concatenate((np.repeat(np.mean(F_temp[2:5]),pts),F_temp))
    F_temp = np.concatenate((F_temp, np.repeat(np.mean(F_temp[-5:-2]),pts)))
    # 25th percentile medfilt
    F_temp = signal.medfilt(F_temp, pts)
    # remove padding
    raw_new = F_temp[pts:-pts]
    #code.interact(local=dict(globals(), **locals()))
    raw_new = np.divide((raw - raw_new), raw_new)
    
    #raw_newlpf = signal.medfilt(raw_new, 91)
    raw_newlpf = prctfilt1(raw_new, 91)

    raw_new = raw_new - raw_newlpf
    
    return raw_new

def replace_missing_frame_triggers(frame_triggers):
    '''
    Replace any relative frametriggers that may be missing from a series

    Parameters:
        frame_triggers (np.array): 1D array of relative frame times from
            start of recording. Assumes 10kHz sampling rate.
    Returns:
        frame_triggers_adj (np.array): 1D array of relative frame times from
            start of recording. If any triggers are missing, now included.
    '''
    new_frame_triggers = []
    med_period = median(np.diff(frame_triggers))
    for k in range(1, frame_triggers.shape[0]):
        if (frame_triggers[k] - frame_triggers[k-1]) > (med_period + med_period/4):
            new_frame_triggers.append(frame_triggers[k-1] + med_period)
        new_frame_triggers.append(frame_triggers[k])
    return np.array([new_frame_triggers])
 
def genfromtxt_with_progress(filename, **kwargs):
    '''
    np.genfromtxt call, but wrapping file with tqdm to add a progress bar
        while iterating over the file lines.
    
    Parameters:
        filename (str): Destination to read from. Local or absolute.
        **kwargs - arguments for np.genfromtxt.
    Returns:
        (np.array): Results of genfromtxt call.
    '''
    with open(filename, 'r') as f:
        total_lines = sum(1 for line in f)
    with open(filename, 'r') as f:
        return np.genfromtxt(tqdm(f, total=total_lines), **kwargs)
 
def neuropil_subtraction(dff_mat, dff_neuro, is_soma):
    '''
    Remove out the neuropil signal from somas.
    Uses statsmodels 'RLM'.

    Parameters:
        dff_mat (np.array): 2d array of (time, cell) for the dff trace per mask
        dff_neuro (np.array): 1d array of dff trace for the neuropil signal
        is_soma (list of Bool): Identity of whether cells are somas or not.
    Returns:
        dff_depilled (np.array): 2d array of (time, cell).
            Contains dff_mat, but fit to the slope found 
        dff_slopes (np.array): 1d array of each slope found through the robustfit process.
    '''
    # statsmodels.sm RLM (Robust Linear Model) attempt
    # scikit-learn attempt - RANSAC Regressor
    #
    # From MATLAB robustfit...
    #   The algorithm uses iteratively reweighted least squares with the bisquare weighting function. 
    #   By default, ROBUSTFIT adds a column of ones to X, corresponding to a
    #   constant term in the first element of B.  Do not enter a column of ones
    #   directly into the X matrix.

def dendrite_subtraction():
    '''
    Remove dendrite signal from spine ROIs.
    Navigate it here once you have a way to get it out of main script.
    Parameters:
        -
    Returns:
        -
    '''
    

def gen_stim_cyc():
    '''
    Gather waveforms for each stimulus condition, for each ROI.
    Navigate it here once you have a way to get it out of main script.
    Parameters:
        - 
    Returns:
        -
    '''

def read_xml_file(fname):
    '''
    Extract frame timing information from BRUKER output file.
    
    Parameters:
        fname (str): String indicating path to read XML data from.
    Returns:
        rec_info (np.array): Array containing relative frame timing information.
    '''
    try:
        tree = ET.parse(fname)
        root = tree.getroot()
        # Count the number, then get the values
        rel_frametimes = [child.attrib['relativeTime'] for child in root[2] if child.tag == 'Frame']
        return np.array(rel_frametimes).astype('float')
    
    except FileNotFoundError:
        print(f'File {fname} was not found.')
    except ET.ParseError:
        print(f'Error parsing XML file {fname}')
    except Exception as e:
        print('Exception during read_xml_file: ', e)

def get_target_folders_v2(loc, date, fnames, filetype='TSeries'):
    '''
    Grab folder(s) that match given patterns.
    loc (str): Location to parent directory within which to search.
    date (str): MMDDYYYY date string to search for
    fnames (list or int): List of names to gather based on this querey.
    filetype (str): Specify which kind of file from the scanner is required.
    '''
    if 'BRUKER' in loc:
        folder_list = sorted(glob(loc + filetype + '-' + date + '*'))
    elif 'SCANIMAGE' in loc:
        date_str = '-'.join([date[4:], date[:2], date[2:4]])
        folder_list = sorted(glob(loc + filetype + '_' + date_str + '*'))
    
    assert len(folder_list) > 0, 'No target folders matched given specification.'

    if isinstance(fnames, int):
        suffixes = ["{:03d}".format(fnames)]
    else:
        suffixes = ["{:03d}".format(fname) for fname in fnames]

    dir_list = []
    for f in folder_list:
        if os.path.isdir(f) and any(f.endswith(suffix) for suffix in suffixes):
            dir_list.append(f)
    
    return dir_list



def pearson_r_from_movie(h5_path, template_start=0,template_end=1000):
    '''
    Characterizes amount of movement across a calcium movie over time.
    Not recommended anymore. After using with enough movies, it's not
    that informative of a measure compared to viewing sample tif stacks.

    Parameters:
        h5_path (str): String encoding the path to the movie to see.
    Returns:
        corrs (np.array): 1D array for Pearson's r per frame of given movie.
    '''
    with h5py.File(h5_path, 'r') as f:
        key = 'mov' if 'mov' in f.keys() else 'data'
        data = f[key]
        n_frames = data.shape[0]
        
        template = np.mean(data[template_start:min(template_end, n_frames) :, :], axis=0)
        template = template.flatten()

        r_vec = np.empty((n_frames,))
        for i in range(n_frames):
            r_vec[i] = np.corrcoef(template, data[i,:,:].flatten())[0,1]
    return r_vec

def tif_stacks_to_h5(tif_dir, h5_savename, h5_key='mov', delete_tiffs=False, frame_offset=True, offset=30):
    '''
    Convert .tif stacks from BRUKER/SCANIMAGE to monolithic .h5 files.
    If frame_offset, frames from the start and end of the series are appended to either
    end of the resulting .h5 and flipped. This is performed when data is meant to be
    given to DeepInterpolation.

        Parameters:
            tif_dir (str): String for path of directory to convert tifs in.
            h5_savename (str): Name of conversion .h5 file to save.
            h5_key (str): h5 key to save data under. CaImAn assumes 'mov'.
            delete_tiffs (bool): Whether to remove .tif files during conversion
            frame_offset (int): 
        Returns:
            None - Writes .h5 file to disk at 'h5_savename' containing calcium movie data.
    '''
    #assert len(tif_fnames) > 0, "List of tifs to convert must be nonzero in length."
    tif_fnames = sorted(glob(os.path.join(tif_dir, "*.tif")))
    # Don't know how to get width, height without loading into memory.
    #first_tif_handle = tifffile.TiffFile(tif_fnames[0])
    #stack_depth = len(first_tif_handle.pages)
    first_tif = tifffile.imread(tif_fnames[0])
    if len(first_tif.shape) < 3:
        stack_depth = 1
        stack_width, stack_height = first_tif.shape
    else:
        stack_depth, stack_width, stack_height = first_tif.shape

    # All multi-frame .tif stacks should be same size except for the last.
    # Quick check w/out loading into RAM.
    if stack_depth > 1:
        for i in range(1, len(tif_fnames) - 1):
            tif_stack_handle = tifffile.TiffFile(tif_fnames[i])
            this_stack_depth = len(tif_stack_handle.pages)
            assert this_stack_depth == stack_depth, \
                f"Stack sizes inconsistent : expected {stack_depth} frames \
                    but got {this_stack_depth} for file {tif_fnames[i]}"
    
    # Adding offsets, flipped to front and back of data.
    if frame_offset: 
        first_frames = np.zeros((offset, stack_width, stack_height))
        last_frames  = np.zeros((offset, stack_width, stack_height))
        
        if stack_depth == 1: # .ome.tifs
            # Check for sufficient frames to prevent overlap of frames
            assert len(tif_fnames) > 2 * offset, f"Frame offset is True for {tif_dir}, and offset is {offset}. Insufficient frames ({len(tif_fnames)})"
            last_fnames  = tif_fnames[(-1*offset):]
            for i in range(offset):
                first_frames[i,:,:] = tifffile.imread(tif_fnames[i], is_ome=False)
                last_frames[i,:,:]  = tifffile.imread(last_fnames[i], is_ome=False)
            last_stack_length = 1
        
        else: # multi-page .tifs
            first_stack = tifffile.imread(tif_fnames[0])
            last_stack  = tifffile.imread(tif_fnames[-1])
            
            first_frames[:,:,:] = first_stack[0:offset, :, :]

            if last_stack.shape[0] < offset:
                #raise IndexError(f'Frame flip offset of {offset} exceeds \
                #                 last stack len {last_stack.shape[0]}. Write handling for this.')
                second_last_stack = tifffile.imread(tif_fnames[-2])
                both_last = np.concatenate((second_last_stack, last_stack), axis=0)
                last_frames = both_last[(-1 * offset), :, :]
            else:    
                last_frames[:,:,:] = last_stack[(-1 * offset):, : ,:]
            last_stack_length = last_stack.shape[0]
            del first_stack
            del last_stack
    # Just to guarantee last_stack_length
    else:
        last_stack = tifffile.imread(tif_fnames[-1])
        last_stack_length = last_stack.shape[0]

    # first and last offset, all regular-length stacks, last stack
    out_data_frames = (offset * 2) + (stack_depth * (len(tif_fnames) - 1)) + last_stack_length
    # Writing the main movie itself
    # multi-frame tif size consistency check
    f_out = h5py.File(h5_savename, 'w')
    f_out.create_dataset(h5_key, (out_data_frames, stack_width, stack_height))
    write_start_ind, write_end_ind = (0, 0)

    if frame_offset:
        f_out[h5_key][0:offset, :, :] = np.flip(first_frames, axis=0)
        write_end_ind += 30
    
    for i in tqdm(range(len(tif_fnames) - 1), desc="Writing all but last stack...", ncols=75):
        this_stack_data = tifffile.imread(tif_fnames[i], is_ome=False)
        write_start_ind = write_end_ind
        write_end_ind = write_start_ind + stack_depth
        f_out[h5_key][write_start_ind:write_end_ind, :, :] = this_stack_data
    
    # Write the end of the file
    # This will also require knowing if the last stack was at least 'offset' many pages.
    last_stack = tifffile.imread(tif_fnames[-1], is_ome=False)
    write_start_ind = write_end_ind
    
    print('Writing last stack...')
    f_out[h5_key][write_start_ind:-offset, :, :] = last_stack
    f_out[h5_key][-offset:, :, :] = last_frames

    f_out.close()

def grab_visuals(dir, start_ind=0, end_ind=1000):
    '''
    ImageJ/FIJI refuses to load singular .ome.tif files anymore.
    This makes samples for directories involved so I can do my job.
    
    Parameters:
        dir (str): Directory to take samples from per channel.
        start_ind (int): Time index to start sample from.
        end_ind (int): Time index to end sample from.

    Returns:
        Nothing, just creates files in sample directory of the given dir.

    Ex.
    ~ workdirs = sorted(glob("/mnt/md0/BRUKER/TSeries-11032024"))
    ~ for w in workdirs:
    ~   grab_visuals(w)
    '''
    try:
        print(f'Creating samples for {dir}')
        os.chdir(dir)
        ch1_fnames = sorted(glob("TSeries*Ch1*.ome.tif"))
        ch2_fnames = sorted(glob("TSeries*Ch2*.ome.tif"))
        
        if  not os.path.isdir(os.path.join(dir, 'samples')) and (len(ch1_fnames) > 0 or len(ch2_fnames) > 0):
            os.mkdir(os.path.join(dir, 'samples'))
            print('New samples directory created')

        if len(ch1_fnames) > 0:
            ch1_savename = os.path.join('samples', f'ch1_f{start_ind}-{end_ind}.tif')
            ch1_first = tifffile.imread(ch1_fnames[0], is_ome=False)
            tifffile.imwrite(ch1_savename, ch1_first[start_ind:end_ind, :, :])
            print('Saved Ch1 sample')
        
        if len(ch2_fnames) > 0:
            ch2_savename = os.path.join('samples', f'ch2_f{start_ind}-{end_ind}.tif')
            ch2_first = tifffile.imread(ch2_fnames[0], is_ome=False)
            tifffile.imwrite(ch2_savename, ch2_first[start_ind:end_ind, :, :])
            print('Saved Ch2 sample')

    except IndexError:
        print(f'Indices of {start_ind} and {end_ind} not working. :(')

def sample_stack_from_h5(h5_path, save_path, start_frame, end_frame):
    '''
    Load the main 'data'/'mov' time x width x height data from a given h5 file.
    Slice from start_frame -> end_frame in time.
    Save this data at the save_path as a viewable .tif stack.

    Large .h5 files are not viewable as memory-loadable objects for ImageJ/Fiji,
    so samples are created as drag-and-drop-'able .tifs.
    
        Parameters:
            h5_path (str): Absolute path to the .h5 file to be read.
            save_path (str): Absolute path for the .tif file to be saved.
            start_frame (int): Starting time index for the sample.
            end_frame (int): Ending time index for the sample.
        Returns:
            e (exception): If exception is raised while writing data.
    
    '''
    with h5py.File(h5_path, 'r') as f:
        key = 'mov' if 'mov' in f.keys() else 'data'
        data = f[key]
        data_slice = data[start_frame:end_frame, :, :]
        
        try:
            if not os.path.exists(os.path.dirname(save_path)):
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
            tifffile.imwrite(save_path, data_slice)
        except Exception as e:
            print("Error while saving: ", e)
        print(f"Saved sample at {save_path}")


def std_dev_from_h5(h5_path, start_frame, end_frame, std_width, std_jmps):
    '''
    Load the main 'data'/'mov' [time, width, height] data from a given h5 file's path.
    Slice from start_frame -> end_frame in time.
    Take standard deviations through time with described std-dev widths and jumps per.
    Save frames at the specified save location.

        Parameters:
            h5_path (str): Absolute path to the .h5 file to be read.
            save_path (str): Absolute path to the .tif file to be saved.
            start_frame (int): Starting time index to take samples.
            end_frame (int): Ending time index to take samples.
            std_width (int): Width to take std_deviations from.
            std_jmps (int): What distance forward in time to take between samples.

        Returns:
            e (exception): If exception is raised while writing data.
    '''
    #f = h5py.File(h5_path, "r")
    #data = f['data']
    with h5py.File(h5_path, 'r') as f:
        key = 'mov' if 'mov' in f.keys() else 'data'
        data = f[key]

        curr_strt = start_frame
        curr_end = start_frame + std_width
        termination_val = min(end_frame, data.shape[0]) # Don't go beyond the data

        std_dev_sample_list = []

        while curr_end < termination_val:
            sample = data[curr_strt:curr_end, :, :]
            std_dev_sample_list.append()

        std_dev_sample_array = np.array(std_dev_sample_list)

        try:
            tifffile.imwrite(save_path, std_dev_sample_array)
        except Exception as e:
            print("Error while saving: ", e)


def deinterleave_movies(parent_dir, scope_format):
    """
    Splits tiff stacks where two channels have been interleaved.
    Matches how our SCANIMAGE acquisitions are done for 2ch.

    TODO : 
        - escaped slashes aren't system-agnostic. Change this w/ os.path?
        - Confirm shutil behavior is system-agnostic. Known differences between
            Windows, Ubutu behavior
    """
    is_bruker = 'BRUKER' in parent_dir
    is_scanimage = 'SCANIMAGE' in parent_dir

    fnames = glob(os.path.join(parent_dir, '*.tif'))
    assert len(fnames) > 0, "No valid tif files found in {}".format(parent_dir)
    
    #os.mkdir(parent_dir + '//ch1')
    #os.mkdir(parent_dir + '//ch2')

    os.mkdir(os.path.join(parent_dir, 'ch1'))
    os.mkdir(os.path.join(parent_dir, 'ch2'))

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
        
        s

