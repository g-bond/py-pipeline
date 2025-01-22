# A module to handle common formats for calcium movies.
import os
import code

import cv2
import glob
import h5py
import tifffile
import numpy as np

def tif_stacks_to_h5(tif_dir, h5_savename, h5_key='mov', delete_tiffs=False, frame_offset=False, offset=30):
    '''
    Convert .tif stacks from BRUKER/SCANIMAGE to monolithic .h5 files.
    If frame_offset, frames from the start and end of the series are appended to either
    end of the resulting .h5 and flipped. This is performed when data is meant to be
    given to DeepInterpolation.

        Parameters:
            tif_dir (str): Path to directory containing .tif files to convert.
            h5_savename (str): Name of conversion .h5 file to save.
            h5_key (str): h5 key to save data under. CaImAn assumes 'mov'.
            delete_tiffs (bool): Whether to remove .tif files during conversion
            frame_offset (int): 
        Returns:
            None - Writes .h5 file to disk at 'h5_savename' containing calcium movie data.
    '''

    tif_fnames = sorted(glob.glob(os.path.join(tif_dir, "*.tif")))
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

    for i in range(len(tif_fnames) - 1):
        this_stack_data = tifffile.imread(tif_fnames[i], is_ome=False)
        write_start_ind = write_end_ind
        write_end_ind = write_start_ind + stack_depth
        f_out[h5_key][write_start_ind:write_end_ind, :, :] = this_stack_data
    
    # Write the end of the file
    # This will also require knowing if the last stack was at least 'offset' many pages.
    last_stack = tifffile.imread(tif_fnames[-1], is_ome=False)
    write_start_ind = write_end_ind

    f_out[h5_key][write_start_ind:-offset, :, :] = last_stack
    f_out[h5_key][-offset:, :, :] = last_frames

    f_out.close()


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
    # Assert h5 exists
    #f = h5py.File(h5_path, "r")
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


def avi_from_h5(h5_path, save_path, start_index, end_index, fps=30, chunk_size=1000):
    '''
    Create an .avi movie from a .h5 movie.
    
    Currently has problems.
    - Normalization per frame causes flickering of global brightness.
    - No ability to change brightness and contrast on the fly.

    For QC, seems that .tif export is still the best option.
    '''
    with h5py.File(h5_path, 'r') as f:
        key = 'mov' if 'mov' in f.keys() else 'data'
        data = f[key]

        if start_index < 0 or end_index >= data.shape[0]:
            raise ValueError("Invalid start or end index.")
        
        first_frame = data[start_index]
        height, width = first_frame.shape

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video_out = cv2.VideoWriter(save_path, fourcc, fps, (width, height), isColor=False)

        # Processing by chunk to keep memory utilization reasonable.
        try:
            for chunk_start in range(start_index, end_index+1, chunk_size):
                chunk_end = min(chunk_start + chunk_size, end_index + 1)
                chunk = data[chunk_start:chunk_end,:,:]

                for frame in chunk:
                    frame_norm = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX)
                    frame_8bit = frame_norm.astype(np.uint8)
                    video_out.write(frame_8bit)
        finally:
            video_out.release()


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

    fnames = glob.glob(os.path.join(parent_dir, '*.tif'))
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
        
        

# Only including this here for testing purposes - gnb, Dec'24
def main():
    test_data_dir = '/home/gnb/Documents/pipeline_test_data/BRUKER/TSeries-03282023-1243-012/temp'
    h5_savename = os.path.join(test_data_dir, 'out.h5')
    tif_stacks_to_h5(test_data_dir, h5_savename=h5_savename, frame_offset=True)

if __name__ == '__main__':
    main()

