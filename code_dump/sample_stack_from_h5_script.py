import os
import pathlib

import glob
import h5py
import tifffile

# Quick script: take the inference results and make samples for each one


def std_dev_from_stack(h5_path, start_frame, end_frame, std_width, std_jmps):
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
    f = h5py.File(h5_path, "r")
    data = f['data']
    
    curr_strt = start_frame
    curr_end = start_frame + std_width
    termination_val = min(end_frame, data.shape[0]) # Don't go beyond the data

    std_dev_sample_list = []

    while curr_end < termination_val:
        sample = data[curr_strt:curr_end, :, :]
        std_dev_sample_list.append()

    try:
        tifffile.imwrite(save_path, std_dev_sample_array)
    except Exception as e:
        print("Error while saving: ", e)


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
    f = h5py.File(h5_path, "r")
    data = f['data']
    data_slice = data[start_frame:end_frame, :, :]
    try:
        tifffile.imwrite(save_path, data_slice)
    except Exception as e:
        print("Error while saving: ", e)
    print(f"Saved sample at {save_path}")

def main():
    sessions_to_run = ['/mnt/md0/BRUKER/TSeries-11052024-1002-004',
                '/mnt/md0/BRUKER/TSeries-11052024-1002-005',
                '/mnt/md0/BRUKER/TSeries-11052024-1002-006',
                '/mnt/md0/BRUKER/TSeries-11052024-1002-008',
                '/mnt/md0/BRUKER/TSeries-11052024-1002-009',
                '/mnt/md0/BRUKER/TSeries-11052024-1002-010',
                '/mnt/md0/BRUKER/TSeries-11052024-1002-011',
                '/mnt/md0/BRUKER/TSeries-11052024-1002-012',
                '/mnt/md0/BRUKER/TSeries-11052024-1002-013',
                '/mnt/md0/BRUKER/TSeries-11052024-1002-014',
                '/mnt/md0/BRUKER/TSeries-11052024-1002-015',
                '/mnt/md0/BRUKER/TSeries-11052024-1002-016',
                '/mnt/md0/BRUKER/TSeries-11052024-1002-017',
                '/mnt/md0/BRUKER/TSeries-11052024-1002-018',
                '/mnt/md0/BRUKER/TSeries-11052024-1002-019',
                '/mnt/md0/BRUKER/TSeries-11052024-1002-020',
                '/mnt/md0/BRUKER/TSeries-11052024-1002-021',
                '/mnt/md0/BRUKER/TSeries-11052024-1002-022',
                '/mnt/md0/BRUKER/TSeries-11052024-1002-023',
                '/mnt/md0/BRUKER/TSeries-11052024-1002-024',
                '/mnt/md0/BRUKER/TSeries-11052024-1002-025',
                '/mnt/md0/BRUKER/TSeries-11052024-1002-026',
                '/mnt/md0/BRUKER/TSeries-11052024-1002-027',
                '/mnt/md0/BRUKER/TSeries-11052024-1002-028',
                '/mnt/md0/BRUKER/TSeries-11052024-1002-029',
                '/mnt/md0/BRUKER/TSeries-11052024-1002-030',
                '/mnt/md0/BRUKER/TSeries-11052024-1002-031',
                '/mnt/md0/BRUKER/TSeries-11052024-1002-032',
                '/mnt/md0/BRUKER/TSeries-11052024-1002-034',
                '/mnt/md0/BRUKER/TSeries-11052024-1002-035',
                '/mnt/md0/BRUKER/TSeries-11052024-1002-036',
                '/mnt/md0/BRUKER/TSeries-11052024-1002-037',
                '/mnt/md0/BRUKER/TSeries-11052024-1002-038',
                '/mnt/md0/BRUKER/TSeries-11052024-1002-039',
                '/mnt/md0/BRUKER/TSeries-11052024-1002-040',
                '/mnt/md0/BRUKER/TSeries-11052024-1002-041',
                '/mnt/md0/BRUKER/TSeries-11052024-1002-042',
                '/mnt/md0/BRUKER/TSeries-11052024-1002-043',
                '/mnt/md0/BRUKER/TSeries-11052024-1002-044',
                '/mnt/md0/BRUKER/TSeries-11052024-1002-045',
                '/mnt/md0/BRUKER/TSeries-11052024-1002-046',
                '/mnt/md0/BRUKER/TSeries-11052024-1002-047']
    start_f = 0
    end_f   = 1000

    for ses in sessions_to_run:
        h5_file_maybe = glob.glob(os.path.join(ses, "inference_results.h5"))
        
        if len(h5_file_maybe) == 0:
            print(f"No full inference results found for {ses}")
            continue
        
        tif_savename = os.path.join(ses, "samples", f"inference_{start_f}_{end_f}.tif")
        
        sample_stack_from_h5(h5_file_maybe[0], tif_savename, start_f, end_f)

if __name__ == "__main__":
    main()