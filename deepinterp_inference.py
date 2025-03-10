import os
import sys
import glob
import h5py
import code

sys.path.append('/home/schollab-dion/Documents/deepinterpolation-0.2.0')
from deepinterpolation.generator_collection import SingleTifGenerator, OphysGenerator
from deepinterpolation.inference_collection import core_inference
 
import tensorflow as tf

from img_utils import get_h5_size, sample_stack_from_h5
from wx_denoising_gui import run_selection_configuration, DirectorySelection

# TF will allocate all VRAM on GPU and crash due to OOM if this is not done. - gnb ~Dec24
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            #tf.config.set_logical_device_configuration(
            #    gpu,
            #    [tf.config.LogicalDeviceConfiguration(memory_limit=1024 * 22)]
            #)
    except RuntimeError as e:
        print(e)


if __name__ == '__main__':
    sessions_to_run = DirectorySelection.get_directories()
    data_lengths = []
    for path in sessions_to_run:
        data_size_maybe = get_h5_size(os.path.join(path, 'registered.h5'))
        if data_size_maybe:
            data_lengths.append(data_size_maybe[0])
        else:
            data_lengths.append(-1)

    process_choices = run_selection_configuration(sessions_to_run, data_lengths)
    
    do_denoising = process_choices['column1']
    make_samples = process_choices['column2']
    
    #code.interact(local=dict(globals(), **locals())) 
    for i in range(len(sessions_to_run)):
        if do_denoising[i]:
            # Quick check - deepinterp inference will only work with field "data"
            #       but "mov" is default for CaImAn.
            parent_dir = sessions_to_run[i]
            data_file_maybe = glob.glob(os.path.join(parent_dir, "registered.h5"))
            data_file = data_file_maybe[0]
            #code.interact(local=dict(globals(), **locals()))
            f = h5py.File(data_file, "a")
            if 'mov' in f.keys():
                f["data"] =f["mov"]
                del f["mov"]
            f.close()

            generator_param = {}
            inference_param = {}

            # We are reusing the data generator for training here.
            generator_param["pre_post_frame"] = 30
            generator_param["pre_post_omission"] = 0
            generator_param[
                "steps_per_epoch"
            ] = -1
            # No steps necessary for inference as epochs are not relevant.
            # -1 deactivate it.
            generator_param["train_path"] = os.path.join(
                sessions_to_run[i], "registered.h5"
            )
            #generator_param["train_path"] = os.path.join(
            #    "/mnt/md0/BRUKER/TSeries-11042024-1556-031/registered.h5"
            #)

            generator_param["batch_size"] = 1 # Batch size >1 crashes due to OOM
            generator_param["start_frame"] = 0
            #generator_param["end_frame"] = 1100  # Just a sample to check model quality.
            generator_param["end_frame"] = -1 # -1 to go to the end.
            
            generator_param[
                "randomize"
            ] = 0
            # This is important to keep the order
            # and avoid the randomization used during training
            inference_param["model_path"] = glob.glob(
                os.path.join(sessions_to_run[i], "*transfer_model.h5")
            )[0]
            #inference_param[
            #    "model_path"
            #] = r"/mnt/md0/BRUKER/TSeries-11042024-1556-031/2024_12_06_13_25_mean_squared_error_transfer_model.h5"
            
            # Replace this path to where you want to store your output file
            inference_param[
                "output_file"
            ] = os.path.join(sessions_to_run[i], "inference_results.h5")

            jobdir = "./"

            try:
                os.mkdir(jobdir)
            except Exception:
                print("folder already exists")

            data_generator = OphysGenerator(generator_param)
            inference_class = core_inference(inference_param, data_generator)
            inference_class.run()

            if make_samples[i]:
                low_ind  = process_choices['low_numbers'][i]
                high_ind = process_choices['high_numbers'][i]
                acq_name = os.path.join(sessions_to_run[i], 'inference_results.h5')
                savename = os.path.join(sessions_to_run[i], 'samples', f'denoised_f{low_ind}_{high_ind}.tif')
                sample_stack_from_h5(acq_name, savename, low_ind, high_ind)