import os
import sys
import glob
import h5py
import code

sys.path.append('/home/schollab-beyonce/Documents/deepinterpolation-0.2.0')
from deepinterpolation.generator_collection import SingleTifGenerator, OphysGenerator
from deepinterpolation.inference_collection import core_inference
 
import pathlib
import tensorflow as tf

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

sessions_to_run = ['/mnt/md0/BRUKER/TSeries-11052024-1002-005',
                '/mnt/md0/BRUKER/TSeries-11052024-1002-008',
                '/mnt/md0/BRUKER/TSeries-11052024-1002-012',
                '/mnt/md0/BRUKER/TSeries-11052024-1002-013',
                '/mnt/md0/BRUKER/TSeries-11052024-1002-014',
                '/mnt/md0/BRUKER/TSeries-11052024-1002-016',
                '/mnt/md0/BRUKER/TSeries-11052024-1002-017',
                '/mnt/md0/BRUKER/TSeries-11052024-1002-019',
                '/mnt/md0/BRUKER/TSeries-11052024-1002-020',
                '/mnt/md0/BRUKER/TSeries-11052024-1002-021',
                '/mnt/md0/BRUKER/TSeries-11052024-1002-022',
                '/mnt/md0/BRUKER/TSeries-11052024-1002-024',
                '/mnt/md0/BRUKER/TSeries-11052024-1002-025',
                '/mnt/md0/BRUKER/TSeries-11052024-1002-027',
                '/mnt/md0/BRUKER/TSeries-11052024-1002-028',
                '/mnt/md0/BRUKER/TSeries-11052024-1002-030',
                '/mnt/md0/BRUKER/TSeries-11052024-1002-031',
                '/mnt/md0/BRUKER/TSeries-11052024-1002-032',
                '/mnt/md0/BRUKER/TSeries-11052024-1002-035',
                '/mnt/md0/BRUKER/TSeries-11052024-1002-036',
                '/mnt/md0/BRUKER/TSeries-11052024-1002-037',
                '/mnt/md0/BRUKER/TSeries-11052024-1002-038',
                '/mnt/md0/BRUKER/TSeries-11052024-1002-041',
                '/mnt/md0/BRUKER/TSeries-11052024-1002-042',
                '/mnt/md0/BRUKER/TSeries-11052024-1002-043',
                '/mnt/md0/BRUKER/TSeries-11052024-1002-046',
                '/mnt/md0/BRUKER/TSeries-11052024-1002-047']

sessions_to_run = ['/mnt/md0/BRUKER/TSeries-12182024-1024-002',
                    '/mnt/md0/BRUKER/TSeries-12182024-1024-004',
                     '/mnt/md0/BRUKER/TSeries-12182024-1024-006',
                     '/mnt/md0/BRUKER/TSeries-12182024-1024-008',
                     '/mnt/md0/BRUKER/TSeries-12182024-1024-010',
                     '/mnt/md0/BRUKER/TSeries-12182024-1024-012',
                     '/mnt/md0/BRUKER/TSeries-12182024-1024-014',
                     '/mnt/md0/BRUKER/TSeries-12182024-1024-016',
                     '/mnt/md0/BRUKER/TSeries-12182024-1024-019',
                     '/mnt/md0/BRUKER/TSeries-12182024-1024-022',
                     '/mnt/md0/BRUKER/TSeries-12182024-1024-025',
                     '/mnt/md0/BRUKER/TSeries-12182024-1024-029']

########### - Done already -
#                    '/mnt/md0/BRUKER/TSeries-12172024-1304-001',
#                    '/mnt/md0/BRUKER/TSeries-12172024-1304-002',
#                    '/mnt/md0/BRUKER/TSeries-12172024-1304-003',
#                    '/mnt/md0/BRUKER/TSeries-12172024-1304-004',
#                    '/mnt/md0/BRUKER/TSeries-12172024-1304-005',
#                    '/mnt/md0/BRUKER/TSeries-12172024-1304-006',
#                    '/mnt/md0/BRUKER/TSeries-12172024-1304-007',
#                    '/mnt/md0/BRUKER/TSeries-12172024-1304-008',
#                    '/mnt/md0/BRUKER/TSeries-12172024-1304-009',
#                    '/mnt/md0/BRUKER/TSeries-12172024-1304-010',
#                    '/mnt/md0/BRUKER/TSeries-12172024-1304-011',
#                    '/mnt/md0/BRUKER/TSeries-12172024-1304-012',
#                    '/mnt/md0/BRUKER/TSeries-12172024-1304-013',
#                    '/mnt/md0/BRUKER/TSeries-12172024-1304-014',
#                    '/mnt/md0/BRUKER/TSeries-12172024-1304-015',
#                    '/mnt/md0/BRUKER/TSeries-12172024-1304-016',
#                    '/mnt/md0/BRUKER/TSeries-12172024-1304-017',
sessions_to_run = ['/mnt/md0/BRUKER/TSeries-12172024-1304-017',
                    '/mnt/md0/BRUKER/TSeries-12172024-1304-018',
                    '/mnt/md0/BRUKER/TSeries-12172024-1304-019',
                    '/mnt/md0/BRUKER/TSeries-12172024-1304-020',
                    '/mnt/md0/BRUKER/TSeries-12172024-1304-021',
                    '/mnt/md0/BRUKER/TSeries-12172024-1304-022',
                    '/mnt/md0/BRUKER/TSeries-12172024-1304-023',
                    '/mnt/md0/BRUKER/TSeries-12172024-1304-024',
                    '/mnt/md0/BRUKER/TSeries-12172024-1304-025',
                    '/mnt/md0/BRUKER/TSeries-12172024-1304-026',
                    '/mnt/md0/BRUKER/TSeries-12172024-1304-027',
                    '/mnt/md0/BRUKER/TSeries-12172024-1304-028',
                    '/mnt/md0/BRUKER/TSeries-12172024-1304-029',
                    '/mnt/md0/BRUKER/TSeries-12182024-1024-019',
                     '/mnt/md0/BRUKER/TSeries-12182024-1024-020',
                     '/mnt/md0/BRUKER/TSeries-12182024-1024-021',
                     '/mnt/md0/BRUKER/TSeries-12182024-1024-022',
                     '/mnt/md0/BRUKER/TSeries-12182024-1024-023',
                     '/mnt/md0/BRUKER/TSeries-12182024-1024-024',
                     '/mnt/md0/BRUKER/TSeries-12182024-1024-025',
                     '/mnt/md0/BRUKER/TSeries-12182024-1024-026',
                     '/mnt/md0/BRUKER/TSeries-12182024-1024-027',
                     '/mnt/md0/BRUKER/TSeries-12182024-1024-028',
                     '/mnt/md0/BRUKER/TSeries-12182024-1024-029',
                     '/mnt/md0/BRUKER/TSeries-12182024-1024-030',
                     '/mnt/md0/BRUKER/TSeries-12182024-1024-031']

if __name__ == '__main__':
    for i in range(len(sessions_to_run)):
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