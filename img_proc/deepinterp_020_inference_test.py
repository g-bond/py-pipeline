import os
import sys
import glob

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
sessions_to_run = ['/mnt/md0/BRUKER/TSeries-11052024-1002-004',
                '/mnt/md0/BRUKER/TSeries-11052024-1002-006',
                '/mnt/md0/BRUKER/TSeries-11052024-1002-009',
                '/mnt/md0/BRUKER/TSeries-11052024-1002-010',
                '/mnt/md0/BRUKER/TSeries-11052024-1002-011',
                '/mnt/md0/BRUKER/TSeries-11052024-1002-015',
                '/mnt/md0/BRUKER/TSeries-11052024-1002-018',
                '/mnt/md0/BRUKER/TSeries-11052024-1002-023',
                '/mnt/md0/BRUKER/TSeries-11052024-1002-026',
                '/mnt/md0/BRUKER/TSeries-11052024-1002-029',
                '/mnt/md0/BRUKER/TSeries-11052024-1002-034',
                '/mnt/md0/BRUKER/TSeries-11052024-1002-039',
                '/mnt/md0/BRUKER/TSeries-11052024-1002-040',
                '/mnt/md0/BRUKER/TSeries-11052024-1002-044',
                '/mnt/md0/BRUKER/TSeries-11052024-1002-045']

if __name__ == '__main__':
    for i in range(len(sessions_to_run)):
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
        generator_param["end_frame"] = -1  # -1 to go until the end.
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

        # Expect this to be slow on a laptop without GPU. Inference needs
        # parallelization to be effective.
        inference_class.run()