import os
import sys
import time
import glob
import pathlib
import numpy as np
from datetime import datetime

sys.path.append('/home/schollab-beyonce/Documents/deepinterpolation-0.1.4')
from deepinterpolation.cli.inference import Inference

# Addresses out-of-memory problems. TF allocates all the VRAM regardless
#   of whether it needs it. Changing to memory growth only increases the
#   amount of VRAM it needs dynamically.
import tensorflow as tf


# Leak is fixed when 0.1.4 -> 0.1.5.
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

def main():
    """
    Runs DeepInterpolation inference on each parent directory.
    Needs "registered.h5" file from running motion correction.
    Needs "*transfer_model.h5" from fine-tuning DeepInterpolation.

    Check for quality motion correction before fine-tuning.
    Check for quality model fine-tuning results before completing inference.
    """
    data_paths = ['/mnt/md0/BRUKER/TSeries-11042024-1556-031']

    start_frames = np.repeat(1, len(data_paths))
    #end_frames = np.repeat(-1,  len(data_paths)) # full movies
    end_frames = np.repeat(500, len(data_paths)) # just samples

    generator_param = {}
    inference_param = {}

    generator_param["name"] = "OphysGenerator"
    generator_param["batch_size"] = 1 
    generator_param["pre_frame"] = 30
    generator_param["post_frame"] = 30
    generator_param["pre_post_omission"] = 0
    #generator_param["output_padding"] = False

    time_deltas = []

    # Handle each work directory at a time
    for i in range(0, len(data_paths)):
        ## Setting up values and checking for required files
        #   Registration results?
        data_maybe = glob.glob(os.path.join(data_paths[i], "registered.h5"))
        generator_param["data_path"] = data_maybe[0] # If this file is found, use it
        generator_param["start_frame"] = start_frames[i] 
        generator_param["end_frame"] = end_frames[i]
        
        inference_param["name"] = "core_inferrence" # *sp
        
        #   Fine-tuned DeepInterpolation transfer model?
        model_path = glob.glob(os.path.join(data_paths[i], "*transfer_model.h5"))
        if len(model_path) != 1:
            # Logging statement would go better here
            print(f"A unique transfer model was not found in directory {data_paths[i]}")
            # Try the others instead of breaking
            continue
        inference_param["model_source"] = {
            "local_path": model_path[0]
        }

        inference_param["output_file"] = os.path.join(data_paths[i], "inference_results.h5")

        args = {
            "generator_params": generator_param,
            "inference_params": inference_param,
            "output_full_args": True # Dump all inputs to local directory as well
        }

        ## Running section
        t_start = datetime.now()
        
        inference_obj = Inference(input_data=args, args=[])
        inference_obj.run()
        
        t_stop = datetime.now()
        t_delta = t_stop - t_start
        
        time_deltas.append(t_delta)
        print(f"{data_paths[i]} completed running. {t_start} to {t_stop} (duration={t_delta})")

        ## Saving results
        # Don't remove previous results in case the denoising isn't good enough.
        # 'Duplicate' date can be removed by end user after QC

        f = h5py.File(os.path.join(parent_dir, "inference_results.h5", "r"))
        numframes = f["data"].shape[0]

        # Sample directory should always be present from motion correction...
        if not os.path.isdir(os.path.join(parent_dir, "samples")):
            os.makedirs(os.path.join(parent_dir, "samples"))
        
        # Find a way to save these such that dragging into Fiji/ImageJ doesn't pull an extra dialogue box
        with tifffile.TiffWriter(os.path.join(parent_dir, "samples", "denoised.tif"), bigtiff=True, imagej=False) as tif:
            for i in range(0, min(5000, numframes)):
                curfr = f["data"][i, :, :].astype(np.int16) # f32 for some reason, doesn't need this bit depth
                tif.save(curfr)                             # Might be faster to do this in batches (1k?)
        f.close()

if __name__ == "__main__":
    main()

        


