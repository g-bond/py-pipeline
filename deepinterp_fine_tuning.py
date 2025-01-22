import os
import sys
import code
import glob
import h5py

sys.path.append('/home/schollab-dion/Documents/py-pipeline/deepinterpolation-0.2.0')
from deepinterpolation.generator_collection import OphysGenerator
from deepinterpolation.trainor_collection import transfer_trainer

import datetime
import pathlib

import tensorflow as tf

# TF will allocate all VRAM on GPU and crash due to OOM if this is not done. - gnb ~Dec24
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            #tf.config.experimental.set_memory_growth(gpu, True)
            tf.config.set_logical_device_configuration(
                gpu,
                [tf.config.LogicalDeviceConfiguration(memory_limit=1024 * 22)]
            )
    except RuntimeError as e:
        print(e)






sessions_to_run = ["/mnt/md0/BRUKER/TSeries-11042024-1556-031"]

if __name__ == '__main__':
    for i in range(len(sessions_to_run)):
        # Check that data is present here
        parent_dir = sessions_to_run[i]
        data_file_maybe = glob.glob(os.path.join(parent_dir, "registered.h5"))
        data_file = data_file_maybe[0]
         
        f = h5py.File(data_file, "a")
        rename = False
        try:
            total_frames = f["mov"].shape=[0]
            rename = True
        except:
            total_frames = f["data"].shape[0]
        if total_frames < 20000:
            num_training_samples = total_frames
        else:
            num_training_samples = 20000
        
        if rename:
            f["data"] = f["mov"]
            del f["mov"]
        f.close()

        # Recordkeeping
        now = datetime.datetime.now()
        run_uid = now.strftime("%Y_%m_%d_%H_%M")

        # Initialize meta-parameters objects
        training_param = {}
        generator_param = {}
        network_param = {}
        generator_test_param = {}

        # An epoch is defined as the number of batches pulled from the dataset.
        # Because our datasets are VERY large. Often, we cannot
        # go through the entirity of the data so we define an epoch slightly
        # differently than is usual.
        steps_per_epoch = 200

        # Those are parameters used for the Validation test generator.
        # Here the test is done on the beginning of the data but
        # this can be a separate file
        generator_test_param[
            "pre_post_frame"
        ] = 30  # Number of frame provided before and after the predicted frame
        
        generator_test_param["train_path"] = data_file
        #generator_test_param["train_path"] = os.path.join(
        #    pathlib.Path(__file__).parent.absolute(),
        #    "deepinterpolation-0.2.0",
        #    "sample_data",
        #    "ephys_tiny_continuous.dat2",
        #)
        generator_test_param["batch_size"] = 10
        generator_test_param["start_frame"] = 0
        generator_test_param["end_frame"] = 20000
        generator_test_param["total_samples"] = 500
        generator_test_param[
            "pre_post_omission"
        ] = 0  # Number of frame omitted before and after the predicted frame
        generator_test_param[
            "steps_per_epoch"
        ] = -1
        # No step necessary for testing as epochs are not relevant.
        # -1 deactivate it.

        # Those are parameters used for the main data generator
        generator_param["steps_per_epoch"] = steps_per_epoch
        generator_param["pre_post_frame"] = 30
        generator_param["train_path"] = data_file
        #generator_param["train_path"] = os.path.join(
        #    pathlib.Path(__file__).parent.absolute(),
        #    "deepinterpolation-0.2.0",
        #    "sample_data",
        #    "ephys_tiny_continuous.dat2",
        #)
        generator_param["batch_size"] = 10
        generator_param["start_frame"] = 0
        generator_param["end_frame"] = total_frames if total_frames < 20000 else 20000
        generator_param["total_samples"] = num_training_samples
        generator_param["pre_post_omission"] = 0

        # Those are parameters used for the training process
        training_param["run_uid"] = run_uid

        # Change this path to any model you wish to improve
        #training_param[
        #    "model_path"
        #] = r"./sample_data/2020_02_29_15_28_unet_single_ephys_" \
        #    + r"1024_mean_squared_error-1050.h5"
        #training_param["model_path"] = r"2020_02_29_15_28_unet_single_ephys_1024_mean_squared_error-1050.h5"
        training_param["model_path"] = "2019_09_11_23_32_unet_single_1024_mean_absolute_error_Ai93-0450.h5"
        training_param["batch_size"] = generator_test_param["batch_size"]
        training_param["steps_per_epoch"] = steps_per_epoch
        training_param[
            "period_save"
        ] = 5
        # network model is potentially saved during training
        # between a regular nb epochs
        training_param["nb_gpus"] = 1
        training_param["apply_learning_decay"] = 0
        
        training_param[
            "nb_times_through_data"
        ] = 2
        # if you want to cycle through the entire data.
        # Too many iterations will cause noise overfitting
        
        
        training_param["learning_rate"] = 0.0001
        training_param["pre_post_frame"] = generator_test_param["pre_post_frame"]
        training_param["loss"] = "mean_squared_error" # or "mean_absolute_error"
        
        training_param[
            "nb_workers"
        ] = 16  # this is to enable multiple threads for data generator loading.
        
        training_param["caching_validation"] = False # self.cache_validation() fails w/out this - gnb Dec'24

        training_param["model_string"] = (
            "transfer" + "_" + training_param["loss"]
            + "_" + training_param["run_uid"]
        )

        # Where do you store ongoing training progress
        jobdir = os.path.join(
            ".", training_param["model_string"] + "_" + run_uid,
        )
        training_param["output_dir"] = jobdir

        try:
            os.mkdir(jobdir)
        except Exception:
            print("folder already exists")

        # We find the generator obj in the collection using the json file
        #train_generator = EphysGenerator(generator_param)
        #test_generator = EphysGenerator(generator_test_param)
        train_generator = OphysGenerator(generator_param)
        test_generator = OphysGenerator(generator_test_param)

        # We build the training object.
        training_class = transfer_trainer(
            train_generator, test_generator, training_param
        )

        # Start training. This can take very long time.
        training_class.run()

        # Finalize and save output of the training.
        training_class.finalize()
    # One completed
