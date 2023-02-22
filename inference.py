import tensorflow as tf
import pandas as pd
from data_generator import DataGenerator
from autoencoder_convolucional import ConvAutoencoder
from utils import save_reconstructed_images, create_environment, create_json
from tensorflow.keras.models import load_model
import os
import numpy as np
import cv2
import sys

experiment_name = str(sys.argv[1])
print("Experiment name: ", experiment_name)
#Hyper-parameters
batch_size = 8
input_dim = (128, 128, 1)


base_folder = "/app/Otros/AC/"
run_folders = {"tsv_path": base_folder + "data/partition.csv", 
           "data_path": base_folder + "data/images/", 
           "model_path": base_folder + "data/Models/", 
           "results_path": base_folder + "data/Results/", 
           "log_filename": base_folder + "data/Results/log/CAE.log",
           "exp_name": experiment_name 
           }


df_pneumo_2d = pd.read_csv(run_folders["tsv_path"], sep=";")
df_pneumo_2d.columns = ['ImageID', 'Partition', 'Class']
data_filter = df_pneumo_2d["Partition"] == "test"
data_flow_test = DataGenerator(df_pneumo_2d[data_filter],
                               input_dim[1],
                               input_dim[0],
                               input_dim[2],
                               path_to_img=run_folders["data_path"],
                               batch_size=batch_size,
                               shuffle=False)
experimet_path = run_folders["model_path"] + experiment_name + "/"

my_CAE = ConvAutoencoder.load_model(run_folders)
example_batch = data_flow_test.__getitem__(index=0)
example_images = example_batch[0]


y_pred = my_CAE.model.predict(example_images)
save_reconstructed_images(example_images, y_pred, run_folders)
