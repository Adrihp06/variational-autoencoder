#Imports necesarios
import tensorflow as tf
import pandas as pd
from data_generator import DataGenerator
from autoencoder_convolucional import ConvAutoencoder
from utils import save_reconstructed_images, create_environment, create_json
from tensorflow.keras.models import load_model
import os
import numpy as np
import cv2


#Hyper-parameters
input_dim = (128, 128, 1)
encoder_conv_filters = [32, 64, 64, 64]
encoder_conv_kernel_size = [3, 3, 3, 3]
encoder_conv_strides = [2, 2, 2, 2]
decoder_conv_t_filters = [64, 64, 32, 1]
decoder_conv_t_kernel_size = [3, 3, 3, 3]
decoder_conv_t_strides = [2, 2, 2, 2]
sess = tf.function()
z_dim = 100
lr = 0.00005
batch_size = 10
epochs = 80
r_loss_factor = 0.00001
VCAE = True
base_folder = "PATH_TO_PROJECT"
#I/O paths
run_folders = {"tsv_path": base_folder + "data/partition.csv", 
           "data_path": base_folder + "data/images/", 
           "model_path": base_folder + "data/Models/", 
           "results_path": base_folder + "data/Results/", 
           "log_filename": base_folder + "data/Results/log/CAE.log"
           }
# Creando el entramado de directorios

create_environment(run_folders)
hyperparameters = {"input_dim": input_dim, 
                   "encoder_conv_filters": encoder_conv_filters, 
                   "encoder_conv_kernel_size":encoder_conv_kernel_size, 
                   "encoder_conv_strides":encoder_conv_strides, 
                   "decoder_conv_t_filters":decoder_conv_t_filters, 
                   "decoder_conv_t_kernel_size":decoder_conv_t_kernel_size, 
                   "decoder_conv_t_strides":decoder_conv_t_strides, 
                   "z_dim":z_dim, 
                   "lr":lr, 
                   "batch_size":batch_size, 
    "epochs":epochs, 
                   "r_loss_factor":r_loss_factor, 
                   "opt":"Adam", 
                   "loss_function":"mse",
                   "data_path": run_folders["data_path"]
                   }
# Fase de entrenamiento
create_json(hyperparameters, run_folders)
# Carga de datos
df_pneumo_2d = pd.read_csv(run_folders["tsv_path"], sep=";")
df_pneumo_2d.columns = ['ImageID', 'Partition', 'Class']
data_filter = df_pneumo_2d['Partition']=='train'
# Cargador de datos training
data_flow_train = DataGenerator(df_pneumo_2d[data_filter], 
                                  input_dim[1], 
                                  input_dim[0], 
                                  input_dim[2], 
                                  batch_size=batch_size, 
                                  path_to_img=run_folders["data_path"],
                                  shuffle = True)
  # Cargador de datos validation
data_filter = df_pneumo_2d['Partition']=='val'
data_flow_val = DataGenerator(df_pneumo_2d[data_filter], 
                                  input_dim[1], 
                                  input_dim[0], 
                                  input_dim[2], 
                                  batch_size=batch_size, 
                                  path_to_img=run_folders["data_path"],
                                  )
  # Creando el CAE
my_CAE = ConvAutoencoder(input_dim, 
                           encoder_conv_filters, 
                           encoder_conv_kernel_size, 
                           encoder_conv_strides, 
                           decoder_conv_t_filters,
                           decoder_conv_t_kernel_size, 
                           decoder_conv_t_strides,
                           z_dim)
  
  # Construyendo arquitectura
my_CAE.build(use_batch_norm=True, use_dropout=False, VCAE=VCAE)
print(my_CAE.model.summary())

  # Compilando el CAE
vae_loss = my_CAE.compile(learning_rate=lr, r_loss_factor=r_loss_factor, VCAE=VCAE)


steps_per_epoch = len(data_flow_train)
H = my_CAE.train(data_flow_train, epochs, steps_per_epoch, data_flow_val, run_folders)


