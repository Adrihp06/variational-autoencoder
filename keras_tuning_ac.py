import tensorflow as tf
import pandas as pd
from data_generator import DataGenerator
from autoencoder_convolucional import ConvAutoencoder
from utils import save_reconstructed_images, create_environment, create_json
from tensorflow.keras.models import load_model
import os
import numpy as np
import cv2
from keras_tuner.tuners import RandomSearch
from keras_tuner.engine.hyperparameters import HyperParameters


#paths
base_folder = "/app/Otros/AC/"
#I/O paths
run_folders = {"tsv_path": base_folder + "data/partition.csv", 
           "data_path": base_folder + "data/images/", 
           "model_path": base_folder + "data/Models/", 
           "results_path": base_folder + "data/Results/", 
           "log_filename": base_folder + "data/Results/log/CAE.log"
           }
input_dim = (128, 128, 1)

# Define la función del modelo
def build_model(hp):
    # Define los hiperparámetros para el modelo
    encoder_conv_filters = [64, 64, 128, 256, 256, 512]
    encoder_conv_kernel_size = [3, 3, 3, 3, 3, 3, 5]
    encoder_conv_strides = [2, 1, 2, 1, 2, 1, 2]
    decoder_conv_t_filters = [512, 256, 64, 64, 32, 16, 1]
    decoder_conv_t_kernel_size = [5, 3, 3, 3, 3, 3, 3]
    decoder_conv_t_strides = [2, 1, 2, 1, 2, 1, 1]
    z_dim = hp.Int('z_dim', min_value=100, max_value=700, step=50)
    lr = hp.Choice('learning_rate', values=[1e-3, 5e-3, 5e-4, 5e-5, 1e-4])
    r_loss_factor = hp.Float('r_loss_factor', min_value=0.1, max_value=1.0, step=0.1)
    
    # Crear el modelo
    my_CAE = ConvAutoencoder(input_dim, 
                               encoder_conv_filters, 
                               encoder_conv_kernel_size, 
                               encoder_conv_strides, 
                               decoder_conv_t_filters,
                               decoder_conv_t_kernel_size, 
                               decoder_conv_t_strides,
                               z_dim)
    my_CAE.build(use_batch_norm=True, use_dropout=False, VCAE=True)
    my_CAE = my_CAE.compile(learning_rate=lr, r_loss_factor=r_loss_factor, VCAE=True)
    return my_CAE

# Cargar los datos
def load_data():
    # Carga de datos
    df_pneumo_2d = pd.read_csv(run_folders["tsv_path"], sep=";")
    df_pneumo_2d.columns = ['ImageID', 'Partition', 'Class']

    return df_pneumo_2d, run_folders["data_path"]
    
# Entrenar el modelo
def train_model(model, data_flow_train, data_flow_val, run_folders):
    # Entrenando el modelo
    steps_per_epoch = len(data_flow_train)
    epochs = 50
    H = model.train(data_flow_train, epochs, steps_per_epoch, data_flow_val, run_folders)
    return H

# Crea la función de llamado de Keras Tuner
# Crea la función de llamado de Keras Tuner
def run_tuning(run_folders, input_dim):

    batch_size = 6   # Definir el sintonizador
    tuner = RandomSearch(
        build_model,
        objective='val_loss',
        max_trials=20,
        executions_per_trial=1,
        allow_new_entries=True,
        directory='tuner_results',
        project_name='my_autoencoder_test3'
    )
    #best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
       # Cargando los datos
    #obtain the batch size
    df_pneumo_2d, data_path = load_data()

    # Cargando el cargador de datos de entrenamiento
    data_filter = df_pneumo_2d['Partition'] == 'train'
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
    # Crear generadores de datos de entrenamiento y validación
    # Buscar los mejores hiperparámetros

    tuner.search(
        data_flow_train,
        epochs=20,
        validation_data=data_flow_val
    )

run_tuning(run_folders, input_dim)
