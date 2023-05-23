# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, BatchNormalization, LeakyReLU, Flatten, Lambda, Dense, Activation, Dropout, Reshape, Input, Layer
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, CSVLogger, EarlyStopping
from tensorflow.keras.utils import plot_model
from tensorflow.keras.losses import Loss, mse
from tensorflow.keras import regularizers
from utils import learning_curve_plot
from tensorflow.keras import backend as K
import numpy as np
import os
import pickle

class ConvAutoencoder:

    def __init__(self, 
               input_dim,
               encoder_conv_filters,
               encoder_conv_kernels,
               encoder_conv_strides,
               decoder_conv_filters,
               decoder_conv_kernels,
               decoder_conv_strides,
               z_dim):
        super(ConvAutoencoder, self).__init__()
        self.input_dim = input_dim
        self.encoder_conv_filters = encoder_conv_filters
        self.encoder_conv_kernels = encoder_conv_kernels
        self.encoder_conv_strides = encoder_conv_strides
        self.decoder_conv_filters = decoder_conv_filters
        self.decoder_conv_kernels = decoder_conv_kernels
        self.decoder_conv_strides = decoder_conv_strides
        self.z_dim = z_dim

    def build(self, use_batch_norm=False, use_dropout=False, VCAE=False):
        #Defiminos el encoder

        encoder_input = Input(shape=self.input_dim, name='encoder_input')
        x = encoder_input
        for i in range (len(self.encoder_conv_filters)):
            conv_layer = Conv2D(filters=self.encoder_conv_filters[i], 
                            kernel_size=self.encoder_conv_kernels[i],
                            strides=self.encoder_conv_strides[i],
                            padding='same', name='encoder_conv'+str(i))
            x = conv_layer(x)
            if use_batch_norm:
                x = BatchNormalization()(x)

            x = LeakyReLU(alpha = 0.2)(x)

            if use_dropout:
                x = Dropout(0.25)(x)

            #Almacenamos la dimensionalidad, obviamo sla primera que indica el tamano del batch
        shape_before_flattening = K.int_shape(x)[1:]
        x = Flatten()(x)
        encoder_output = Dense(self.z_dim, name='encoder_output')(x)
       

      #La versión variacional pretende regularizar el espacio latente. a partir de distribuciones de probabilidad multivariante
        if VCAE:
        #Se define por un vector de medias y un vector de desviaciones
            self.mu = Dense(self.z_dim, name='mu')(x)
            self.log_var = Dense(self.z_dim, name='log_var')(x)
        #Tenemos que definir al funcion sampling
            def sampling(args):
                mu, log_var = args
                #Definimos la ecuación de la gaussiana
                epsilon = K.random_normal(shape = K.shape(mu), mean = 0., stddev=0.1)
                return mu + K.exp(log_var/2)*epsilon
            encoder_output = Lambda(sampling, name='encoder_output')([self.mu, self.log_var]) 
        #self.encoder = Model(encoder_input, atencion)
        self.encoder = Model(encoder_input, encoder_output)
        #Definimos el decoder
        decoder_input = Input(shape=(self.z_dim,), name='decoder_input')
        #Tenemos que pasar de la dimensionalidad del espacio latente a la dimensionalidad de antes del flattening
        x = Dense(np.prod(shape_before_flattening))(decoder_input)
        x = Reshape(shape_before_flattening)(x)

        for i in range (len(self.decoder_conv_filters)):
            conv_layer_t = Conv2DTranspose(filters=self.decoder_conv_filters[i], 
                                     kernel_size=self.decoder_conv_kernels[i],
                                     strides=self.decoder_conv_strides[i],
                                     padding='same', kernel_regularizer = regularizers.l1(0.00001), name='decoder_conv'+str(i))
        
            x = conv_layer_t(x)

            if i < len(self.decoder_conv_filters) - 1:
                x = LeakyReLU(alpha=0.2)(x)
            else:
                #Si estamos en la ultima capa devolvemos a valores entre 0 y 1
                x = Activation('sigmoid')(x)


        decoder_output = x
        self.decoder = Model(decoder_input, decoder_output)

        #Creando el autoencoder convolucional

        autoencoder_input = encoder_input
        autoencoder_output = self.decoder(encoder_output)
        autoencoder = Model(autoencoder_input, autoencoder_output)
        self.model = autoencoder
        

    def compile(self, learning_rate=0.005, r_loss_factor=0.4, VCAE=False):
        self.learning_rate = learning_rate
        self.r_loss_factor = r_loss_factor
        optimizer = Adam(learning_rate = self.learning_rate)
        #Definimos las pérdidas en caso del variacional y para ello tenemos que determinar una pérdida asociada al mse y otra al espacio latente
        #Debido a esto ponderamos la pérdida asociado al espacio latente como un hiperparámetro mas
        if VCAE:
            reconstruction_loss = K.mean(K.square(self.model.input - self.model.output), axis=[1, 2, 3])
            kl_loss = -0.5 * K.sum(1 + K.log(K.square(self.log_var)) - K.square(self.mu) - K.square(self.log_var), axis=-1)
            vae_loss = K.mean(reconstruction_loss + kl_loss*self.r_loss_factor)
            
            self.model.add_loss(vae_loss)
            self.model.compile(optimizer=optimizer)
        else:
            self.model.compile(optimizer=optimizer, loss='mse')
        return self.model


    def train(self, data_flow, epochs, steps_per_epoch, data_flow_val, run_folders):
        csv_logger = CSVLogger(run_folders["log_filename"])
        checkpoint = ModelCheckpoint(os.path.join(run_folders["model_path"], run_folders["exp_name"]+'/weights/CAE_weights.h5'), save_weights_only=True, verbose=1)
        lr_sched = self.step_decay_schedule(initial_lr=self.learning_rate, decay_factor=1, step_size=1)
        early_stop = EarlyStopping(monitor='val_loss', patience=25)
        callbacks_list = [csv_logger, checkpoint, lr_sched, early_stop]
        print("[INFO]: Training")
        history = self.model.fit(data_flow, 
                                 epochs=epochs, 
                                 validation_data=data_flow_val, 
                                 steps_per_epoch=steps_per_epoch, 
                                 callbacks=callbacks_list)
        self.save_model(run_folders, history)

    #Reescribimos el metodo step_decay_schedule de Keras para poder aplicar nuestros parametros.
    def step_decay_schedule(self, initial_lr , decay_factor=0.5, step_size=1):
        def schedule(epoch):
            new_lr = initial_lr * (decay_factor ** np.floor(epoch/step_size))
            return new_lr
        return LearningRateScheduler(schedule)

    #Metodos de carga y almacenamiento del modelo
    def save_model(self, run_folders, history):
        with open(os.path.join(run_folders["model_path"], run_folders["exp_name"]+'/CAE_model.pkl'), 'wb') as f:
            pickle.dump([self.input_dim
                    , self.encoder_conv_filters
                    , self.encoder_conv_kernels
                    , self.encoder_conv_strides
                    , self.decoder_conv_filters
                    , self.decoder_conv_kernels
                    , self.decoder_conv_strides
                    , self.z_dim], f)
        self.plot_model(run_folders)
        learning_curve_plot(history, run_folders)

    @staticmethod
    def load_model(run_folders, VCAE=False):
        with open(os.path.join(run_folders["model_path"], run_folders["exp_name"]+'/CAE_model.pkl'), 'rb') as f:
            params = pickle.load(f)

        my_CAE = ConvAutoencoder(*params)
        my_CAE.build(use_batch_norm=True, use_dropout=True, VCAE=VCAE)
        my_CAE.model.load_weights(os.path.join(run_folders["model_path"], run_folders["exp_name"]+'/weights/CAE_weights.h5'))
        return my_CAE
    #Almacener una imagen con la arquitectura de la red
    def plot_model(self, run_folders):
        plot_model(self.model, to_file=os.path.join(run_folders["model_path"], run_folders["exp_name"]+'/viz/model_autoencoder.png'), show_shapes=True, show_layer_names=True)
        plot_model(self.encoder, to_file=os.path.join(run_folders["model_path"], run_folders["exp_name"]+'/viz/model_encoder.png'), show_shapes=True, show_layer_names=True)
        plot_model(self.decoder, to_file=os.path.join(run_folders["model_path"], run_folders["exp_name"]+'/viz/model_decoder.png'), show_shapes=True, show_layer_names=True)
