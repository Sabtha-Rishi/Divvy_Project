import pandas as pd
import numpy as np
import  tensorflow as tf
import keras
import pickle


class NNModel:

    def __init__(self):
        pass

    def build_NN(self, hid_layers_count, neuron_per_layer, hid_activation):

        self.models_layers = []

        for layer in range(hid_layers_count):
            layer = keras.layers.Dense(units=neuron_per_layer[layer], activation=hid_activation, kernel_initializer=tf.keras.initializers.HeNormal())
            self.models_layers.append(layer)

        try:    
            self.model= keras.Sequential(self.models_layers)
            return self.model
        except Exception:
            raise Exception


    def fit_NN(self,model, x_train, y_train, x_val, y_val, epochs, steps_per_epoch, opt, loss_fn, metrics):

        self.model = model
        self.model.compile(optimizer=opt, loss=loss_fn, metrics=metrics)

        try:
            self.model.fit(
                x_train,
                y_train, 
                epochs=epochs, 
                steps_per_epoch=steps_per_epoch,
                validation_data= (x_val,y_val), 
                )

            return self.model

        except Exception:
            raise Exception

    
    def save_model(self,model,model_name,location_to_save):

        try:
            pickle.dump(model, open(location_to_save+"/"+model_name+".pkl", "wb"))
            print("Model Saved")
        except Exception:
            raise Exception






