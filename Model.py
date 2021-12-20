import numpy as np
from Data_Preparation import Feature_Engineering
from NN_Builder import NNModel
import tensorflow as tf
import pickle
import keras

class CreateModel:

    def __init__(self):
        pass

    def train_model(self,file_path):

        self.x_train, self.y_train, self.x_val, self.y_val, self.x_test, self.y_test, self.scaler = Feature_Engineering(file_path).prepare_data()
        self.model = NNModel().build_NN(9, [256,256,192,192,100,100,100,100,100], tf.keras.layers.LeakyReLU(alpha=0.01))
        self.model = NNModel().fit_NN(self.model,self.x_train, self.y_train, self.x_val, self.y_val,10, 128, 'adam', 'mean_squared_error', ['mean_squared_error'])
        pickle.dump(self.scaler, open("Dictionary Files\scaler.pkl", "wb"))

        return self.model
    


    def predict_model(self,model,X):
        
        self.scaler = pickle.load(open("Dictionary Files\scaler.pkl", "rb"))
        self.X = self.scaler.transform(X.values)
        self.prediction_array = model.predict(self.X)
        self.final_prediction = []
        for pred in self.prediction_array:
            pred = np.mean(pred)
            self.final_prediction.append(pred)

        return self.final_prediction
        

    