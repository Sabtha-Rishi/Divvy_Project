import tensorflow as tf
from flask import Flask, request, render_template
from Prediction_Builder import Prediction_Data
from Model import CreateModel
import os

app = Flask(__name__)

if len(os.listdir('Models')) == 0:

    model = CreateModel().train_model("Raw Data")
    model.save('Models/model')
else: 
    model = tf.keras.models.load_model('Models/model')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''

    
    user_inputs = request.form
    pred_data = Prediction_Data().generate_pred_X(user_inputs)
    prediction = CreateModel().predict_model(model,pred_data)

    return render_template('index.html', prediction_text='It will take you approximately {} minutes to reach'.format(str(round(prediction[0]))))


if __name__ == "__main__":
    app.run(debug=True, use_reloader = False)