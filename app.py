import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, 'Zomato_df.csv')

df = pd.read_csv(csv_path)


df = pd.read_csv('Zomato_df.csv')

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    features = [int(x) for x in request.form.values()]
    final_features = [np.array(features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 1)

    return render_template('index.html', prediction_text='Your Rating is: {}'.format(output))


if __name__ == "__main__":
    app.run(debug=True)