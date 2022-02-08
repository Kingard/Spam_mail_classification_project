import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

# Create flask app
app = Flask(__name__)

# load the pickle model
model = pickle.load(open("model.pkl","rb"))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=["POST"])
def predict():
    input = request.form.values()
    #input = [np.array(input1)]

    prediction = model.predict(input)

    return render_template('index.html',prediction_text='This is a {} type of mail'.format(prediction))

if __name__ == "__main__":
    app.run(debug=True)