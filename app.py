from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = [float(x) for x in request.form.values()]
    scaled = scaler.transform([features])
    prediction = model.predict(scaled)[0]
    result = "Safe to Drink" if prediction == 1 else "Not Safe to Drink"
    return render_template('index.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)