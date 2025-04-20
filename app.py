from flask import Flask, jsonify, request, Response
import os
import sys  

import numpy as np
import pandas as pd
import xgboost
import joblib

# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier

def cat_binary(df):
    df["default"] = df["default"] == "yes"
    df["housing"] = df["housing"] == "yes"
    df["loan"] = df["loan"] == "yes"
    return df
sys.modules['__main__'].cat_binary = cat_binary

app = Flask(__name__)

# Enruta la landing page (endpoint /)
@app.route('/', methods=['GET'])
def hello():
    text = """
Bienvenido a la API del modelo: Predicting a customer's acquisition of a banking service.
------------------------------------------------------

Descripción:
Esta API conecta con un modelo que predice si un cliente adquirirá un servicio bancario.

Cómo usar:
- Endpoint: /api/v1/predict
- Método: POST
- Content-Type: application/json
- Envía un JSON con las siguientes características:

Ejemplo datos de entrada:
    "age": 35,
    "job": "technician",
    "marital": "married",
    "education": "tertiary",
    "default": "no",
    "balance": 1473,
    "housing": "yes",
    "loan": "no",
    "day": 12,
    "month": "may",
    "duration": 84,
    "campaign": 3,
    "pdays": -1,
    "previous": 0

Valores numéricos esperados (rangos aproximados):
- age: 18 a 93
- balance: -6847 a 81204
- day: 1 a 31
- duration: 4 a 3284
- campaign: 1 a 43
- pdays: -1 a 826
- previous: 0 a 55

Valores categóricos permitidos:
- job: retired, blue-collar, technician, management, admin., self-employed,
       student, entrepreneur, services, housemaid, unemployed, unknown
- marital: divorced, married, single
- education: secondary, tertiary, primary, unknown
- default: yes, no
- housing: yes, no
- loan: yes, no
- month: jan, feb, mar, apr, may, jun, jul, aug, sep, oct, nov, dec
"""
    return Response(text, mimetype='text/plain')


@app.route('/api/v1/predict', methods=['GET'])
def predict():
    with open('modelo_pipeline.pkl', 'rb') as f: #Load the model each time the endpoint is hit
        model = joblib.load(f) 

    # Extract query parameters
    age = request.args.get('age', type=int)
    job = request.args.get('job')
    marital = request.args.get('marital')
    education = request.args.get('education')
    default = request.args.get('default')
    balance = request.args.get('balance', type=int)
    housing = request.args.get('housing')
    loan = request.args.get('loan')
    day = request.args.get('day', type=int)
    month = request.args.get('month')
    duration = request.args.get('duration', type=int)
    campaign = request.args.get('campaign', type=int)
    pdays = request.args.get('pdays', type=int)
    previous = request.args.get('previous', type=int)

    print(age,job,marital,education,default,balance,housing,loan,day,month,duration,campaign,pdays,previous) # Imprime los valores de las variables

    # Check for any missing inputs
    required = [age, job, marital, education, default, balance, housing, loan,
                day, month, duration, campaign, pdays, previous]
    if any(field is None for field in required):
        return "Missing query parameters. All fields must be provided.", 400

    # If all required fields are present, create a DataFrame for prediction        
    input_data = pd.DataFrame([{  
    "age": age,
    "job": job,
    "marital": marital,
    "education": education,
    "default": default,
    "balance": balance,
    "housing": housing,
    "loan": loan,
    "day": day,
    "month": month,
    "duration": duration,
    "campaign": campaign,
    "pdays": pdays,
    "previous": previous
}])

    # Make prediction
    prediction = model.predict(input_data)[0]
    label = "yes" if prediction == 1 else "no"


    return jsonify({ # Return the prediction
            # "prediction": int(prediction),
            "¿contratará el depósito?": label
        }) 

if __name__ == '__main__':
    app.run(debug=True)

# if __name__ == '__main__':
#     from os import environ
#     port = int(environ.get("PORT", 5000))
#     app.run(debug=True, host="0.0.0.0", port=port)