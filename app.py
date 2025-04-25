from flask import Flask, jsonify, request, Response
import os
import sys  

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
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
            "contratara el deposito?": label
        }) 

# Enruta la funcion al endpoint /api/v1/retrain

@app.route('/api/v1/retrain', methods=['GET'])
def retrain():
    if os.path.exists("data/bank_new.csv"):
        data = pd.read_csv('data/bank_new.csv')

        data['deposit'] = data['deposit'].map({'yes': 1, 'no': 0})  # Convertir a numérico

        X_train, X_test, y_train, y_test = train_test_split(data.drop(columns=['deposit']),
                                                            data['deposit'],
                                                            test_size = 0.20,
                                                            random_state=42)

        columns_to_exclude = ["poutcome", "contact"]
        bin_features =['default', 'housing', 'loan']
        cat_features =['job', 'marital', 'education', 'month']
        num_features = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
        
        encod_bin = FunctionTransformer(cat_binary, feature_names_out='one-to-one')
        bin_pipeline = Pipeline([("Binary", encod_bin)])
        cat_pipeline = Pipeline([("OHEncoder", OneHotEncoder())])
        num_pipeline = Pipeline([('Standar_Scaler',StandardScaler())])
        preprocessor = ColumnTransformer(transformers=[("bin", bin_pipeline, bin_features),
                        ("cat", cat_pipeline, cat_features),
                        ("num", num_pipeline, num_features),
                        ("elimina","drop", columns_to_exclude)
                    ])
        
        feature_selector = SelectFromModel(RandomForestClassifier(n_estimators=100, random_state=42))

        model = Pipeline([("preprocessor", preprocessor),
                            ("feature_selector", feature_selector),
                            ("classifier", xgboost.XGBClassifier())])
        
        # pipe_xgb_param = {'Modelo__n_estimators': [10, 100, 200, 400],
        #                     'Modelo__max_depth': [1,2,4,8],
        #                     'Modelo__learning_rate': [0.1,0.2,0.5,1.0],
        #                     }

        # gs_xgb = GridSearchCV(xgb_pipeline,
        #                 pipe_xgb_param,
        #                 cv=cv,
        #                 scoring=metric,
        #                 verbose=1,
        #                 n_jobs=-1)

        model.fit(X_train, y_train)
        with open('ad_model.pkl', 'wb') as f:
            joblib.dump(model, f)

        # X_test = data.drop(columns=['deposit'])
        # y_test = data['deposit'].map({'yes': 1, 'no': 0})  # Convertir a numérico

        # return f"Model retrained. New evaluation metric: {classification_report(y_test, model.predict(X_test))}"
    
        # Pretty classification report
        y_pred = model.predict(X_test)
        report_dict = classification_report(y_test, y_pred, output_dict=True)

        # Build an HTML table
        html = """
        <h3>Model Retrained Successfully!</h3>
        <h4>Classification Report:</h4>
        <table border="1" cellpadding="5" cellspacing="0">
            <tr>
                <th>Class</th>
                <th>Precision</th>
                <th>Recall</th>
                <th>F1-Score</th>
                <th>Support</th>
            </tr>
        """

        for label, metrics in report_dict.items():
            if label not in ('accuracy',):
                html += f"""
                <tr>
                    <td>{label}</td>
                    <td>{metrics['precision']:.2f}</td>
                    <td>{metrics['recall']:.2f}</td>
                    <td>{metrics['f1-score']:.2f}</td>
                    <td>{metrics['support']}</td>
                </tr>
                """

        # Add overall accuracy
        html += f"""
            <tr>
                <td colspan="4" align="right"><strong>Accuracy</strong></td>
                <td>{report_dict['accuracy']:.2f}</td>
            </tr>
        """

        html += "</table>"

        return html

    
    
    
    
    
    else:
        return f"<h2>New data for retrain NOT FOUND. Nothing done!</h2>"


if __name__ == '__main__':
    app.run(debug=True)






# if __name__ == '__main__':
#     from os import environ
#     port = int(environ.get("PORT", 5000))
#     app.run(debug=True, host="0.0.0.0", port=port)