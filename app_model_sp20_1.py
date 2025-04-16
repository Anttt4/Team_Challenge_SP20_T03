from flask import Flask, jsonify, request
import os

import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier



app = Flask(__name__)


# Enruta la landing page (endpoint /)

@app.route('/', methods=['GET'])
def hello():
    return "Bienvenido a la API del modelo: Predicting a customer's acquisition of a banking service"



# Enruta la landing page (endpoint /)

@app.route('/api/v1/endpoints', methods=['GET'])
def visualization():
    return '/predict = predice el modelo con los valores de los argumentos dados \n' \
           '/retrain = entrena el modelo con los nuevos datos'



# Enruta la funcion al endpoint /api/v1/predict

@app.route('/api/v1/predict', methods=['GET','POST'])
def predict():
    
    def cat_binary(val):
        if val in ['yes', 'true']:
            return 1
        else:
            return 0
    
    with open('modelo_pipeline.pkl', 'rb') as f:
        model = pickle.load(f)

    try:

        params = {
            'age': float(request.args.get('age', None)),
            'job': request.args.get('job', None),
            'marital': request.args.get('marital', None),
            'education': request.args.get('education', None),
            'default': request.args.get('default', None),
            'balance': float(request.args.get('balance', None)),
            'housing': request.args.get('housing', None),
            'loan': request.args.get('loan', None),
            'day': int(request.args.get('day', None)),
            'month': request.args.get('month', None),
            'duration': float(request.args.get('duration', None)),
            'campaign': float(request.args.get('campaign', None)),
            'pdays': float(request.args.get('pdays', None)),
            'previous': float(request.args.get('previous', None))
        }

        for param, value in params.items():
                if value is None:
                    return jsonify({'error': f"Falta el parámetro '{param}'"}), 400

        data = [
            [
                params['age'], 
                params['job'], 
                params['marital'], 
                params['education'], 
                params['default'], 
                params['balance'], 
                params['housing'], 
                params['loan'], 
                params['day'], 
                params['month'], 
                params['duration'], 
                params['campaign'], 
                params['pdays'], 
                params['previous']
            ]
        ]

        prediction = model.predict(data)
    
        return jsonify({'predictions': prediction[0]})

    except Exception as e:
        return jsonify({'error': f'Ha ocurrido un error: {str(e)}'}), 500


# # Enruta la funcion al endpoint /api/v1/retrain

# @app.route('/api/v1/retrain', methods=['GET'])
# def retrain():
#     if os.path.exists("data/bank_new.csv"):
#         data = pd.read_csv('data/bank_new.csv')

#         X_train, X_test, y_train, y_test = train_test_split(data.drop(columns=['sales']),
#                                                             data['sales'],
#                                                             test_size = 0.20,
#                                                             random_state=42)

#         model = Lasso(alpha=6000)
#         model.fit(X_train, y_train)
#         rmse = np.sqrt(mean_squared_error(y_test, model.predict(X_test)))
#         mape = mean_absolute_percentage_error(y_test, model.predict(X_test))
#         model.fit(data.drop(columns=['sales']), data['sales'])
#         with open('ad_model.pkl', 'wb') as f:
#             pickle.dump(model, f)

#         return f"Model retrained. New evaluation metric RMSE: {str(rmse)}, MAPE: {str(mape)}"
#     else:
#         return f"<h2>New data for retrain NOT FOUND. Nothing done!</h2>"


if __name__ == '__main__':
    app.run(debug=True)
