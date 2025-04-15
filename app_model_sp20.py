from flask import Flask, jsonify, request
import os

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

import pickle


app = Flask(__name__)


# Enruta la landing page (endpoint /)

@app.route('/', methods=['GET'])
def hello():
    return "Bienvenido a la API del modelo 'Predicting a customer's acquisition of a banking service'"



# Enruta la landing page (endpoint /)

@app.route('/api/v1/endpoints', methods=['GET'])
def visualization():
    return "/predict = predice el modelo con los valores de los argumentos datos" \
           "/retrain = entrena el modelo con los nuevos datos" \



# Enruta la funcion al endpoint /api/v1/predict

@app.route('/api/v1/predict', methods=['GET','POST'])
def predict():
    with open('modelo_pipeline.pkl', 'rb') as f:
        model = pickle.load(f)

    age = request.args.get('age', None)
    job = request.args.get('job', None)
    marital = request.args.get('marital', None)
    education = request.args.get('education', None)
    default = request.args.get('default', None)
    balance = request.args.get('balance', None)
    housing = request.args.get('housing', None)
    loan = request.args.get('loan', None)
    day = request.args.get('day', None)
    month = request.args.get('month', None)
    duration = request.args.get('duration', None)
    campaign = request.args.get('campaign', None)
    pdays = request.args.get('pdays', None)
    previous = request.args.get('previous', None)

    print(age,job,marital,education,default,balance,housing,loan,day,month,duration,campaign,pdays,previous)

    if age is None or job is None or marital is None or education is None or default is None or balance is None or housing is None or loan is None \
          or day is None or month is None or duration is None or campaign is None or pdays is None or previous is None:
        return "Args empty, not enough data to predict"
    else:
        prediction = model.predict([[float(age),job,marital,education,default,float(balance),housing,loan,\
                                     float(day),month,float(duration),float(campaign),float(pdays),float(previous)]])
    
    return jsonify({'predictions': prediction[0]})



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
