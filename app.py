import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, BatchNormalization

application=Flask(__name__)
app=application


lr_model=pickle.load(open('models/model_lr_heart_disease_prediction.pkl', 'rb'))
standard_scaler=pickle.load(open('models/scaler_heart_disease_prediction.pkl', 'rb'))


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predictheartdisease', methods=['Get', 'Post'])
def predict_heart_disease():

    if request.method=='POST':

        msg_name = f"Hi,"
        image_input = ""
        message = f""

        try:
            #################################################
            # Get values from the form
            #################################################
            name = request.form.get('name')
            age_category = request.form.get('age')
            sex = request.form.get('sex')
            weight = float(request.form.get('weight'))
            height = float(request.form.get('height'))
            general_health = request.form.get('generalhealth')
            sleep_hours = int(request.form.get('sleephours'))
            smoker_status = request.form.get('smokerstatus')
            alcohol_drinker = request.form.get('alcoholdrinker')
            diabetes = request.form.get('diabetes')
            stroke = request.form.get('stroke')
            kidney_disease = request.form.get('kidneydisease')
            asthma = request.form.get('asthma')
            cancer = request.form.get('cancer')
            phyactive = request.form.get('phyactive')
            diffwalking = request.form.get('diffwalking')

            #################################################
            # Preparing values before scaling
            #################################################
            bmi_input = weight / (height ** 2)
            sex_input = 1 if sex == 'male' else 0
            phyactive_input = 1 if phyactive == 'yes' else 0
            stroke_input = 1 if stroke == 'yes' else 0
            asthma_input = 1 if asthma == 'yes' else 0
            cancer_input = 1 if cancer == 'yes' else 0
            kidney_disease_input = 1 if kidney_disease == 'yes' else 0
            diabetes_input = 1 if diabetes == 'yes' else 0
            diffwalking_input = 1 if diffwalking == 'yes' else 0
            
            smoker_status = [0, 0, 0]
            if general_health == 'currentsmoker':
                smoker_status[0] = 1
            elif general_health == 'formersmoker':
                smoker_status[1] = 1
            elif general_health == 'neversmoked':
                smoker_status[2] = 1

            alcohol_drinker_input = 1 if alcohol_drinker == 'yes' else 0

            general_health_input = [0, 0, 0, 0, 0]
            if general_health == 'poor':
                general_health_input[0] = 1
            elif general_health == 'fair':
                general_health_input[1] = 1
            elif general_health == 'good':
                general_health_input[2] = 1
            elif general_health == 'verygood':
                general_health_input[3] = 1
            elif general_health == 'excellent':
                general_health_input[4] = 1

            age_health_input = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            if age_category == 'age18to24':
                age_health_input[0] = 1
            elif age_category == 'age25to29':
                age_health_input[1] = 1
            elif age_category == 'age30to34':
                age_health_input[2] = 1
            elif age_category == 'age35to39':
                age_health_input[3] = 1
            elif age_category == 'age40to44':
                age_health_input[4] = 1
            elif age_category == 'age45to49':
                age_health_input[5] = 1
            elif age_category == 'age50to54':
                age_health_input[6] = 1
            elif age_category == 'age55to59':
                age_health_input[7] = 1
            elif age_category == 'age60to64':
                age_health_input[8] = 1
            elif age_category == 'age65to69':
                age_health_input[9] = 1
            elif age_category == 'age70to74':
                age_health_input[10] = 1
            elif age_category == 'age75to79':
                age_health_input[11] = 1
            elif age_category == 'age80andolder':
                age_health_input[12] = 1

            #################################################
            # Scaling the data
            #################################################
            new_data_sc = standard_scaler.transform([[
                sleep_hours,
                bmi_input,
                sex_input,
                phyactive_input,
                stroke_input,
                asthma_input,
                cancer_input,
                kidney_disease_input,
                diabetes_input,
                diffwalking_input,
                smoker_status[0], 
                smoker_status[1], 
                smoker_status[2],
                alcohol_drinker_input,
                general_health_input[0],
                general_health_input[1],
                general_health_input[2],
                general_health_input[3],
                general_health_input[4],
                age_health_input[0],
                age_health_input[1],
                age_health_input[2],
                age_health_input[3],
                age_health_input[4],
                age_health_input[5],
                age_health_input[6],
                age_health_input[7],
                age_health_input[8],
                age_health_input[9],
                age_health_input[10],
                age_health_input[11],
                age_health_input[12]
            ]])

            #################################################
            # Predicting Heart Disease
            #################################################
            # prediction_prob = round(ann_model.predict(new_data_sc), 2)
            # prediction = (prediction_prob > 0.5).astype(int)

            prediction = lr_model.predict(new_data_sc)[0]
            prediction_prob = round(lr_model.predict_proba(new_data_sc)[0][1] * 100, 2)
            
            #################################################
            # Reponse
            #################################################
            msg_name = f"Hi {name},"

            if prediction == 0:
                message = f"The probability that you'll have heart disease is " + str(prediction_prob) + "%. You are healthy!"
                image_input = "https://c.tenor.com/MzMjocR0eIEAAAAd/tenor.gif"
            else:
                message = f"The probability that you'll have heart disease is " + str(prediction_prob) + "%. You are not healthy."
                image_input = "https://clipart-library.com/2023/sick-heart-pain.gif"
                
        except Exception as err:
            print(f"[ERROR]: {err}")

        return render_template('index.html', name=msg_name, message=message, img=image_input)
    else:
        return render_template('index.html')


if __name__=="__main__":
    app.run(host="0.0.0.0", port=5000)