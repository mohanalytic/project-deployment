# -*- coding: utf-8 -*-
"""
Created on Sat Jun 29 17:36:44 2024

@author: USER
"""

import numpy as np
import pickle
import streamlit as st


# loading the saved model
loaded_model=pickle.load(open('C:/Users/USER/OneDrive/Desktop/machine/trained_model.pkl', 'rb'))
# Creating a function for prediction
def heart_disease_prediction(input_data):

    # Change the input data into numpy array
    input_data_as_numpy_array = np.asarray(input_data, dtype=float)

    # Reshape the numpy array as we are predicting for only one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    prediction = loaded_model.predict(input_data_reshaped)

    if prediction[0] == 0:
        return 'The Person does not have a Heart Disease'
    else:
        return 'The Person has Heart Disease'


def main():
    

    # Giving title for our webpage
    st.title('HEART DISEASE PREDICTION WEB APP')
    
    # Getting input data from user
    age = st.text_input('Age of the Person')
    sex = st.text_input('Gender (1 = Male, 0 = Female)')
    cp = st.text_input('Chest pain type (0-3)')
    trestbps = st.text_input('Resting Blood Pressure')
    chol = st.text_input('Serum Cholesterol in mg/dl')
    fbs = st.text_input('Fasting blood Sugar > 120 mg/dl (1 = True, 0 = False)')
    restecg = st.text_input('Resting Electrocardiographic Results (0-2)')
    thalach = st.text_input('Maximum Heart Rate Achieved')
    exang = st.text_input('Exercise Induced Angina (1 = Yes, 0 = No)')
    oldpeak = st.text_input('ST depression induced by exercise relative to rest')
    slope = st.text_input('Slope of the peak exercise ST segment (0-2)')
    ca = st.text_input('Number of Major Vessels colored by fluoroscopy (0-3)')
    thal = st.text_input('Thal (0 = Normal, 1 = Fixed Defect, 2 = Reversable Defect)')
    # Code for prediction
    diagnosis = ''
    
    # Creating a button for prediction
    if st.button('Heart Disease Test Result'):
        try:
            diagnosis = heart_disease_prediction([age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal])
            st.success(diagnosis)
        except ValueError:
            st.error("Please enter valid numeric values.")
        
if __name__ == '__main__':
    main()
    

   
  
                      
   
     
        
        
    
  
  
  
     
  

  