# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 12:16:52 2024

@author: USER
"""

import numpy as np
import pickle
import streamlit as st
import os
import requests

# Check for scikit-learn installation
try:
    import sklearn
    st.write(f"Scikit-learn version: {sklearn.__version__}")
except ImportError:
    st.error("Scikit-learn is not installed. Please check your requirements.txt file.")

# Function to download files from GitHub
def download_file(url, filename):
    try:
        response = requests.get(url)
        response.raise_for_status()
        with open(filename, 'wb') as f:
            f.write(response.content)
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to download {filename} from {url}: {e}")

# URLs of the model files on GitHub
parkinsons_model_url = "https://github.com/mohanalytic/project-deployment/blob/072e96e48abdc3b471a80a84b8604f61fa50a42b/parkinsons_model.pkl"

# Download model files if they do not exist
if not os.path.exists('parkinsons_model.pkl'):
    download_file(parkinsons_model_url, 'parkinsons_model.pkl')

# Check if model files exist and load them
try:
    if os.path.exists('parkinsons_model.pkl'):
        parkinsons_model = pickle.load(open('parkinsons_model.pkl', 'rb'))
    else:
        st.error("parkinsons_model.pkl not found")
except Exception as e:
    st.error(f"Error loading model files: {e}")

# Creating a function for prediction
def parkinsons_disease_prediction(input_data):
    # Change the input data into numpy array
    input_data_as_numpy_array = np.asarray(input_data, dtype=float)

    # Reshape the numpy array as we are predicting for only one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    prediction = parkinsons_model.predict(input_data_reshaped)

    if prediction[0] == 0:
        return 'The Person does not have Parkinson\'s Disease'
    else:
        return 'The Person has Parkinson\'s Disease'

def main():
    # Giving title for our webpage
    st.title('Parkinson\'s Disease Prediction Web App')

    # Getting input data from user
    MDVP_Fo_Hz = st.text_input('MDVP_Fo(Hz)')
    MDVP_Fhi_Hz = st.text_input('MDVP_Fhi(Hz)')
    MDVP_Flo_Hz = st.text_input('MDVP_Flo(Hz)')
    MDVP_Jitter_percent = st.text_input('MDVP_Jitter(%)')
    MDVP_Jitter_Abs = st.text_input('MDVP_Jitter(Abs)')
    MDVP_RAP = st.text_input('MDVP_RAP')
    MDVP_PPQ = st.text_input('MDVP_PPQ')
    Jitter_DDP = st.text_input('Jitter_DDP')
    MDVP_Shimmer = st.text_input('MDVP_Shimmer')
    MDVP_Shimmer_dB = st.text_input('MDVP_Shimmer(dB)')
    Shimmer_APQ3 = st.text_input('Shimmer_APQ3')
    Shimmer_APQ5 = st.text_input('Shimmer_APQ5')
    MDVP_APQ = st.text_input('MDVP_APQ')
    Shimmer_DDA = st.text_input('Shimmer_DDA')
    NHR = st.text_input('NHR')
    HNR = st.text_input('HNR')
    RPDE = st.text_input('RPDE')
    DFA = st.text_input('DFA')
    spread1 = st.text_input('spread1')
    spread2 = st.text_input('spread2')
    D2 = st.text_input('D2')
    PPE = st.text_input('PPE')

    # Creating a button for prediction
    if st.button('Parkinson\'s Disease Test Result'):
        try:
            input_data = [MDVP_Fo_Hz, MDVP_Fhi_Hz, MDVP_Flo_Hz, MDVP_Jitter_percent, MDVP_Jitter_Abs, MDVP_RAP, MDVP_PPQ, Jitter_DDP, MDVP_Shimmer, MDVP_Shimmer_dB, Shimmer_APQ3, Shimmer_APQ5, MDVP_APQ, Shimmer_DDA, NHR, HNR, RPDE, DFA, spread1, spread2, D2, PPE]
            diagnosis = parkinsons_disease_prediction(input_data)
            st.success(diagnosis)
        except ValueError:
            st.error("Please enter valid numeric values.")

if __name__ == '__main__':
    main()
