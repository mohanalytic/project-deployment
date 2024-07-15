# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 12:16:52 2024

@author: USER
"""

import numpy as np
import pickle
import streamlit as st

loaded_model=pickle.load(open('C:/Users/USER/OneDrive/Desktop/machine/parkinsons_model.pkl', 'rb'))

# Creating a function for prediction
def perkison_disease_prediction(input_data):
    
    # Change the input data into numpy array
    input_data_as_numpy_array = np.asarray(input_data, dtype=float)

    # Reshape the numpy array as we are predicting for only one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    prediction = loaded_model.predict(input_data_reshaped)

    if prediction[0] == 0:
        return 'The Person does not have perkison Disease'
    else:
        return 'The Person has perkison Disease'
    
def main():
    
    
    # Giving title for our webpage
    st.title('PERKISON DISEASE PREDICTION WEB APP')
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
    if st.button('perkison Disease Test Result'):
        try:
            diagnosis = perkison_disease_prediction([MDVP_Fo_Hz, MDVP_Fhi_Hz, MDVP_Flo_Hz, MDVP_Jitter_percent, MDVP_Jitter_Abs, MDVP_RAP, MDVP_PPQ, Jitter_DDP, MDVP_Shimmer, MDVP_Shimmer_dB, Shimmer_APQ3, Shimmer_APQ5, MDVP_APQ, Shimmer_DDA, NHR, HNR, RPDE, DFA, spread1, spread2, D2, PPE])
            st.success(diagnosis)
        except ValueError:
            st.error("Please enter valid numeric values.")
        
if __name__ == '__main__':
    main()
    
    
    