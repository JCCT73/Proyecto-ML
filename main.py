import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import auc
import warnings
warnings.filterwarnings("ignore")

from utils.custom_functions import estandarizar_columnas

model = joblib.load(r'C:\Users\axa\THE BRIDGE_23\SEMANA 24. CORE. PROYECTO ML\81. PROYECTO ML\src\model\my_model.joblib')

def hacer_prediccion(valores_de_entrada):
    entrada_array = np.array(valores_de_entrada).reshape(1, -1)
    prediccion = model.predict(entrada_array)
    return prediccion


st.title('Predicción de compras por Internet')

administrative = st.number_input('Administrative', value=0)
administrative_duration = st.number_input('Administrative Duration', value=0.0)
informational = st.number_input('Informational', value=0)
informational_duration = st.number_input('Informational Duration', value=0.0)
product_related = st.number_input('Product Related', value=0)
product_related_duration = st.number_input('Product Related Duration', value=0.0)
bounce_rates = st.number_input('Bounce Rates', value=0.0)
exit_rates = st.number_input('Exit Rates', value=0.0)
page_values = st.number_input('Page Values', value=0.0)
special_day = st.number_input('Special Day', value=0.0)
operating_systems = st.number_input('Operating Systems', value=0)
browser = st.number_input('Browser', value=0)
region = st.number_input('Region', value=0)
traffic_type = st.number_input('Traffic Type', value=0)
weekend = st.checkbox('Weekend')
visitor_type_other = st.checkbox('Visitor Type - Other')
visitor_type_returning_visitor = st.checkbox('Visitor Type - Returning Visitor')

if st.button('Realizar Predicción'):
    user_input = pd.DataFrame({
        'Administrative': [administrative],
        'Administrative_Duration': [administrative_duration],
        'Informational': [informational],
        'Informational_Duration': [informational_duration],
        'ProductRelated': [product_related],
        'ProductRelated_Duration': [product_related_duration],
        'BounceRates': [bounce_rates],
        'ExitRates': [exit_rates],
        'PageValues': [page_values],
        'SpecialDay': [special_day],
        'OperatingSystems': [operating_systems],
        'Browser': [browser],
        'Region': [region],
        'TrafficType': [traffic_type],
        'Weekend': [weekend],
        'VisitorType_Other': [visitor_type_other],
        'VisitorType_Returning_Visitor': [visitor_type_returning_visitor]
    })

    columnas_numericas = ['Administrative', 'Administrative_Duration', 'Informational',
                          'Informational_Duration', 'ProductRelated', 'ProductRelated_Duration',
                          'BounceRates', 'ExitRates', 'PageValues', 'SpecialDay',
                          'OperatingSystems', 'Browser', 'Region', 'TrafficType']
    
   
    user_input = estandarizar_columnas(user_input, columnas_numericas)

    # Realiza la predicción
    prediction = model.predict(user_input)

    st.write(f'La predicción es: {prediction[0]}')

