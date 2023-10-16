import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
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

st.title("Predicción de compra por Internet")

uploaded_file = st.file_uploader("Cargar el archivo CSV", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    st.write('**Datos Cargados:**')
    st.write(data)

    # Utilizar el LabelEncoder para transformar la columna "Weekend"
    column_to_encode = 'Weekend'
    le = LabelEncoder()
    data[column_to_encode] = le.fit_transform(data[column_to_encode])

    st.write('**Datos con la columna "Weekend" transformada:**')
    st.write(data)


    # Utilizar el LabelEncoder para transformar la columna "Revenue"
    column_to_encode = 'Revenue'
    le = LabelEncoder()
    data[column_to_encode] = le.fit_transform(data[column_to_encode])

    st.write('**Datos con la columna "Revenue" transformada:**')
    st.write(data)

    # Transformar la columna "Month" en dummies
    data = pd.get_dummies(data, columns=['Month'], prefix='Month', drop_first=True)

    # Transformar la columna "VisitorType" en dummies
    data = pd.get_dummies(data, columns=['VisitorType'], prefix='VisitorType', drop_first=True)

    st.write('**Datos transformados con variables dummies:**')

    # Escalar las características numéricas
    caracteristicas_a_escalar = ['Administrative', 'Administrative_Duration', 'Informational', 'Informational_Duration',
                                'ProductRelated', 'ProductRelated_Duration', 'BounceRates', 'ExitRates', 'PageValues', 'SpecialDay',
                                'OperatingSystems', 'Browser', 'Region', 'TrafficType', 'Weekend']

    caracteristicas_numericas = data[caracteristicas_a_escalar]
    sc = StandardScaler()
    escaladas = sc.fit_transform(caracteristicas_numericas)
    data[caracteristicas_a_escalar] = escaladas

    st.write('**Datos con características numéricas escaladas:**')
    st.write(data)


    loaded_model = joblib.load('my_model.pkl')
