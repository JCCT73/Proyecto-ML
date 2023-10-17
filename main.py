import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
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

# Solicitar nuevos datos: (solitar los datos en tres bloques: )
    nuevos_datos = ()

uploaded_file = st.file_uploader("Cargar el archivo CSV", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    st.write('**Datos Cargados:**')
    st.write(data)

    from utils.custom_function import codificar_etiquetas
    df = codificar_etiquetas(df, "Weekend")

    from utils.custom_function import codificar_etiquetas
    df = codificar_etiquetas(df, "Revenue")

    from utils.custom_function import transformar_columna_month_en_dummies
    df = transformar_columna_month_en_dummies(df)

    from utils.custom_function import transformar_columna_visitor_type_en_dummies
    df = transformar_columna_visitor_type_en_dummies(df)

   



    loaded_model = joblib.load('my_model.pkl')

    
    # Solicitar nuevos datos: (solitar los datos en tres bloques: )
    nuevos_datos = ()

    

    
    prediccion = loaded_model.predict(nuevos_datos)


   



