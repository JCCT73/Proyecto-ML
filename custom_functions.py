import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import pandas as pd
import numpy as np
import joblib
import pickle
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV


def eliminar_columna(dataframe, columna):
    """
    Elimina una columna específica de un DataFrame.

    Parameters:
    dataframe (pandas.DataFrame): El DataFrame en el que se realizará la transformación.
    columna (str): El nombre de la columna que se eliminará.

    Returns:
    pandas.DataFrame: El DataFrame con la columna eliminada.

    Versión: 1.0
    Descripción: Esta función elimina la columna especificada del DataFrame de entrada.
    Release_date: 17/OCTUBRE/2023
    Update_date: 17/OCTUBRE/2023
    Autor: JCCT
    """
    return dataframe.drop(columna, axis=1)




def transformar_columna_visitor_type(df):
    """
    Transforma la columna 'VisitorType' a variables ficticias (dummies) y elimina la primera columna.

    Parameters:
    df (pandas.DataFrame): El DataFrame que contiene la columna 'VisitorType' a transformar.

    Returns:
    pandas.DataFrame: El DataFrame con las variables ficticias de 'VisitorType' y la primera columna eliminada.

    Versión: 1.0
    Descripción: Esta función toma un DataFrame, transforma la columna 'VisitorType' en variables ficticias
    y elimina la primera columna ficticia para evitar multicolinealidad.
    Release_date: 17/OCTUBRE/2023
    Update_date: 17/OCTUBRE/2023
    Autor: JCCT
    """
    df_transformado = pd.get_dummies(df, columns=['VisitorType'], prefix='VisitorType', drop_first=True)
    return df_transformado

# Ejemplo de uso:
data = {'VisitorType': ['Returning_Visitor', 'New_Visitor', 'Returning_Visitor']}
df = pd.DataFrame(data)
df_transformado = transformar_columna_visitor_type(df)
print(df_transformado)




def codificar_etiquetas(dataframe, columna):
    """
    Codificar etiquetas en una columna de un DataFrame.

    Parametros: 
        dataframe (pandas.DataFrame): El DataFrame en el que se realizará la codificación.
        columna (str): El nombre de la columna que se codificará.

    Returns:
        pandas.DataFrame: El DataFrame con la columnas codificada.

    Versión: 1.0
    Descripción: Esta función utiliza LabelEncoder para codificar etiquetas en la columna especificada.
    Release_date: 17/OCTUBRE/2023
    Update_date: 17/OCTUBRE/2023
    Autor: JCCT
    """

    le = LabelEncoder()
    dataframe[columna] = le.fit_transform(dataframe[columna])
    return dataframe

 


from sklearn.preprocessing import StandardScaler

def estandarizar_columnas(dataframe, columnas):
    """
    Estandariza las columnas especificadas en un DataFrame.

    Parameters:
    dataframe (pandas.DataFrame): El DataFrame en el que se realizará la estandarización.
    columnas (list): Lista de nombres de las columnas a estandarizar.

    Returns:
    pandas.DataFrame: El DataFrame con las columnas especificadas estandarizadas.

    Versión: 1.0
    Descripción: Esta función utiliza StandardScaler para estandarizar las columnas especificadas.
    Release_date: 17/OCTUBRE/2023
    Update_date: 17/OCTUBRE/2023
    Autor: JCCT
    """
    sc = StandardScaler()
    dataframe[columnas] = sc.fit_transform(dataframe[columnas])
    return dataframe




from sklearn.model_selection import train_test_split

def dividir_datos(dataframe, target_column, test_size=0.3, random_state=None, shuffle=True):
    """
    Dividir un DataFrame en conjuntos de entrenamiento y prueba.

    Parameters:
        dataframe (pandas.DataFrame): El DataFrame que se dividirá.
        target_column (str): El nombre de la columna objetivo.
        test_size (float): El tamaño del conjunto de prueba (por defecto, 0.3).
        random_state (int): La semilla para la generación de números aleatorios (opcional).
        shuffle (bool): Indica si los datos deben ser reordenados (por defecto, True).

    Returns:
        pandas.DataFrame: Conjunto de entrenamiento (X_train).
        pandas.DataFrame: Conjunto de prueba (X_test).
        pandas.Series: Etiquetas de entrenamiento (y_train).
        pandas.Series: Etiquetas de prueba (y_test).

    Versión: 1.0
    Descripción: Esta función toma un DataFrame y divide los datos en conjuntos de entrenamiento y prueba. 
    También separa las etiquetas del objetivo.

    Fecha de lanzamiento: 17/OCTUBRE/2023
    Fecha de actualización: 17/OCTUBRE/2023
    Autor: JCCT
    """
    X = dataframe.drop(target_column, axis=1)
    y = dataframe[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, shuffle=shuffle)
    return X_train, X_test, y_train, y_test



def predecir_modelo(modelo, datos_prueba):
    """
    Realiza predicciones utilizando un modelo de Machine Learning.

    Parámetros:
        modelo: El modelo de Machine Learning previamente entrenado.
        datos_prueba: El conjunto de datos de prueba en el que se realizarán las predicciones.

    Retorna:
        numpy.ndarray: Predicciones realizadas por el modelo.

    Versión: 1.0
    Descripción: Esta función toma un modelo de Machine Learning y un conjunto de datos de prueba
    para realizar predicciones utilizando el modelo.

    Fecha de lanzamiento: 17/OCTUBRE/2023
    Fecha de actualización: 17/OCTUBRE/2023
    Autor: JCCT
    """
    predicciones = modelo.predict(datos_prueba)
    return predicciones



from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

def calcular_metricas(modelo, X_test, y_test):
    """
    Calcula múltiples métricas de evaluación de un modelo de Machine Learning.

    Parámetros:
        modelo: El modelo de Machine Learning previamente entrenado.
        X_test: El conjunto de datos de prueba.
        y_test: Las etiquetas verdaderas correspondientes al conjunto de datos de prueba.

    Retorna:
        dict: Un diccionario que contiene las siguientes métricas:
            - "Accuracy"
            - "Precision"
            - "Recall"
            - "F1-score"
            - "AUC-ROC"
            - "Matriz de Confusión"
            - "Especificidad"

    Versión: 1.0
    Descripción: Esta función toma un modelo de Machine Learning, un conjunto de datos de prueba y las etiquetas verdaderas
    para calcular y devolver múltiples métricas de evaluación.

    Fecha de lanzamiento: 17/OCTUBRE/2023
    Fecha de actualización: 17/OCTUBRE/2023
    Autor: JCCT
    """
    y_pred = modelo.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, modelo.predict_proba(X_test)[:, 1])
    conf_matrix = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = conf_matrix.ravel()
    specificity = tn / (tn + fp)

    metricas = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1,
        "AUC-ROC": roc_auc,
        "Matriz de Confusión": conf_matrix,
        "Especificidad": specificity
    }

    return metricas






