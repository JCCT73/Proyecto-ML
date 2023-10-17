--init--.py

"""
Name: Codificar etiquetas en una columna de un DataFrame.
Parametros: 
    dataframe (pandas.DataFrame): El DataFrame en el que se realizará la codificación.
    columna (str): El nombre de la columna que se codificará.
Versión: 1.0
Descripción: Esta función utiliza LabelEncoder para codificar etiquetas en la columna especificada.
Release_date: 17/OCTUBRE/2023
Update_date: 17/OCTUBRE/2023
Autor: JCCT
"""

from sklearn.preprocessing import LabelEncoder

def codificar_etiquetas(dataframe, columna):
    le = LabelEncoder()
    dataframe[columna] = le.fit_transform(dataframe[columna])
    return dataframe