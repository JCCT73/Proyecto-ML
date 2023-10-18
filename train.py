import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import joblib
import mlflow.sklearn
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv( r"C:\Users\axa\THE BRIDGE_23\SEMANA 24. CORE. PROYECTO ML\81. PROYECTO ML\src\data\raw\online_shoppers_intention.csv", sep=",")

df1 = df.drop("Month", axis=1)
df1 = pd.get_dummies(df1, columns=['VisitorType'], prefix='VisitorType', drop_first=True)
le = LabelEncoder()
df1['Weekend'] = le.fit_transform(df1['Weekend'])
df1['Revenue'] = le.fit_transform(df1['Revenue'])
variables_numericas = ['Administrative', 'Administrative_Duration', 'Informational', 'Informational_Duration','ProductRelated', 'ProductRelated_Duration', 
                       'BounceRates','ExitRates','PageValues', 'SpecialDay', 'OperatingSystems', 'Browser', 'Region', 'TrafficType']
sc = StandardScaler()
df1[variables_numericas] = sc.fit_transform(df1[variables_numericas])

ruta_guardado = r"C:\Users\axa\THE BRIDGE_23\SEMANA 24. CORE. PROYECTO ML\81. PROYECTO ML\src\data\processed\df1.csv"
df1.to_csv(ruta_guardado, index=False)
print(f"El archivo CSV se ha guardado en: {ruta_guardado}")

X = df1.drop('Revenue', axis=1)
y = df1['Revenue']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, shuffle=True)


modeloRFC = RandomForestClassifier(n_estimators=100, class_weight='balanced', max_depth=10, random_state=1)
modeloRFC.fit(X_train, y_train)

y_pred = modeloRFC.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, modeloRFC.predict_proba(X_test)[:, 1])
confusionRFC= confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = confusionRFC.ravel()
specificity = tn / (tn + fp)

print(f'Accuracy modeloRFC: {accuracy}')
print(f'Precision modeloRFC: {precision}')
print(f'Recall modeloRFC: {recall}')
print(f'F1-Score modeloRFC: {f1}')
print(f'AUC-ROC modeloRFC: {roc_auc}')
print('Matriz de Confusión modeloRFC:')
print(confusionRFC)
print("Especificidad modeloRFC:", specificity)

new_model = modeloRFC

with mlflow.start_run():
    mlflow.set_experiment("Customer_Classification")
    mlflow.set_tag("model_type", "RandomForest")
    mlflow.sklearn.log_model(new_model, "Random_forest_model")

    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("auc_roc", roc_auc)
   

mlflow.end_run()

ruta_guardado = r"C:\Users\axa\THE BRIDGE_23\SEMANA 24. CORE. PROYECTO ML\81. PROYECTO ML\src\model/"
nombre_archivo = 'new_model.joblib'
ruta_completa = ruta_guardado + nombre_archivo
joblib.dump(new_model, ruta_completa)

ruta_guardado = r"C:\Users\axa\THE BRIDGE_23\SEMANA 24. CORE. PROYECTO ML\81. PROYECTO ML\src\model/"
