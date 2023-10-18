import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.ensemble import BalancedRandomForestClassifier
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


modeloBRF = BalancedRandomForestClassifier(n_estimators=100, max_depth=5, class_weight= "balanced", random_state=1)
modeloBRF.fit(X_train, y_train)

y_pred_BRF = modeloBRF.predict(X_test)

accuracy_BRF = accuracy_score(y_test, y_pred_BRF)
precision_BRF = precision_score(y_test, y_pred_BRF)
recall_BRF = recall_score(y_test, y_pred_BRF)
f1_BRF = f1_score(y_test, y_pred_BRF)
roc_auc_BRF = roc_auc_score(y_test, modeloBRF.predict_proba(X_test)[:, 1])
conf_matrix_BRF = confusion_matrix(y_test, y_pred_BRF)
tn, fp, fn, tp = conf_matrix_BRF.ravel()
specificity = tn / (tn + fp)

print("Accuracy modeloBRF:", accuracy_BRF)
print("Precision modeloBRF:", precision_BRF)
print("Recall modeloBRF:", recall_BRF)
print("F1-score modeloBRF:", f1_BRF)
print("AUC-ROC modeloBRF:", roc_auc_BRF)
print("Matriz de Confusión modeloBRF:")
print(conf_matrix_BRF)
print("Especificidad modeloBRF:", specificity)

new_model = modeloBRF

with mlflow.start_run():
    mlflow.set_experiment("Customer_Classification")
    mlflow.set_tag("model_type", "BalancedRandomForest")
    mlflow.sklearn.log_model(new_model, "balanced_random_forest_model")

    mlflow.log_metric("accuracy", accuracy_BRF)
    mlflow.log_metric("precision", precision_BRF)
    mlflow.log_metric("recall", recall_BRF)
    mlflow.log_metric("f1_score", f1_BRF)
    mlflow.log_metric("auc_roc", roc_auc_BRF)
    mlflow.log_artifact("C:/Users/axa/THE BRIDGE_23/SEMANA 24. CORE. PROYECTO ML/81. PROYECTO ML/src/notebooks/mlruns/0/2271beee01a648ed80f5246ea4edfe28/artifacts/matrices_confusion", "matrices_confusion")

mlflow.end_run()

ruta_guardado = r"C:\Users\axa\THE BRIDGE_23\SEMANA 24. CORE. PROYECTO ML\81. PROYECTO ML\src\model/"
nombre_archivo = 'new_model.joblib'
ruta_completa = ruta_guardado + nombre_archivo
joblib.dump(new_model, ruta_completa)

ruta_guardado = r"C:\Users\axa\THE BRIDGE_23\SEMANA 24. CORE. PROYECTO ML\81. PROYECTO ML\src\model/"
