import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score,f1_score, ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV, train_test_split,GridSearchCV,_search
from scipy.stats import randint
import numpy as np
import pandas as pd 

# pra ver a arvore 
from sklearn.tree import export_graphviz
from IPython.display import Image, display
import graphviz

#  Carregar dataset
df = pd.read_csv('pd_speech_features.csv', header=1)

#  tirando id e classe
X = df.drop(columns=['id', 'class']).values 
y = df['class'].values 
n_features = df.drop(columns=['id', 'class']).shape[1]
#  Divisão entre  treino e  teste
n_pacientes=len(df)//3
#X = X.reshape((n_pacientes, 3, n_features))

X_per_patient=X.reshape((n_pacientes,3,n_features))
y_per_patient=y[::3]
X_train_3d, X_test_3d, y_train_1d, y_test_1d = train_test_split(
    X_per_patient, y_per_patient, test_size=0.2, random_state=42
)
X_train = X_train_3d.reshape(-1, n_features)
X_test = X_test_3d.reshape(-1, n_features)

y_train = np.repeat(y_train_1d, 3)
y_test = np.repeat(y_test_1d, 3)



n_samples_train = X_train.shape[0]
n_samples_test  = X_test.shape[0]


X_train_2d = X_train.reshape(X_train.shape[0], -1) 
X_test_2d = X_test.reshape(X_test.shape[0], -1)

# PARAMETROS 

param_dist = {
  'n_estimators': randint(100, 1100),
  'max_depth': randint(3, 15),
  'min_samples_split': randint(2, 10),
  'min_samples_leaf': randint(1, 5)
}

# Create a random forest classifier
rf = RandomForestClassifier(random_state=42, n_jobs=-1)

# Use random search to find the best hyperparameters

rand_search = RandomizedSearchCV(
  rf, param_distributions=param_dist,
  n_iter=10, cv=5, scoring='accuracy',
  n_jobs=-1, random_state=42,refit=True
)



# Create a variable for the best model 
rand_search.fit(X_train, y_train) 
best_rf = rand_search.best_estimator_
# Generate predictions with the best model
y_pred = best_rf.predict(X_test)
f1= f1_score(y_test,y_pred) 
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)


tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

print("\n" + "="*40)
print("         MATRIZ DE CONFUSÃO")
print("="*40)
print(f"Verdadeiros Positivos (TP): {tp}")
print(f"Verdadeiros Negativos (TN): {tn}")
print(f"Falsos Positivos (FP):      {fp}") 
print(f"Falsos Negativos (FN):      {fn}") 
print("-" * 40)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("f1:", f1)
# Print the best hyperparameters
print('Best hyperparameters:',  rand_search.best_params_)
# Export the first three decision trees from the forest
