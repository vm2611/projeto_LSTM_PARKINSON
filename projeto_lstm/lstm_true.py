import numpy as np
import pandas as pd # Simplificando o import
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix
from keras.models import Sequential
from keras.layers import Dense, LSTM,Dropout,GaussianNoise
from keras.callbacks import EarlyStopping
from matplotlib import pyplot

#  Carregar dataset
df = pd.read_csv('pd_speech_features.csv', header=1)

#  tirando id e classe
X = df.drop(columns=['id', 'class']).values 
y = df['class'].values 
n_features = df.drop(columns=['id', 'class']).shape[1]
#  Divisão entre  treino e  teste
n_pacientes=len(df)//3
X = X.reshape((n_pacientes, 3, n_features))



X_per_patient=X.reshape((n_pacientes,3,n_features))
y_per_patient=y[::3]

X_train, X_test, y_train, y_test = train_test_split(X_per_patient, y_per_patient, test_size=0.2, random_state=42)


n_samples_train = X_train.shape[0]
n_samples_test  = X_test.shape[0]

X_train_2d=X_train.reshape(n_samples_train*3,n_features)
X_test_2d=X_test.reshape(n_samples_test*3,n_features)

scaler = MinMaxScaler()
X_train_2d_scaled=scaler.fit_transform(X_train_2d)
X_test_2d_scaled=scaler.transform(X_test_2d)
X_train_reshaped = X_train_2d_scaled.reshape(n_samples_train, 3, n_features)
X_test_reshaped  = X_test_2d_scaled.reshape(n_samples_test,   3, n_features)

#   LSTM 
#   Rede LSTM
model = Sequential()

# Camada LSTM 
model.add(LSTM(32, input_shape=(3, n_features)))
model.add(GaussianNoise(0.3))
# Camada de Saída 
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#  Treinamento
print("Iniciando treinamento da LSTM...")
#early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True) 
# colocar isso no history caso queira que pare antes para evitar problema #callbacks=[early_stop]
pesos_das_classes = {
    1: 3.0,# puniçao para caso erre true parkison
    0: 1.0  
}
history =  model.fit(X_train_reshaped, y_train, epochs=45,validation_split=0.1, batch_size=16,class_weight=pesos_das_classes, verbose=1)

#Matriz de Confusão
y_pred_prob = model.predict(X_test_reshaped)

#criterio de saudavel ou nao  
y_pred = (y_pred_prob > 0.5).astype(int)


tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

print("\n" + "="*40)
print("         MATRIZ DE CONFUSÃO")
print("="*40)
print(f"Verdadeiros Positivos (TP): {tp}")
print(f"Verdadeiros Negativos (TN): {tn}")
print(f"Falsos Positivos (FP):      {fp}") 
print(f"Falsos Negativos (FN):      {fn}") 
print("-" * 40)
acuracia = (tp + tn) / (tp + tn + fp + fn)
precisao = tp / (tp + fp) 
print(f"Acurácia:{acuracia:.2%}")
print(f"Precisão:{precisao:.3f}")
print(f"recall :{tp/(tp+fn):.3f} ")
print(f"F1-Score:{(2*tp)/(2*tp+fp+fn):.3f}")

print("="*40)