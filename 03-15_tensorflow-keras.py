import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

import tensorflow as tf
from tensorflow.keras.datasets.mnist import load_data
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
import plotly.offline as pyo

np.set_printoptions(precision=12, suppress=True, linewidth=120)
print(tf.__version__)


# WCZYTANIE DANYCH Z PODZIALEM NA TRAIN TEST
(X_train, y_train), (X_test, y_test) = load_data()

print(X_train[0])

# ROZMIAR PIERWSZEGO ELEMENTU DANYCH TO 28x28
print(f'X_train[0] shape: {X_train[0].shape}')

# IMAGE PIERWSZEGO ELEMENTU
plt.imshow(X_train[0], cmap='gray_r')
plt.axis('off')


# WYDRUK PIERWSZYCH 10 ELEMENTOW
plt.figure(figsize=(13, 13))
for i in range(1, 11):
    plt.subplot(1, 10, i)
    plt.axis('off')
    plt.imshow(X_train[i-1], cmap='gray_r')
    plt.title(y_train[i-1], color='black', fontsize=16)
plt.show()


# BUDOWA MODELU SIECI NEURONOWEJ 
model = Sequential()

# SPLASZCZENIE PIERWSZEJ WARSTWY DO ROZMIARU 28*28 = 784
model.add(Flatten(input_shape=(28, 28)))

# DODANIE WARSTWY GLEBOKIEJ Z AKTYWACJA RELU - zeruje wartosci ujemne, dodatnie bez zmian
model.add(Dense(units=128, activation='relu'))

# PORZUCA 20% NEURONOW
model.add(Dropout(0.2))

# WARSTWA WYJSCIOWA MA ROZMIAR 10, PONIEWAZ JEST 10 CYFR - wynikow koncowych, ktore przedstawic ma model
# SOFTMAX ZWRACA PRAWDOPODOBIENSTWO WYSTAPIENIA DANEJ WARTOSCI
model.add(Dense(units=10, activation='softmax'))

# KOMPILOWANIE MODELU Z DANYMI PARAMETRAMI
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# PODSUMOWANIE: ROZMIAR WARSTWY, ILOSC PARAMETROW TESTOWALNYCH
model.summary()

# TRENOWANIE MODELU
history = model.fit(X_train, y_train, epochs=5)


# PODAJE STRATE - LOSS ORAZ ACCURACY - OCENE MODELU
model.evaluate(X_test, y_test, verbose=2)

# PODAJE STRATE - LOSS ORAZ ACCURACY - OCENE MODELU DLA KAZDEJ EPOKI JAKO DATAFRAME
metrics = pd.DataFrame(history.history)
metrics


# WYKRESY FUNKCJI LOSS ORAZ ACCURACY
fig = make_subplots(rows=2, cols=1)
fig.add_trace(go.Scatter(y=metrics['loss'], name='loss'), row=1, col=1)
fig.add_trace(go.Scatter(x=metrics.index, y=metrics['accuracy'], name='accuracy'), row=2, col=1)
fig.update_layout(go.Layout(width=800, height=400))
pyo.plot(fig)


# PREDYKCJA
model.predict(X_test)

y_pred = np.argmax(model.predict(X_test), axis=-1)
y_pred
     
# 
pred = pd.concat([pd.DataFrame(y_test, columns=['y_test']), pd.DataFrame(y_pred, columns=['y_pred'])], axis=1)
pred.head(10)

# ZWRACA INDEKSY ZLE OSZACOWANYCH DANYCH
misclassified = pred[pred['y_test'] != pred['y_pred']]
misclassified.index[:10]


# WYKRES PRZEDSTAWIAJACY 10 ZLE OSZACOWANYCH WYNIKOW
plt.figure(figsize=(16, 16))
for i, j in zip(range(1, 11), misclassified.index[:10]):
    plt.subplot(1, 10, i)
    plt.axis('off')
    plt.imshow(X_test[j], cmap='gray_r')
    plt.title(f'y_test: {y_test[j]}\ny_pred: {y_pred[j]}', color='black', fontsize=12)
plt.show()
     
