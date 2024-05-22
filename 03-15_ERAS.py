import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.offline as pyo
import tensorflow as tf
from tensorflow.keras.datasets.mnist import load_data
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.activations import linear
from tensorflow.keras.activations import sigmoid
from tensorflow.keras.activations import relu
from tensorflow.keras.activations import tanh
from tensorflow.keras.utils import to_categorical


np.set_printoptions(precision=12, suppress=True, linewidth=120)
print(tf.__version__)


model = Sequential()
print(model)

# DODANIE WARSTWY GESTO POLACZONEJ
# PODAJEMY LICZBE NEURONOW JAKIE CHCEMY W TEJ WARSTWIE - units
# PODAJEMY ROZMIAR POJEDYNCZEJ DANEJ - input_shape
model.add(Dense(units=4, input_shape=(10,)))

# PARAM TO (input_shape + 1) * units - TO PARAMETRY DO PRZETRENOWANIA
model.summary()

# Total params: 44
# Trainable params: 44
# Non-trainable params: 0
# Non-trainable mozna wykorzystac do transfer learning

# output_shape tej warstwy to output_shape[-1] + 1 * units
# (4 + 1) * 2
model.add(Dense(units=2))

model.summary()


model.add(Dense(units=4, input_shape=(10,1)))


random_data = np.linspace(start=-3, stop=3, num=300)


# MODEL AKTYWACJI LINIOWY
data = pd.DataFrame({'data':random_data, 'linear':linear(random_data)})
data
# WYKRES
fig = px.line(data, x='data', y='linear', width=500, height=400, range_y=[-3,3])
pyo.plot(fig)


# MODEL AKTYWACJI SIGMOID
data = pd.DataFrame({'data':random_data, 'sigmoid':sigmoid(random_data)})
data
# WYKRES
fig = px.line(data, x='data', y='sigmoid', width=500, height=400, range_y=[-0.5,1.5])
pyo.plot(fig)


# MODEL AKTYWACJI RELU
data = pd.DataFrame({'data':random_data, 'relu':relu(random_data)})
data
# WYKRES
fig = px.line(data, x='data', y='relu', width=500, height=400, range_y=[-0.5,1.5])
pyo.plot(fig)


# MODEL AKTYWACJI TANGENS HIPERBOLICZNY
data = pd.DataFrame({'data':random_data, 'tanh':tanh(random_data)})
data
# WYKRES
fig = px.line(data, x='data', y='tanh', width=500, height=400, range_y=[-3,3])
pyo.plot(fig)



model = Sequential()
model.add(Dense(units=8, activation='relu', input_shape=(10,)))
model.add(Dense(units=1, activation='sigmoid'))
model.summary()

# KOMPILACJA MODELU


# KLASYFIKACJA BINARNA JEDNA Z 2 KLAS
# optimizer MINIMALUZUJE FUNKCJE STRATY
# loss FUNKCJA STRATY
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# KLASYFIKACJA WIELOKLASOWA
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# REGRESJA
model.compile(optimizer='rmsprop',
              loss='mse')



# TRENOWANIE MODELU
# batch_size=32 - PO KAZDYCH 32 PROBKACH NASTEPUJE AKTUALIZACJA WAG
# model.fit(X=data, y=labels, epochs, batch_size=32)

# validation_split -  WYDZIELENIE ZBIORU WALIDACYJNEGO COS JAK TRAIN_TEST_SPLIT w sklearn, DOMYSLNIE 20% Z DANYCH
# model.fit(X=data, y=labels, epochs, batch_size=32, validation_split=0.2)

# validation_data - PRZYJMUJE TUPLE KTORA SKLADA SIE Z DANYCH TESTOWYCH(WALIDACYJNYCH), JESLI MAMY JUZ TAKI ZBIOR
# model.fit(X=data, y=labels, epochs, batch_size=32, validation_data=(x_val, y_val))



data = np.random.randn(10000, 150)
labels = np.random.randint(2, size=(10000,1))

data.shape
labels.shape

data[:3]
labels[:10]

model = Sequential()
model.add(Dense(units=32, activation='relu', input_shape=(150,)))
model.add(Dense(units=1, activation='sigmoid'))

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(data, labels, epochs=20)




model = Sequential()
model.add(Dense(units=32, activation='relu', input_shape=(150,)))
model.add(Dense(units=1, activation='sigmoid'))

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# DODANIE PARAMETRU BATCH_SIZE
model.fit(data, labels, epochs=20, batch_size=20)



model = Sequential()
model.add(Dense(units=32, activation='relu', input_shape=(150,)))
model.add(Dense(units=1, activation='sigmoid'))

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# DODANIE PARAMETRU validation_split, WYSWIETLAJA TAKZE loss i accuracy dla zbioru walidacyjnego
model.fit(data, labels, epochs=20, batch_size=20, validation_split=0.3)



model = Sequential()
model.add(Dense(units=32, activation='relu', input_shape=(150,)))
model.add(Dense(units=1, activation='sigmoid'))

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# DODANIE PARAMETRU verbose=0
history = model.fit(data, labels, epochs=20, batch_size=20, validation_split=0.3, verbose=0)

metrics=history.history

metrics.keys()




test_data = np.random.randn(5,150)
test_labels = np.random.randint(2,size=(5,1))
# PREDYKCJA
# PRAWDOPODOBIENSTWO
model.predict(test_data)

# KLASY
y_pred = np.argmax(model.predict(test_data), axis=-1)












# KLASYFIKACJA WIELOKLASOWA


data = np.random.random((10000, 150))
labels = np.random.randint(10, size=(10000, 1))
     
print(data.shape)
print(labels.shape)



# TWORZY Z KAZDEJ KLASY KOLUMNE GDZIE WARTOSC 1 ODPOWIADA DANEJ KLASIE, RESZTA TO 0
# NP DLA KLASY 1 WYGLADA TO TAK [0 1 0 0 0 0 0 0 0 0]
# NP DLA KLASY 0 WYGLADA TO TAK [1 0 0 0 0 0 0 0 0 0]
labels = to_categorical(labels, num_classes=10)
labels
labels[1]

model = Sequential()

# INPUT SHAPE TO LICZBA KOLUMN DANYCH WEJSCIOWYCH, LICZBA CECH DANEJ PROBKI
model.add(Dense(units=32, activation='relu', input_shape=(150,)))

# LICZBA NEURONOW NA KONCU MUSI ZGADZAC SIE Z LICZBA KLAS
model.add(Dense(units=10, activation='softmax'))

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


model.fit(data, labels, epochs=30, validation_split=0.2)


test_data = np.random.random((10,150))

# ZWRACA 10 PRAWDOPODOBIENSTW KLA KAZDEJ KLASY
model.predict(test_data)

y_pred = np.argmax(model.predict(test_data), axis=-1)
y_pred















# REGRESJA


data = np.random.random((10000, 150))
labels = 50 * np.random.random(10000)

model = Sequential()
model.add(Dense(units=32, activation='relu', input_shape=(150,)))

# 1 NEURON PONIEWAZ PRZEWIDUJEMY WARTOSC CIAGLA, AKTYWACJA STANDARDOWO LINIOWA
model.add(Dense(units=1))

model.compile(optimizer='rmsprop',
              loss='mse')

model.fit(data, labels, epochs=30, batch_size=32, validation_split=0.2)






model = Sequential()
model.add(Dense(units=32, activation='relu', input_shape=(150,)))

# 1 NEURON PONIEWAZ PRZEWIDUJEMY WARTOSC CIAGLA, AKTYWACJA STANDARDOWO LINIOWA
model.add(Dense(units=1))

model.compile(optimizer='rmsprop',
              loss='mae',
              metrics=['mse'])

model.fit(data, labels, epochs=10, batch_size=32, validation_split=0.2)

test_data = np.random.random((10, 150))
model.predict(test_data)
