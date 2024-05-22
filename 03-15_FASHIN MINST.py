import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.offline as pyo

import tensorflow as tf
from tensorflow.keras.datasets.fashion_mnist import load_data
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from keras.utils import to_categorical

np.set_printoptions(precision=12, suppress=True, linewidth=150)
pd.options.display.float_format = '{:.6f}'.format
sns.set()
tf.__version__



(X_train, y_train), (X_test, y_test) = load_data()


print(f'X_train shape: {X_train.shape}')
print(f'y_train shape: {y_train.shape}')
print(f'X_test shape: {X_test.shape}')
print(f'y_test shape: {y_test.shape}')
print(f'X_train[0] shape: {X_train[0].shape}')

X_train[0]


plt.imshow(X_train[0], cmap='gray_r')
plt.axis('off')
plt.show()


class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

plt.figure(figsize=(18, 13))
for i in range(1, 11):
    plt.subplot(2, 5, i)
    plt.axis('off')
    plt.imshow(X_train[i-1], cmap='gray_r')
    plt.title(class_names[y_train[i-1]], color='black', fontsize=16)
plt.show()

plt.figure(figsize=(18, 13))
for i in range(1, 11):
    plt.subplot(1,10,i)
    plt.axis('off')
    plt.imshow(X_train[i-1], cmap='gray_r')
    plt.title(class_names[y_train[i-1]])
plt.show()


# STANDARYZACJA DANYCH
X_train = X_train / 255.
X_test = X_test / 255.


# ZMIANA NA DANE KATEGORYCZNE
y_train = to_categorical(y_train, num_classes=10)
y_test_cat = to_categorical(y_test, num_classes=10)


model = Sequential()
model.add(Flatten(input_shape=(28, 28)))

# RELU JEST NAJCZESCIEJ STOSOWANE DO WARSTW UKRYTYCH
model.add(Dense(units=128, activation='relu'))
model.add(Dense(units=10, activation='softmax'))

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()


history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2)

metrics = pd.DataFrame(history.history)
metrics['epoch'] = history.epoch
metrics


# WYKRESY 
fig = plt.figure(figsize=(15,10))
plt.subplot(212)
plt.plot(metrics['epoch'], metrics['accuracy'], label='train')
plt.plot(metrics['epoch'], metrics['val_accuracy'], label='validation')
plt.legend()

plt.subplot(211)
plt.plot(metrics['epoch'], metrics['loss'], label='loss')
plt.plot(metrics['epoch'], metrics['val_loss'], label='validation loss')
plt.legend()
fig.show()


fig = make_subplots(rows=2, cols=1)
fig.add_trace(go.Scatter(x=metrics['epoch'], y=metrics['accuracy'], name='accuracy'), row=1, col=1)
fig.add_trace(go.Scatter(x=metrics['epoch'], y=metrics['val_accuracy'], name='val_accuracy'), row=1, col=1)
fig.add_trace(go.Scatter(x=metrics['epoch'], y=metrics['loss'], name='loss'), col=1, row=2)
fig.add_trace(go.Scatter(x=metrics['epoch'], y=metrics['val_loss'], name='val_loss'), col=1, row=2)

fig.update_xaxes(title_text='epochs')
fig.update_yaxes(title_text='accuracy')
fig.update_layout(go.Layout(width=1000, title='Accuracy and loss'))
pyo.plot(fig)





test_loss, test_acc = model.evaluate(X_test, y_test_cat, verbose=0)
print(test_acc)

# PREDYKCJA NA PODSTAWIE MODELU


pred = model.predict(X_test)
pred

pred_df = pd.DataFrame(pred)
pred_df

y_pred = np.argmax(pred, axis=-1)
y_pred


plt.bar(class_names, pred[0])
plt.bar(class_names, pred[900])
plt.imshow(X_test[900], cmap='gray_r')
plt.title(class_names[y_test[900]])

predictions_cls = np.argmax(pred, axis=-1)
predictions_cls

predictions_df = pd.DataFrame(pred)
predictions_df.head()











idx = 2900

if predictions_cls[idx] == y_test[idx]:
    color = 'green'
else:
    color = 'red'

fig = go.Figure()
fig.add_trace(go.Bar(x=class_names, y=predictions_df.iloc[idx], orientation='v', 
                     marker_color=color))
fig.update_layout(width=600, height=300,
                  title=f'Predykcja: {class_names[predictions_cls[idx]]}')
pyo.plot(fig)

from PIL import Image, ImageOps
import numpy as np

data = (X_test[idx] * 255).astype(np.uint8)
img = Image.fromarray(data, 'L')
img = ImageOps.invert(img.convert('RGB'))
img.save('sample.png')

from IPython import display
display.Image('sample.png', width=200)













misclassified = []
for idx, _ in enumerate(X_test):
    if predictions_cls[idx] != y_test[idx]:
        misclassified.append(idx)

index_mapper = {}


for idx, idx_real in enumerate(misclassified):
    index_mapper[idx] = idx_real

idx = 121 #@param {type: 'slider', min:0, max:1119}

fig = go.Figure()
fig.add_trace(go.Bar(x=class_names, 
                     y=predictions_df.iloc[index_mapper[idx]], 
                     orientation='v', 
                     marker_color='red'))

fig.update_layout(width=600, height=300,
                  title=(f' Etykieta: {class_names[y_test[index_mapper[idx]]]}'
                      f' ~ Predykcja: {class_names[predictions_cls[index_mapper[idx]]]}'))
pyo.plot(fig)

from PIL import Image, ImageOps
import numpy as np

data = (X_test[index_mapper[idx]] * 255).astype(np.uint8)
img = Image.fromarray(data, 'L')
img = ImageOps.invert(img.convert('RGB'))
img.save('sample.png')

from IPython import display
display.Image('sample.png', width=200)


