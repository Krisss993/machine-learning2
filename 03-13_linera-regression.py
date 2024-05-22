import sklearn
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.metrics import accuracy_score
import plotly.offline as pyo
import plotly.figure_factory as ff
from sklearn.metrics import roc_curve
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()


X = np.arange(0,50,0.5)
noise = 10 * np.random.randn(100)
y = 2 * X + 100 + noise
X = X.reshape(-1,1)

X.shape
y.shape

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
y_pred
y_test

plt.figure(figsize=(7,5))
plt.scatter(X_test, y_test, color='blue')
plt.scatter(X_train, y_train, color='green')
#plt.plot(X_test,y_pred, color='red')
plt.plot(X, lr.intercept_ + lr.coef_*X, color='red')
plt.xlabel('X')
plt.ylabel('y')


lr.coef_
lr.intercept_

# Y = lr.intercept_ + lr.coef_ * X1
# == 
# Y = 99.91 + 1.96 * X1
# ROWNANIE PROSTEJ PRZEWIDUJACEJ WYNIK 

lr.score(X_test, y_test)
