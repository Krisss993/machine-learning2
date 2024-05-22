
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
from sklearn.preprocessing import PolynomialFeatures


sns.set()

X = np.arange(-10,10,0.5)
noise = 80 * np.random.randn(40)
y = -X**3 + 10*X**2-2*X + 3 + noise

X=X.reshape(-1,1)


_=plt.scatter(X,y)

lr = LinearRegression()
lr.fit(X,y)
y_pred = lr.predict(X)

plt.scatter(X,y)
plt.plot(X,y_pred, c='red')

poly = PolynomialFeatures()
X_poly = poly.fit_transform(X)
X_poly

lr_poly = LinearRegression()
lr_poly.fit(X_poly, y)

y_pred_poly = lr_poly.predict(X_poly)

plt.scatter(X,y)
plt.plot(X,y_pred_poly, c='red')


poly3 = PolynomialFeatures(degree=3)
X_poly3 = poly3.fit_transform(X)

lr_poly3 = LinearRegression()
lr_poly3.fit(X_poly3,y)
y_pred_poly3 = lr_poly3.predict(X_poly3)

plt.scatter(X,y)
plt.plot(X,y_pred_poly3, c='red')




poly = PolynomialFeatures(degree=3)
X_poly = poly.fit_transform(X)

lr_poly = LinearRegression()
lr_poly.fit(X_poly,y)
y_poly = lr_poly.predict(X_poly)

plt.plot(X,y_poly)
plt.scatter(X,y)

lr_poly.score(X, y_poly)
