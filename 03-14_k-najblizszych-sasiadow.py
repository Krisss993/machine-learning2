
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
from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV


sns.set()

np.random.seed(10)


raw_data = load_iris()

raw_data.data
raw_data.target

df1 = pd.DataFrame(data = raw_data.data, columns=raw_data.feature_names)
df2 = pd.DataFrame(data = raw_data.target, columns=['class'])
df = pd.concat([df1,df2], axis=1)

# MACIERZ KORELACJI POSZCZEGOLNYCH KOLUMN
sns.pairplot(df, hue='class')
sns.pairplot(df)
sns.heatmap(df)
print(df.corr())

df
df.info()
df.describe().T

X = raw_data.data[:,:2]
X
y=raw_data.target


# WYKRESY
plt.scatter(X[:,0], X[:,1], c=y, cmap='viridis')

def plotyl_plot():
    df = pd.DataFrame(X, columns=['sepal_length','sepal_width'])
    target = pd.DataFrame(y, columns=['class'])
    df = pd.concat([df,target], axis=1)
    fig = px.scatter(df, x='sepal_length', y='sepal_width', color='class', height=400,width=600)
    pyo.plot(fig)
    
plotyl_plot()

classifier = KNeighborsClassifier(n_neighbors=6)

classifier = KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski', metric_params=None, n_jobs=None, n_neighbors=5, p=2, weights='uniform')

classifier.fit(X,y)

accuracy = classifier.score(X, y)
accuracy


x_min, x_max = X[:,0].min() - 0.5, X[:,0].max() + 0.5
y_min, y_max = X[:,1].min() - 0.5, X[:,1].max() + 0.5

xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
mesh = np.c_[xx.ravel(), yy.ravel()]
Z = classifier.predict(mesh)
Z=Z.reshape(xx.shape)


# WYKRES GRANIC DECYZYJNYCH
plt.figure(figsize=(9,7))
plt.pcolormesh(xx, yy, Z, cmap='gnuplot', alpha=0.5)
plt.scatter(X[:,0], X[:,1], c=y, cmap='gnuplot', edgecolors='r')
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.title(f'3-class classif. k=5, accuracy: {accuracy:.4f}')
plt.show()


# WYKRESY GRANIC DECYZYJNYCH DLA k_neighbors od 1 do 6
plt.figure(figsize=(12,12))
for i in range(1,7):
    plt.subplot(3,2,i)
    classifier = KNeighborsClassifier(n_neighbors=i)
    classifier.fit(X,y)
    accuracy = classifier.score(X, y)
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
    mesh = np.c_[xx.ravel(), yy.ravel()]
    Z = classifier.predict(mesh)
    Z=Z.reshape(xx.shape)
    plt.pcolormesh(xx, yy, Z, cmap='gnuplot', alpha=0.3)
    plt.scatter(X[:,0], X[:,1], c=y, cmap='gnuplot', edgecolors='r')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title(f'3-class classif. k={i}, accuracy: {accuracy:.4f}')
plt.show()




grid_params = {'n_neighbors':range(2,30)}
classifier = KNeighborsClassifier()

# kross validation jest przydatne przy malych danych, przy wart 3, dzieli dane na 3 czesci uczy na 2 i sprawdza na 1, wszystkie kombinacje po kolei
gs = GridSearchCV(classifier, grid_params, cv=3)
gs.fit(X,y)

gs.best_params_
k = gs.best_params_['n_neighbors']
k
# ZWRACA NAJLEPSZY ESTYMATOR
cl = gs.best_estimator_


x_min, x_max = X[:,0].min() - 0.5, X[:,0].max() + 0.5
y_min, y_max = X[:,1].min() - 0.5, X[:,1].max() + 0.5

xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
mesh = np.c_[xx.ravel(), yy.ravel()]
Z = cl.predict(mesh)
Z=Z.reshape(xx.shape)


# WYKRES GRANIC DECYZYJNYCH
plt.figure(figsize=(9,7))
plt.pcolormesh(xx, yy, Z, cmap='gnuplot', alpha=0.5)
plt.scatter(X[:,0], X[:,1], c=y, cmap='gnuplot', edgecolors='r')
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.title(f'3-class classif. k={k}, accuracy: {accuracy:.4f}')
plt.show()

