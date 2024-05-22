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
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression


sns.set()

np.random.seed(10)

np.set_printoptions(precision=6, suppress=True)

def sigmoid(x):
    return 1/(1+np.exp(-x))

X = np.arange(-5,5,0.1)
y=sigmoid(X)

plt.figure(figsize=(15,10))
plt.plot(X,y)
plt.title('Sigmoid')

data = load_breast_cancer()

data


X=data.data
y=data.target

X.shape
y.shape

df = pd.DataFrame(data=X, columns=data.feature_names)
df.info()
df.describe().T

df

X_train, X_test, y_train, y_test = train_test_split(X, y)

X_train.shape
X_test.shape
y_train.shape
y_test.shape


classifier = LogisticRegression(max_iter=10000)
classifier.fit(X_train, y_train)

y_prob = classifier.predict_proba(X_test)
y_prob

y_pred = classifier.predict(X_test)
y_pred

cm = confusion_matrix(y_test, y_pred)
cm

acc = accuracy_score(y_test, y_pred)
acc


def plot_confusion_matrix(cm):
    cm = cm[::-1]
    cm = pd.DataFrame(cm, columns=['pred_0', 'pred_1'], index=['true_1','true_0'])
    
    fig = ff.create_annotated_heatmap(z=cm.values, x=list(cm.columns), y=list(cm.index), colorscale='ice', showscale=True, reversescale=True)
    fig.update_layout(width=400, height=400, title='Confusion matrix', font_size=16)
    pyo.plot(fig,'fig.html')
    
plot_confusion_matrix(cm)




print(classification_report(y_test, y_pred, target_names=data.target_names))
