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
from IPython.display import Image
import matplotlib.image as mpimg
from PIL import Image
import urllib.request
from sklearn.tree import DecisionTreeClassifier
from mlxtend.plotting import plot_decision_regions
from sklearn.tree import export_graphviz
from IPython.display import Image
import pydotplus
import joblib
from sklearn import tree


sns.set()
np.random.seed(10)

iris = load_iris()
data = iris.data
target=iris.target
feature_names = [name.replace(' ', '_')[:-5] for name in iris.feature_names]
class_names = iris.target_names


plt.imshow(img_np)
plt.axis('off')  # Wyłączenie osi
plt.show()

data_targets = np.c_[data,target]
h = np.hstack((data,target.reshape(-1,1)))
h
df = pd.DataFrame(data_targets, columns=feature_names + ['class'])

df.describe().T.apply(lambda x:round(x, 2))


plt.figure(figsize=(8,6))
sns.scatterplot( data=df, x='sepal_length', y='sepal_width', hue='class', legend='full', palette=sns.color_palette()[:3])

plt.figure(figsize=(8,6))
sns.scatterplot( data=df, x='petal_length', y='petal_width', hue='class', legend='full', palette=sns.color_palette()[:3])

df['class'].value_counts()

X=df.copy()
X=X[['petal_length', 'petal_width', 'class']]
y = X.pop('class')

X
y

X=X.values
y=y.values.astype('int16')


# Budowa klasyfikatora drzewa decyzyjnego
# DecisionTreeClassifier?
classifier = DecisionTreeClassifier(max_depth=1, random_state=30)
classifier.fit(X,y)




# Wykreślenie granic decyzyjnych drzewa
colors='#f1865b,#31c30f,#64647F,#d62728,#9467bd,#8c564b,#e377c2,#7f7f7f,#bcbd22,#17becf'

acc = classifier.score(X, y)

plt.figure(figsize=(8, 6))
plot_decision_regions(X, y, classifier, legend=2, colors=colors)
plt.xlabel('petal length (cm)')
plt.ylabel('petal width (cm)')
plt.title(f'Drzewo decyzyjne: max_depth=1, accuracy: {acc * 100:.2f}%')
plt.show()



# STWORZENIE GRAFU DRZEWA DECYZYJNEGO
plt.figure(figsize=(12, 8))
tree.plot_tree(classifier, filled=True, feature_names=iris.feature_names, class_names=iris.target_names)
plt.show()



def make_decision_tree(max_depth=2):
    # trenowanie modelu
    classifier = DecisionTreeClassifier(max_depth=max_depth, random_state=30)
    classifier.fit(X, y)

    # eksport grafu drzewa
    plt.figure(figsize=(12, 8))
    tree.plot_tree(classifier, filled=True, feature_names=iris.feature_names, class_names=iris.target_names)
    plt.show()
    
    # obliczenie dokładności
    acc = classifier.score(X, y) 

    # wykreślenie granic decyzyjnych
    colors='#f1865b,#31c30f,#64647F,#d62728,#9467bd,#8c564b,#e377c2,#7f7f7f,#bcbd22,#17becf'
    plt.figure(figsize=(8, 6))
    ax = plot_decision_regions(X, y, classifier, legend=0, colors=colors)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, ['setosa', 'versicolor', 'virginica'], framealpha=0.3)
    plt.xlabel('petal length (cm)')
    plt.ylabel('petal width (cm)')
    plt.title(f'Drzewo decyzyjne: max_depth={max_depth}, accuracy={acc * 100:.2f}')
    plt.show()

make_decision_tree(1)
make_decision_tree(2)
make_decision_tree(3)
make_decision_tree(4)
make_decision_tree(5)
