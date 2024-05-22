import sklearn
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
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


sklearn.__version__

np.random.seed(10)

raw_data = datasets.load_iris()
raw_data.keys()

raw_data.DESCR
data = raw_data.data

target = raw_data.target
target
data.shape
target.shape

X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
len(X_test)
len(y_pred)




classification_report(y_test, y_pred)


accuracy = 1-np.count_nonzero(y_pred-y_test)/len(y_test)
accuracy = (len(y_test) - np.count_nonzero(y_pred-y_test))/len(y_test)
accuracy

accuracy_s = accuracy_score(y_test, y_pred)
accuracy_s


# PRZYGOTOWANIE DANYCH DO WYSWIETLENIA WYKRESU
results = pd.DataFrame({'y_test': y_test, 'y_pred':y_pred})
results
# POSORTOWANIE WG WARTOSCI
results.sort_values(by='y_test', inplace=True)
results
# NADANIE NUMEROW PROBKOM
results.reset_index(inplace=True, drop=True)
results['sample'] = results.index+1
results

# WIZUALIZACJA POMYLEK NA 2 WYKRESACH PUNKTOWYCH
fig = make_subplots(rows=2, cols=1)
fig.add_trace(go.Scatter(x=results['sample'], y=results['y_test'], mode='markers', name='y_test'), row=1,col=1)
fig.add_trace(go.Scatter(x=results['sample'], y=results['y_pred'], mode='markers', name='y_pred'), row=2,col=1)
fig.update_layout(width=900, height=500, title='Klasyfikator binarny')
pyo.plot(fig,'fig.html')


# MACIERZ POMYLEK
cm = confusion_matrix(y_test , y_pred)
cm

def plot_confusion_matrix(cm):
    cm=cm[::-1]
    cm = pd.DataFrame(cm, columns=['pred_0', 'pred_1', 'pred_2'], index=['true_2','true_1', 'true_0'])
    
    fig = ff.create_annotated_heatmap(z=cm.values, x=list(cm.columns), y=list(cm.index), colorscale='ice', showscale=True, reversescale=True)
    fig.update_layout(width=400, height=400, title='Confusion matrix', font_size=16)
    pyo.plot(fig,'fig.html')
    
plot_confusion_matrix(cm)




# KLASYFIKACJA BINARNA
data = raw_data.data[:100]
target = raw_data.target[:100]
target = raw_data.target
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2)
model = LogisticRegression(max_iter=50)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
cm = confusion_matrix(y_test , y_pred)
cm

tn, fp, fn, tp = cm.ravel()
tn

# TYPE I ERROR- False Positive Rate
# JAK WIELE RAZY PRZEWIDZIELISMY KLASE POZYTYWNA GDY BYLA KLASA NEGATYWNA
# BLAD MNIEJSZY
fpr = fp/(fp+tn)
fpr

# TYPE II ERROR- False Negative Rate
# JAK WIELE RAZY PRZEWIDZIELISMY KLASE NEGATYWNA GDY BYLA KLASA POZYTYWNA
# BLAD DUZY
fnr = fn/(fn+tp)
fnr

# PRECISION
# ILE OBSERWACJI POZYTYWNYCH JEST PRZEWIDZIANE JAKO POZYTYWNE
precision = tp / (tp+fp)
precision

# RECALL
# JAK WIELE INFORMACJI ZE WSZYSTKICH POZYTYWNYCH JEST PRZEWIDZIANE JAKO POZYTYWNE
recall = tp / (tp+fn)
recall






# KRZYWA ROC
fpr, tpr, tresh = roc_curve(y_test, y_pred, pos_label=1)
roc = pd.DataFrame({'fpr':fpr,'tpr':tpr})


def plot_roc_curve(y_true, y_pred):
    fpr, tpr, tresh = roc_curve(y_true, y_pred, pos_label=1)

    fig = go.Figure(data=[go.Scatter(x=roc['fpr'], y=roc['tpr'], line_color='red', name='ROC Curve'),
                        go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line_dash='dash', line_color='navy')],
                    layout=go.Layout(xaxis_title='False Positive Rate',
                                    yaxis_title='True Positive Rate',
                                    title='ROC Curve',
                                    showlegend=False,
                                    width=800,
                                    height=400))
    pyo.plot(fig,'fig.html')
    
plot_roc_curve(y_test, y_pred)




#####################################

# KLASYFIKACJA WIELOKLASOWA

#####################################


y_true = np.array([1, 0, 1, 2, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 2, 1, 1, 2, 2, 1, 0, 1, 1, 0, 2, 1, 1, 2, 2])
y_pred = np.array([0, 0, 1, 2, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 2, 1, 2, 1, 2, 1, 0, 2, 1, 0, 1, 1, 1, 2, 2])

accuracy_score(y_true, y_pred)


cm = confusion_matrix(y_true, y_pred)
cm

def plot_confusion_matrix(cm):
    cm=cm[::-1]
    cm = pd.DataFrame(cm, columns=['pred_0', 'pred_1', 'pred_2'], index=['true_2','true_1', 'true_0'])
    
    fig = ff.create_annotated_heatmap(z=cm.values, x=list(cm.columns), y=list(cm.index), colorscale='ice', showscale=True, reversescale=True)
    fig.update_layout(width=400, height=400, title='Confusion matrix', font_size=16)
    pyo.plot(fig,'fig.html')
    
plot_confusion_matrix(cm)

classification_report(y_true, y_pred, target_names=['label_1','label_2','label_3'])




#####################################

#####################################                 MODEL REGRESJI 

#####################################





y_true = 100 + 20 * np.random.randn(50)
y_true

y_pred = y_true + 10 * np.random.randn(50)
y_pred


results = pd.DataFrame({'y_true':y_true, 'y_pred':y_pred})
results['Error'] = results['y_true'] - results['y_pred']
results


def plot_regression_results(y_true, y_pred): 
    results = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred})
    mi = results[['y_true', 'y_pred']].min().min()
    ma = results[['y_true', 'y_pred']].max().max()

    
    fig = go.Figure(data=[go.Scatter(x=results['y_true'], y=results['y_pred'], mode='markers'),
                    go.Scatter(x=[mi, ma], y=[mi, ma])],
                    layout=go.Layout(showlegend=False, width=800, height=500,
                                     xaxis_title='y_true', 
                                     yaxis_title='y_pred',
                                     title='Regression results'))
    pyo.plot(fig,'fig.html')
    
plot_regression_results(y_true, y_pred)

mi = results[['y_true', 'y_pred']].min().min()
ma = results[['y_true', 'y_pred']].max().max()

fig = go.Figure(data=[
    go.Scatter(x=results['y_true'], y=results['y_pred'], mode='markers'),
    go.Scatter(x=[mi,ma], y=[mi,ma])
                      ], layout=go.Layout(showlegend=False, title='Title', xaxis_title = 'true', yaxis_title='pred', width=800,height=600))
pyo.plot(fig,'fig.html')

y_true = 100 + 20 * np.random.randn(1000)
y_true

y_pred = y_true + 10 * np.random.randn(1000)
y_pred

results = pd.DataFrame({'y_true':y_true, 'y_pred':y_pred})
results['error'] = results['y_true'] - results['y_pred']
results

fig = px.histogram(data_frame=results, x='error', nbins=50)
pyo.plot(fig)

def mean_absolut_error(y_true, y_pred):
    return abs(y_true-y_pred).sum()/len(y_true)

mean_absolut_error(y_true, y_pred)

mae=mean_absolute_error(y_true, y_pred)
mae

def mean_square_error(y_true,y_pred):
    return ((y_true-y_pred)**2).sum()/len(y_true)

mean_square_error(y_true, y_pred)

mse=mean_squared_error(y_true, y_pred)
mse

rmse = mse**(1/2)
rmse

r2score = 1-((y_true-y_pred)**2).sum()/((y_true-y_pred.mean())**2).sum()
r2score

r2 = r2_score(y_true, y_pred)
r2

