import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.figure_factory as ff
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
import plotly.offline as pyo


sns.set()

digits = datasets.load_digits()

digits.keys()

images = digits.images
labels = digits.target

images[0]


plt.figure(figsize=(10, 10))
for index, (image, label) in enumerate(list(zip(images, labels))[:4]):
    plt.subplot(2, 4, index + 1)
    plt.axis('off')
    plt.imshow(image, cmap='Greys')
    plt.title('Label: {}'.format(index, label))
    
X_train, X_test, y_train, y_test = train_test_split(images, labels)

print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)
print('y_train shape:', y_train.shape)
print('y_test shape:', y_test.shape)

X_train = X_train.reshape(X_train.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)

print()
print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)

classifier = SVC(gamma=0.001)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
y_pred

print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
cm

# MACIERZ HEATMAP
_ = sns.heatmap(cm, annot=True, cmap=sns.cm.rocket_r)


# MACIERZ HEATMAP PLOTLY
columns = ['pred_' + str(i) for i in range(10)]
index = ['true_' + str(i) for i in range(10)]

def plot_confusion_matrix(cm):
    # Mulitclass classification, 3 classes
    cm = cm[::-1]
    cm = pd.DataFrame(cm, columns=columns, index=index[::-1])

    fig = ff.create_annotated_heatmap(z=cm.values, x=list(cm.columns), y=list(cm.index), 
                                      colorscale='ice', showscale=True, reversescale=True)
    fig.update_layout(width=700, height=500, title='Confusion Matrix', font_size=16)
    pyo.plot(fig)

plot_confusion_matrix(cm)


results = pd.DataFrame(data={'y_pred':y_pred,'y_test':y_test})

errors = results['y_pred'] != results['y_test']
results[errors]
errors_idx = list(results[errors].index)
errors_idx

results.loc[errors_idx]

plt.figure(figsize=(10,10))
for idx, error_idx in enumerate(errors_idx[:4]):
    image = X_test[error_idx].reshape(8,8)
    plt.subplot(2, 4, idx+1)
    plt.axis('off')
    plt.imshow(image, cmap='Greys')
    plt.title(f"True {results.loc[error_idx, 'y_test']} Prediction: {results.loc[error_idx, 'y_pred']}")
    
