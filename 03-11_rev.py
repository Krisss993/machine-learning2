import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import sys
from sklearn.preprocessing import StandardScaler


class Perceptron:
    def __init__(self, epochs=100, eta=0.1, is_verbose=True):
        self.epochs=epochs
        self.eta=eta
        self.is_verbose=is_verbose
        
    def predict(self, x):
        y_pred = np.dot(x, self.w)
        return np.where(y_pred > 0, 1,-1)
    
    def get_activation(self, X):
        activation = np.dot(X, self.w)
        return activation
    
    def fit(self, x, y):
        ones = np.ones((x.shape[0],1))
        x_1 = np.append(x, ones, axis=1)
        self.list_of_errors = []
        self.w = np.random.rand(x_1.shape[1])
        for e in range(self.epochs):
            y_pred = self.get_activation(x_1)
            delta_w = self.eta * np.dot((y-y_pred),x_1)
            self.w+=delta_w
            errors = np.sum(np.square((y-y_pred)))
            self.list_of_errors.append(errors)
            if self.is_verbose:
                print(f'Epoch: {e}, Weighs: {self.w}, Errors: {errors}')
        








X = np.array([
    [2,4,20],
    [4,3,-10],
    [5,6,13],
    [5,4,8],
    [3,4,5],
])

y = np.array([1,-1,-1,1,-1])

perc = Perceptron(eta=0.001, epochs=10, is_verbose=True)
perc.fit(X, y)


np.dot(X,[0.78055391,0.83153925,0.66069487])

plt.plot(range(perc.epochs), perc.list_of_errors)




class RANSAC:
    def __init__(self, max_iters=100, treshold=6, min_acc_inliers=100):
        self.max_iters=max_iters
        self.treshold=treshold
        self.min_acc_inliers=min_acc_inliers
        
        self.best_mask = None
        self.best_model = None
        self.best_inliers_count=0
        
    def fit(self, X, y, show_partial_results=False):
        assert X.shape[1] == 1, '1'
        assert X.shape[0] >= self.min_acc_inliers, '>'
            
        data = np.hstack((X[:,0].reshape(-1,1), y.reshape(-1,1)))
        sample_size = 2
        
        for i in range(self.max_iters):
            idx = np.random.choice(len(data), size=sample_size, replace=False)
            points = data[idx]
            a = (points[0,1]-points[1,1])/(points[0,0]-points[1,0]+sys.float_info.epsilon)
            b = points[0,1] - a * points[0,0]
            
            y_pred = a * data[:,0] + b
            this_inliers_mask = np.square(y-y_pred) < self.treshold
            this_inliers_count = np.sum(this_inliers_mask)
            
            better_found = (this_inliers_count > self.best_inliers_count) and (this_inliers_count >= self.min_acc_inliers)
            
            if better_found:
                self.best_model = (a,b)
                self.best_inliers_count = this_inliers_count
                self.best_mask = this_inliers_mask
                
            if show_partial_results:
                line_x = np.arange(X.min(), X.max())[:,np.newaxis]
                line_y = a*line_x+b
                plt.scatter(X[this_inliers_mask], y[this_inliers_mask], color='green')
                plt.scatter(X[~this_inliers_mask], y[~this_inliers_mask], color='red')
                plt.plot(line_x, line_y, color='blue')
                plt.scatter(points[:,0], points[:,1])
                plt.show()
                
                
cols = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS', 'RAD','TAX','PTRATIO','B','LSTAT','MEDV']


housing = pd.read_csv(r'F:/\UdemyMachineLearning/\housing/\housing.data', sep=' +', engine='python', header=None, names=cols)

X=housing['LSTAT'].values.reshape(-1,1)
X
y=housing['MEDV'].values
y

scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

ran=RANSAC()

ran.fit(X_train, y_train, show_partial_results=True)


