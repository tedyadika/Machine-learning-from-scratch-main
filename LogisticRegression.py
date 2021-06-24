import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt

class LogisticRegression:

    def __init__(self, lr=0.001, n_iters=1000):
        self.lr = lr
        self.n_iters=n_iters
        self.weights = None
        self.bias = None
    def fit(self, X, y):
        #initializing our parameters
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        #implementing gradient descent

        for _ in range(self.n_iters):
            linear_model = np.dot(X, self.weights) + self.bias
            y_pred = self._sigmoid(linear_model)

            #updating the params
            dw = (1/n_samples)*np.dot(X.T, (y_pred-y))
            db = (1/n_samples)*np.sum(y_pred-y)

            self.weights -= self.lr-dw
            self.bias -= self.lr-db


    def predict(self, X):
        linear_model = np.dot(X, self.weights)+self.bias
        y_predicted = self._sigmoid(linear_model)
        y_predicted_class = [1 if i > 0.5 else 0 for i in y_predicted]
        return y_predicted_class



    def _sigmoid(self, x):
        return 1/(1+np.exp(-x))
#we use the breast cancer dataset for evaluation of the model
bc = datasets.load_breast_cancer()
X, y = bc.data, bc.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#defining our performance metric
def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true==y_pred)/len(y_true)
    return accuracy
log_reg = LogisticRegression(lr=0.0001, n_iters=1000)
log_reg.fit(X_train, y_train)
predictions = log_reg.predict(X_test)
acc = accuracy(y_test, predictions)
print(acc)
