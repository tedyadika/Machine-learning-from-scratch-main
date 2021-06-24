import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

X, y = datasets.make_blobs(n_samples=50, n_features=2)
y = np.where(y == 0, -1, 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

class Perceptron:
    def __init__(self, lr = 0.01, n_iters=1000):
        self.lr = lr
        self.n_iters=n_iters
        self.activation_func = self._unit_step_func
        self.weight = None
        self.bias = None
    def  _unit_step_func(self,x):
        return np.where(x>=0, 1, 0)

    def fit(self, X, y):
        n_samples, n_features = X.shape
        #initializing our parameters
        self.weight = np.zeros(n_features)
        self.bias = 0
        y_ = np.array([1 if i > 0 else 0 for i in y])

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weight)+self.bias
                y_predicted = self.activation_func(linear_output)

                update = self.lr*(y_[idx]-y_predicted)
                self.weight += update*x_i
                self.bias += update


    def predict(self, X):
        linear_output = np.dot(X, self.weight) + self.bias
        y_predicted = self.activation_func(linear_output)
        return y_predicted
model = Perceptron()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
def accuracy(y_true, predictions):
    acc = (np.sum(y_true==predictions))/len(y_true)
    return acc
score = accuracy(y_test, predictions)
print (score)
