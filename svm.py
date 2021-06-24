import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

X, y = datasets.make_blobs(n_samples=50, n_features=2)
y = np.where(y == 0, -1, 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
class SVM:

    def __init__(self, lr = 0.0001, lambda_params=0.01, n_iters=1000):
        self.lr = lr
        self.lambda_param = lambda_params
        self.n_iters = n_iters
        self.w = None
        self.b = None
    def fit(self, X, y):
        y_ = np.where(y <= 0, -1, 1)
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y_[idx] *(np.dot(x_i, self.w)-self.b) >= 1
                if condition:
                    self.w -= self.lr*(2*self.lambda_param*self.w)
                else:
                    self.w -= self.lr*(2*self.lambda_param*self.w-np.dot(x_i, y_[idx]))
                    self.b -= self.lr * y_[idx]
    def predict(self, X):
        linear_output = np.dot(X, self.w) - self.b
        return np.sign(linear_output)
clf = SVM()
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
def acc(y_true, predictions):
    accuracy = (np.sum(predictions == y_true))/len(y_true)
    return accuracy
score = acc(y_test, predictions)
print(score)
