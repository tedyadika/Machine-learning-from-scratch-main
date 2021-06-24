import numpy as np
from collections import Counter
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
iris = datasets.load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
#defining the Eucledian distance
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1-x2)**2))
#implementing KNN classifier from scratch
class KNN:
    def __init__(self, k=3): # k is the number of default neighbors
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self,X):
        predicted_labels = [self._predict(x) for x in X]
        return np.array(predicted_labels)

    #helper method to return one sample
    def _predict(self, x):
        #compute the distances between one sample x to all training samples x

        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
        #commpute the k nearest samples
        k_indices = np.argsort(distances)[:self.k] #indices of the k nearest samples
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        #identify the majority vote(most common label)
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]
knn = KNN(k=4)
knn.fit(X_train, y_train)
predictions = knn.predict(X_test)
acc = np.sum(predictions == y_test)/len(y_test)
print(acc)

