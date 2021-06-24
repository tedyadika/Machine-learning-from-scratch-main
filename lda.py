import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

data = datasets.load_iris()
X = data.data
y = data.target


class LDA:
    def  __init__(self, n_components):
        self.n_components = n_components
        self.linear_discriminants = None #stores the eigen vectors

    def fit(self, X, y):
        n_features = X.shape[1] #columns of X 150, 4 size
        class_labels = np.unique(y)
        #calculating S_W and S_B
        #calculating mean of samples
        mean_overall = np.mean(X, axis=0)
        S_W = np.zeros((n_features, n_features))#size 4, 4
        S_B = np.zeros((n_features, n_features)) #4, 4 using the iris dataset

        for c in class_labels:
            X_c = X[y == c]
            mean_c = np.mean(X_c, axis=0)
            S_W += (X_c-mean_c).T.dot(X_c-mean_c) #we transpose first term to maintain our 4,4 shape. 
            #these are basic rules of matrix multiplication
            n_c = X_c.shape[0]
            mean_diff = (mean_c-mean_overall).reshape(n_features, 1)
            S_B += n_c * mean_diff.dot(mean_diff.T)

        
        A = np.linalg.inv(S_W).dot(S_B)
        eigenvalues, eigenvectors = np.linalg.eig(A)
        eigenvectors = eigenvectors.T
        idxs = np.argsort(abs(eigenvalues))[::-1]#start to end with step of minus 1
        eigenvalues = eigenvalues[idxs]
        eigenvectors = eigenvectors[idxs]
        self.linear_discriminants = eigenvectors[0:self.n_components]
    def transform(self, X):
        #projecting the data

        return np.dot(X, self.linear_discriminants.T)

lda = LDA(2)
lda.fit(X,y)
X_projected = lda.transform(X)
print('Original X shape:', X.shape)
print('Transformed X shape:', X_projected.shape)

x1 = X_projected[:, 0]
x2 = X_projected[:, 1]

plt.scatter(x1, x2,
        c=y, edgecolor='none', alpha=0.8,
        cmap=plt.cm.get_cmap('viridis', 3))

plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar()
plt.show()
