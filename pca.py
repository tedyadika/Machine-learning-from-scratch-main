import numpy as np 
import matplotlib.pyplot as plt
from sklearn import datasets

data = datasets.load_iris()
X = data.data
y = data.target

class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None #eigen vectors
        self.mean = None

    def fit(self, X):
        #mean
        self.mean = np.mean(X, axis=0)
        X = X-self.mean
        #calculate the covariance matrix
        #row = 1 sample, columns = feature
        cov = np.cov(X.T)#documentation reverses the above
        #calculate the eigen vectors and values
        eigenvalues, eigenvectors = np.linalg.eig(cov)

        #sort eigen vectors
        eigenvectors = eigenvectors.T
        idxs = np.argsort(eigenvalues)[::-1] #sorting from start to end in reverse order
        eigenvalues = eigenvalues[idxs]
        eigenvectors = eigenvectors[idxs]

        #store the first n eigen vectors
        self.components = eigenvectors[0:self.n_components]


    def transform(self, X):
        #project the data
        X = X-self.mean
        return np.dot(X, self.components.T)
pca = PCA(2)
pca.fit(X)
X_projected = pca.transform(X)

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
