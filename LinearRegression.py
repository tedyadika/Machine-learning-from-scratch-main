import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
X, y = datasets.make_regression(n_samples=1000, n_features=1, noise=20, random_state=4)
X_train,X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
#implementing LinearRegression from Scratch
class LinearRegression:

    def __init__(self, lr=0.001, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None #we start with no weights
        self.bias = None # we also start with no bias
    #The fit function fits our model in the train and test set
    #In this function we will define our loss as well as update our gradients. 
    def fit(self, X, y):
        #passing initial parameters
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features) #starting with weights of zeros of the size ofour features
        self.bias = 0

        for _ in range(self.n_iters):
            y_predicted = np.dot(X, self.weights)+self.bias #y_pred = mx+b
            #calculating the derivatives
            dw = (1/n_samples) * np.dot(X.T, (y_predicted-y)) #multiplying by the transpose of x to find derivative of the weight
            db = (1/n_samples) * np.sum(y_predicted-y) #derivative of the bias

            #updating the weights
            self.weights -= self.lr*dw
            self.bias -= self.lr *db
            
            #essentially the above is the gradient descent

    def predict(self, X):
        y_predicted = np.dot(X, self.weights) + self.bias
        return y_predicted

regressor = LinearRegression()
regressor.fit(X_train, y_train)
predicted_values = regressor.predict(X_test)

def mse(y_true, y_predicted):
   return  np.mean((y_true-y_predicted)**2)
mse_value = mse(y_test, predicted_values)
print(mse_value)
    
