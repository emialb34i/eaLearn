import numpy as np

"""

Logistic Regression

"""

def sigmoid(X):
    return 1/(1+np.exp(-X))

class LogisticRegression:

    def __init__(self, learning_rate=0.1, fit_intercept=True, max_iter=1500):
        self.learning_rate = learning_rate
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
    
    def fit(self, X, y):
        # if we data does not have intercept we add it
        if self.fit_intercept:
            ones = np.ones(len(X)).reshape(len(X), 1)
            X = np.concatenate((ones, X), axis=1)

        # m training examples, n features
        m, n = X.shape
        
        # init weights randomly
        self.w = np.random.rand(n)

        # gradient descent on cross entropy loss
        for _ in range(self.max_iter):
            y_pred = sigmoid(X@self.w)
            # grad of mse wrt to w
            grad_w = -(y-y_pred)@X*(1/m)
            # gradient descent step
            self.w = self.w - self.learning_rate*grad_w
    
        return self

    def predict(self, X):
        # insert intercept
        if self.fit_intercept:
            X = np.insert(X, 0, 1, axis=1)
        y_pred = np.round(sigmoid(X.dot(self.w))).astype(int)
        return y_pred

