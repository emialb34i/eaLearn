import numpy as np

"""

Logistic Regression

"""

def sigmoid(X):
    return 1/(1+np.exp(-X))

class LogisticRegression:

    def __init__(self, alpha=0.001, fit_intercept=True, max_iter=100):
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter

    def fit(self, X, y):
        # if we data does not have intercept we add it
        if self.fit_intercept:
            ones = np.ones(len(X)).reshape(len(X), 1)
            X = np.concatenate((ones, X), axis=1)

        self.X = np.array(X)
        self.y = np.array(y)
        self.N, self.D = self.X.shape
        
        # init betas randomly
        self.beta_hats = np.random.randn(self.D, )

        # gradient descent on squared loss
        self.training_errors = []
        for i in range(self.max_iter):
            y_pred = sigmoid(self.X@self.beta_hats)
            # mean squared error
            mse = np.mean((y_pred-self.y)**2)
            self.training_errors.append(mse)
            # grad of mse wrt to betas
            grad = ((y_pred-self.y)@X)
            # gradient descent step
            self.beta_hats = self.beta_hats - self.alpha*grad
    
    def predict(self, X):
        # insert intercept
        if self.fit_intercept:
            X = np.insert(X, 0, 1, axis=1)
        self.y_pred = X@self.beta_hats
        return self.y_pred

