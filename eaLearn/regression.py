import numpy as np

"""

Implementation of linear regression algortihms.

    1. LinearRegression - basic linear regression (OLS)
    2. Lasso - linear regression with l1 regularization
    3. Ridge - linear regression with l2 regularization
    4. ElasticNet - linear regression with l1 AND l2 regularizations

"""

class l1_regularization:

    def __init__(self, lam):
        self.lam = lam

    def __call__(self, beta):
        return np.linalg.norm(beta, ord=1) * self.lam

    def grad(self, beta):
        return self.lam * np.sign(beta)

class l2_regularization:

    def __init__(self, lam):
        self.lam = lam
    
    def __call__(self, beta):
        return 0.5 * self.lam * np.linalg.norm(beta, ord=2) 
    
    def grad(self, beta):
        return self.lam * beta

class l1_l2_regularization:

    def __init__(self, lam):
        self.lam = lam

    def __call__(self, beta):
        l1 = np.linalg.norm(beta, ord=1)
        l2 = 0.5 * np.linalg.norm(beta, ord=2) 
        return self.lam * (l1 + l2)
    
    def grad(self, beta):
        l1_grad = np.sign(beta)
        l2_grad = 0.5 * np.linalg.norm(beta, ord=2) 
        return self.lam * (l1_grad + l2_grad)
   

class Regression:

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
        
        # init betas randomly between [-1/D, 1/D]
        self.beta_hats = np.random.uniform(-1/self.D, 1/self.D, size=(self.D, ))

        # gradient descent on squared loss
        self.training_errors = []
        for i in range(self.max_iter):
            y_pred = self.X@self.beta_hats
            # mean squared error
            mse = np.mean((y_pred-self.y)**2 + self.regualiraztion(self.beta_hats))
            self.training_errors.append(mse)
            # grad of mse wrt to betas
            grad = ((y_pred-self.y)@X) + self.regualiraztion.grad(self.beta_hats)
            # gradient descent step
            self.beta_hats = self.beta_hats - self.alpha*grad

    def predict(self, X):
        # insert intercept
        if self.fit_intercept:
            X = np.insert(X, 0, 1, axis=1)
        self.y_pred = X@self.beta_hats
        return self.y_pred



class LinearRegression(Regression):

    def __init__(self, fit_intercept=True):
        self.fit_intercept = fit_intercept
        pass

    def fit(self, X, y):
       
        # if we data does not have intercept we add it
        if self.fit_intercept:
            ones = np.ones(len(X)).reshape(len(X), 1)
            X = np.concatenate((ones, X), axis=1)

        self.X = np.array(X)
        self.y = np.array(y)
        self.N, self.D = self.X.shape
        
        # estimate params (closed form)
        XtX_inverse = np.linalg.inv(self.X.T@self.X)
        Xty = self.X.T@self.y
        self.beta_hats = XtX_inverse@Xty

        # make_predictions
        self.y_hat = np.dot(self.X, self.beta_hats)


    def predict(self, X):
        return super(LinearRegression, self).predict(X)


class Lasso(Regression):

    def __init__(self, lam=1.0, alpha=0.001, fit_intercept=True, max_iter=100):
        self.regualiraztion = l1_regularization(lam)
        super(Lasso, self).__init__(alpha, fit_intercept, max_iter)

class Ridge(Regression):

    def __init__(self, lam=1.0, alpha=0.001, fit_intercept=True, max_iter=100):
        self.regualiraztion = l2_regularization(lam)
        super(Ridge, self).__init__(alpha, fit_intercept, max_iter)

class ElasticNet(Regression):

    def __init__(self, lam=1.0, alpha=0.001, fit_intercept=True, max_iter=100):
        self.regualiraztion = l1_l2_regularization(lam)
        super(ElasticNet, self).__init__(alpha, fit_intercept, max_iter)
    
        



