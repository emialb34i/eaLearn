import numpy as np
from eaLearn.utils import multivariate_normal

class QDA:

    """
    Instead of common covariance matrix we use class specific covariance matrices.
    """

    def fit(self, X, y):
        # m data points, n features
        m, n = X.shape
        self.y_unique, self.y_unique_count = np.unique(y, return_counts=True)
        self.n_classes = len(self.y_unique)
        
        # calculate prior
        self.priors = self.y_unique_count/m
        
        # estimate class means and class cov matrix
        self.mu_ks = np.zeros((self.n_classes, n))
        self.covs = np.zeros((self.n_classes, n, n))
        
        for i, k in enumerate(self.y_unique):
            X_k = X[y == k]
            mu_k = X_k.mean(axis=0)
            self.mu_ks[i] = mu_k

        for i, k in enumerate(self.y_unique):
            self.covs[i] = ((X[y==k]-self.mu_ks[i]).T @ (X[y==k]-self.mu_ks[i]))
            self.covs[i] /= len(X[y==k])-1
        
        return self
        
    def predict(self, X):
        m = len(X)
        # we find the class k that maximizes the posterior
        y_pred = np.empty(m)
        print("y_pred: ", y_pred)
        for i, xi in enumerate(X):
            p_ks = np.empty(self.n_classes)
            # calculate the posterior probablity for each class
            # using class sepcifc covariance matrix
            for j in range(self.n_classes):
                p_x_given_y = multivariate_normal(xi, self.mu_ks[j], self.cov[j])
                p_y_given_x = p_x_given_y*self.priors[j]
                p_ks[j] = p_y_given_x
            # classify observation to class with the highest posterior
            y_pred[i] = self.y_unique[np.argmax(p_ks)]
            
        return y_pred
