import numpy as np

from eaLearn.utils import normal

class NaiveBayes:

    """
    We assume the n predictors are indipendent and drawn from a normal distribution.
    """

    def fit(self, X, y):
        # m data points, n features
        m, n = X.shape
        self.y_unique, self.y_unique_count = np.unique(y, return_counts=True)
        self.n_classes = len(self.y_unique)
        
        # calculate prior
        self.priors = self.y_unique_count/m
        
        # estimate means and var for each predictor in each class
        self.mu_ks = np.zeros((self.n_classes, n))
        self.vars = np.zeros((self.n_classes,n))
        
        for i, k in enumerate(self.y_unique):
            X_k = X[y == k]
            mu_k = X_k.mean(axis=0)
            self.mu_ks[i] = mu_k
            
            for j in range(n):
                self.mu_ks[i,j] = X_k[:,j].mean(axis=0)
                self.vars[i,j] = X_k[:,j].var(axis=0)
        
        return self
        
    def predict(self, X):
        m, n = X.shape
        y_pred = np.empty(m)
        for i, xi in enumerate(X):
            p_ks = np.empty(self.n_classes)
            for j in range(self.n_classes):
                p_x_given_y = 1
                for p in range(n):
                    p_x_given_y *= normal(self.mu_ks[j,p], self.vars[j,p], xi[p])
                p_y_given_x = p_x_given_y*self.priors[j]
                p_ks[j] = p_y_given_x
            # classify observation to class with the highest posterior
            y_pred[i] = self.y_unique[np.argmax(p_ks)]
            
        return y_pred
