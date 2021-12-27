# sys hacking for import resultion
import sys
sys.path.insert(0, '/Users/emilioalberini/Desktop/eaLearn')

import numpy as np
from sklearn.datasets import load_boston
from sklearn.linear_model import LassoCV
import matplotlib.pyplot as plt


from eaLearn import Lasso
from eaLearn.utils import train_test_split
from eaLearn.utils import standardize
from eaLearn.utils import mean_squared_error


def main():

    # load data, preprocess it, and split it
    X, y = load_boston(return_X_y=True)
    X = standardize(X)
    X_train, X_test, y_train, y_test = train_test_split(X,y)
    
    # init model and fit it to the data
    lambdas = np.logspace(-3, 2, num=30)
    weights = []
    for lam in lambdas:
        model = Lasso(lam=lam, learning_rate=0.001)
        model.fit(X_train, y_train)
        weights.append(model.w)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        print(f"eaLearn Model (lambda: {round(lam,3)}) MSE: {round(mse, 3)}")

    # comapare it to sklearn model
    sk_model = LassoCV(alphas=lambdas)
    sk_model.fit(X_train, y_train)
    # test out model
    y_pred = sk_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print()
    print(f"sklearn Model (lambda: {round(sk_model.alpha_,3)}) MSE: {round(mse,3)}")

    # plot results
    plt.plot(lambdas, weights)
    plt.xscale("log")
    plt.xlabel("lambda")
    plt.ylabel("weights")
    plt.title("Lasso coefficients as a function of the regularization")
    plt.axis("tight")
    plt.show()

if __name__ == "__main__":
    main()