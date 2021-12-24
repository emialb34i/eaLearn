# sys hacking for import resultion
import sys
sys.path.insert(0, '/Users/emilioalberini/Desktop/eaLearn')

from sklearn.datasets import make_regression
import matplotlib.pyplot as plt

from eaLearn import LinearRegression, Lasso , Ridge
from eaLearn.utils.data_manipulation import train_test_split


def main():

    # data
    X, y = make_regression(n_samples=100, n_features=1, noise=20)
    model = Ridge()
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)
    
    model.fit(X_train, y_train)
    y_pred_line = model.predict(X)

    print(model.beta_hats)

    m1 = plt.scatter(366 * X_train, y_train, s=10)
    m2 = plt.scatter(366 * X_test, y_test, s=10)
    plt.plot(366 * X, y_pred_line, color='black', linewidth=2, label="Prediction")
    plt.legend((m1, m2), ("Training data", "Test data"), loc='lower right')
    plt.show()

if __name__ == "__main__":
    main()