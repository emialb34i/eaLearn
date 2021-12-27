# sys hacking for import resultion
import sys
sys.path.insert(0, '/Users/emilioalberini/Desktop/eaLearn')

from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression as LinearRegressionSK

from eaLearn import LinearRegression
from eaLearn.utils.data_manipulation import train_test_split
from eaLearn.utils.data_operation import mean_squared_error


def main():

    # load data and split it
    X, y = load_boston(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X,y)
    # init modle and fit it to the data
    model = LinearRegression()
    model.fit(X_train, y_train)
    # test out model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"eaLearn Model MSE: {mse}")

    # comapare it to sklearn model
    sk_model = LinearRegressionSK()
    sk_model.fit(X_train, y_train)
    # test out model
    y_pred = sk_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"sklearn Model MSE: {mse}")

if __name__ == "__main__":
    main()