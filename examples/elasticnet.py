# sys hacking for import resultion
import sys
sys.path.insert(0, '/Users/emilioalberini/Desktop/eaLearn')

from sklearn.datasets import load_boston
from sklearn.linear_model import ElasticNet as ElasticNetSK

from eaLearn import ElasticNet
from eaLearn.utils.data_manipulation import train_test_split
from eaLearn.utils.data_manipulation import standardize
from eaLearn.utils.data_operation import mean_squared_error


def main():

    # load data, preprocess it, and split it
    X, y = load_boston(return_X_y=True)
    X = standardize(X)
    X_train, X_test, y_train, y_test = train_test_split(X,y)
    
    # init model and fit it to the data
    model = ElasticNet()
    model.fit(X_train, y_train)
    # test out model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"eaLearn Model MSE: {mse}")

    # comapare it to sklearn model
    sk_model = ElasticNetSK()
    sk_model.fit(X_train, y_train)
    # test out model
    y_pred = sk_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"sklearn Model MSE: {mse}")

if __name__ == "__main__":
    main()