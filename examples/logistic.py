# sys hacking for import resultion
import sys
sys.path.insert(0, '/Users/emilioalberini/Desktop/eaLearn')

from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression as LogisticRegressionSK

from eaLearn import LogisticRegression
from eaLearn.utils import train_test_split
from eaLearn.utils import standardize
from eaLearn.utils import accuracy_score


def main():

    # load data, preprocess it, and split it
    X, y = load_breast_cancer(return_X_y=True)
    X = standardize(X)
    X_train, X_test, y_train, y_test = train_test_split(X,y)
    
    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"eaLearn classifier accuracy: {round(accuracy*100,3)}%")

    # comapare it to sklearn model
    sk_model = LogisticRegressionSK()
    sk_model.fit(X_train, y_train)
    # test out model
    y_pred = sk_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"sklearn classifier accuracy: {round(accuracy*100,3)}%")


if __name__ == "__main__":
    main()