# sys hacking for import resultion
import sys
sys.path.insert(0, '/Users/emilioalberini/Desktop/eaLearn')

import numpy as np
from sklearn.datasets import load_wine
import matplotlib.pyplot as plt

from eaLearn import LDA
from eaLearn.utils import standardize, train_test_split, accuracy_score

def main():
    X, y = load_wine(return_X_y=True)
    # we only use two features because we can graph the boundaries
    X = X[:,:2]
    X = standardize(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y) 
    
    model = LDA()
    model.fit(X, y)
    accuracy = accuracy_score(y_test, model.predict(X_test))
    print(f"eaLearn LDA classifier accuracy: {round(accuracy*100,3)}%")

    # plot decision boundaries
    plt.scatter(X_test[:,0], X_test[:,1], c=y_test, cmap='Set1', s=10)
    nx, ny = 200, 1000
    x_min, x_max = plt.xlim()
    y_min, y_max = plt.ylim()
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, nx), np.linspace(y_min, y_max, ny))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.pcolormesh(xx, yy, Z, zorder=0, cmap='Pastel1', shading='auto')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title('LDA Decision Boundaries')
    plt.show()

if __name__ == '__main__':
    main()