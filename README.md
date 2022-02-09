# eaLearn

  - [About](#about)
  - [Examples](#examples)
    - [Linear Regression](#linear-regression)
    - [Logistic Regression](#logistic-regression)
    - [Lasso](#lasso)
    - [Ridge](#ridge)
    - [ElasticNet](#elasticnet)
    - [LDA](#lda)
    - [QDA](#qda)
    - [Naive Bayes](#naive-bayes)


## About

Building out a machine learing learning library from scratch as a learning experience. The library will include the most common ML algorithms and will not use external libraries, except numpy. eaLearn contains the following algorithms:

- Linear Regression (lasso, ridge, elasticnet)
- Logistic Regression
- LDA, QDA, Naive Bayes
- Decision and Regression Trees

Each algorithm contains an example script where it is applied on real data. I've include the outputs of most of the example scripts below. The algorithms are not meant to be efficient so do not use them on large datasets.

## Examples

### Linear Regression
```python
X, y = load_boston(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X,y)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
```

### Logistic Regression
```python
X, y = load_boston(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X,y)
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
```

### Lasso
    $ python examples/lasso.py

![Lasso Regularization](/examples/images/lasso.png)

### Ridge
    $ python examples/ridge.py  

![Ridge Regularization](/examples/images/ridge.png)

### ElasticNet
    $ python examples/elasticnet.py  

![ElasticNet Regularization](/examples/images/elastic_net.png)

### LDA
    $ python examples/lda.py

![LDA Decision Boundaries](/examples/images/lda.png)

### QDA
    $ python examples/qda.py

![QDA Decision Boundaries](/examples/images/qda.png)

### Naive Bayes
    $ python examples/qda.py

![Naive Bayes Decision Boundaries](/examples/images/naive_bayes.png)


