## eaLearn


## About

Building out a machine learing learning library from scratch as a learning experience. The library will include the most common ML algorithms and will not use external libraries, except numpy.

# Algorithms
- [x] Linear Regression (Lasso, Ridge, ElasticNet)
- [x] Logistic Regression
- [ ] LDA
- [ ] QDA
- [ ] Naive Bayes
- [ ] Regression and Classification Trees
- [ ] Bagging, Random Forests, Boosting (AdaBoost and XGBoost)
- [ ] Neural Networks (MLP, CNN)
- [ ] SVM

# Examples

## Linear Regression
```python
X, y = load_boston(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X,y)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
```

## Lasso
    $ python examples/lasso.py
    
![Lasso Regularization](/examples/lasso.png)
*Lasso*