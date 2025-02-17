import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge as ridgeRegression
from sklearn.linear_model import Lasso as LassoRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR as SupportVectorRegressor
import lightgbm as lgb


def linear(X_train, y_train, X_test):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model.predict(X_test)

def ridge(X_train, y_train, X_test):
    model = ridgeRegression(alpha=0.01)
    model.fit(X_train, y_train)
    return model.predict(X_test)

def lasso(X_train, y_train, X_test):
    model = LassoRegression(alpha=0.2)
    model.fit(X_train, y_train)
    return model.predict(X_test)

def svr(X_train, y_train, X_test):
    model = SupportVectorRegressor(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)
    model.fit(X_train, y_train)
    return model.predict(X_test)

def random_forest(X_train, y_train, X_test):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model.predict(X_test)

def lightgbm(X_train, y_train, X_test):
    train_data = lgb.Dataset(X_train, label=y_train)
    params = {
    'objective': 'regression',  # For regression tasks
    'metric': 'mape',           # Root Mean Squared Error
    'boosting_type': 'gbdt',    # Gradient Boosting Decision Tree
    'learning_rate': 0.1,       # Learning rate
    'num_leaves': 31,           # Number of leaves in one tree
    'verbose': -1               # Suppress warning messages
    }
    gbm = lgb.train(params, train_data, num_boost_round=100)
    return gbm.predict(X_test, num_iteration=gbm.best_iteration)

