#!/usr/bin/env python3

import logging
import sys
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
# from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.preprocessing import PolynomialFeatures

import matplotlib.pyplot as plt

import shutil
from colorama import Fore,  Style
from utils import save, rmse_cv, evaluation,metrics_to_dataframe, save_metrics_report, plot_model_performance



columns = shutil.get_terminal_size().columns

logging.basicConfig(
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        level=logging.INFO,
        datefmt='%H: %M: %S',
        stream=sys.stderr
        )
logger = logging.getLogger(__name__)

logging.info("Loading prepared data")

train = pd.read_csv('dataset/train.csv')
test = pd.read_csv('dataset/test.csv')
val = pd.read_csv('dataset/validation.csv')
X_train = train.iloc[:,:-1]
y_train = train.iloc[:,-1:]

X_test = test.iloc[:,:-1]
y_test = test.iloc[:,-1:]

X = val.iloc[:,:-1]
y = val.iloc[:,-1:]


models = pd.DataFrame(columns=["Model", "MAE", "MSE", "RMSE", "R2 Score", "RMSE (Cross-Validation)"])


logging.info(Fore.WHITE + "Training models")


print(Fore.MAGENTA, end="")
print("Linear regression".center(columns))

lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
predictions = lin_reg.predict(X_test)

mae, mse, rmse, r_squared = evaluation(y_test, predictions)

rmse_cross_val = rmse_cv(lin_reg, X, y)
save(lin_reg, 'linear_regression')

new_row = metrics_to_dataframe("LinearRegression", mae, mse, rmse, r_squared, rmse_cross_val)
models = pd.concat([models, new_row])
models.reset_index()

plot_model_performance(y_test, predictions, "linear")

print(Fore.MAGENTA, end="")
print("Lasso regression".center(columns))

lasso = Lasso()
lasso.fit(X_train, y_train)
predictions = lasso.predict(X_test)

mae, mse, rmse, r_squared = evaluation(y_test, predictions)

rmse_cross_val = rmse_cv(lasso, X, y)

save(lasso, 'lasso_regression')

new_row = metrics_to_dataframe("LassoRegression", mae, mse, rmse, r_squared, rmse_cross_val)

models = pd.concat([models, new_row], ignore_index=True)
models.reset_index()
plot_model_performance(y_test, predictions, "lasso")


print(Fore.MAGENTA, end="")
print("Rigde regression".center(columns))

ridge = Ridge()
ridge.fit(X_train, y_train)
predictions = ridge.predict(X_test)

mae, mse, rmse, r_squared = evaluation(y_test, predictions)

rmse_cross_val = rmse_cv(ridge, X, y)

save(ridge, 'ridge_regression')

new_row = metrics_to_dataframe("RidgeRegression", mae, mse, rmse, r_squared, rmse_cross_val)
models = pd.concat([models, new_row], ignore_index=True)
models.reset_index()
plot_model_performance(y_test, predictions, "ridge")

print(Fore.MAGENTA, end="")
print("Elastic Net regression".center(columns))

elastic_net = ElasticNet()
elastic_net.fit(X_train, y_train)
predictions = elastic_net.predict(X_test)

mae, mse, rmse, r_squared = evaluation(y_test, predictions)
rmse_cross_val = rmse_cv(elastic_net, X, y)
save(elastic_net, 'elastic_net')

new_row = metrics_to_dataframe("ElasticNetRegressor", mae, mse, rmse, r_squared, rmse_cross_val)
models = pd.concat([models, new_row], ignore_index=True)
models.reset_index()
plot_model_performance(y_test, predictions, "elastic")

print(Fore.MAGENTA, end="")
print("Random forest regressor".center(columns))


random_forest = RandomForestRegressor(n_estimators=100)
random_forest.fit(X_train, y_train)
predictions = random_forest.predict(X_test)

mae, mse, rmse, r_squared = evaluation(y_test, predictions)
rmse_cross_val = rmse_cv(random_forest, X, y)
save(random_forest, 'random_forest')

new_row = metrics_to_dataframe("RandomForestRegressor", mae, mse, rmse, r_squared, rmse_cross_val)
models = pd.concat([models, new_row], ignore_index=True)
models.reset_index()
plot_model_performance(y_test, predictions, "forest")


print(Fore.MAGENTA, end="")
print("XGBoost regressor".center(columns))


xgb = XGBRegressor(n_estimators=1000, learning_rate=0.01)
xgb.fit(X_train, y_train)
predictions = xgb.predict(X_test)

mae, mse, rmse, r_squared = evaluation(y_test, predictions)
rmse_cross_val = rmse_cv(xgb, X, y)
save(xgb, 'xgb_regressor')


new_row = metrics_to_dataframe("XGBRegressor", mae, mse, rmse, r_squared, rmse_cross_val)
models = pd.concat([models, new_row], ignore_index=True)
models.reset_index()
plot_model_performance(y_test, predictions, "XGB")
logging.info("Saving metrics to report")

logging.info("Model comparison")

models = models.sort_values(by="RMSE (Cross-Validation)")
save_metrics_report(models)
plt.figure(figsize=(12,8))
sns.barplot(x=models["Model"], y=models["RMSE (Cross-Validation)"])
plt.title("Models' RMSE Scores (Cross-Validated)", size=15)
plt.xticks(rotation=30, size=12)
plt.show()

logging.info("Training  finished")

