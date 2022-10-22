#!/usr/bin/env python3

import joblib
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from colorama import Fore,  Style
from mpl_toolkits.mplot3d import Axes3D
import shutil
columns = shutil.get_terminal_size().columns

def save(model, type_model):
    """
    Saving model..
    """
    filename = 'models/' + type_model + '.joblib'
    joblib.dump(model, filename)
    print("### model persisted ####" )


def rmse_cv(model, X, y):
    rmse = np.sqrt(-cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv=5)).mean()
    print("RMSE Cross-Validation:", rmse)
    return rmse

def evaluation(y, predicitons):
    mae = mean_absolute_error(y, predicitons)
    mse = mean_squared_error(y, predicitons)
    rmse = np.sqrt(mean_squared_error(y, predicitons))
    r_squared = r2_score(y, predicitons)
    print(Fore.CYAN, end="")
    print("MAE:", mae)
    print("MSE:", mse)
    print("RMSE:", rmse)
    print("R2 Score:", r_squared)
    print("-"*30)
    return mae, mse, rmse, r_squared

def metrics_to_dataframe(model, mae, mse, rmse, r2_score, rmse_cross_val):
    row = {"Model": [model],
           "MAE": [mae],
           "MSE": [mse],
           "RMSE": [rmse],
           "R2 Score": [r2_score],
           "RMSE (Cross-Validation)": [rmse_cross_val]}
    df = pd.DataFrame.from_dict(row)
    print("Returning dataframe")
    return df

def save_metrics_report(dataframe):
    with open('report.txt', 'w') as report_file:
        print(dataframe, file=report_file)


def plot_model_performance(y, predictions, model):
    fig, ax = plt.subplots()
    fig.set_figheight(8)
    fig.set_figwidth(8)
    sns.regplot(x=predictions, y=y, ax=ax)
    ax.set_xlabel('Predicted price')
    ax.set_ylabel('Real price')
    ax.set_title('Behavior of model prediction')
    plt.show()
    fig.savefig('images/{}_behavior.png'.format(model))

