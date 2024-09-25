import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import warnings
import seaborn as sns
import scipy as sp
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.model_selection import StratifiedKFold
from sklearn.datasets import make_classification
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Lasso
from sklearn.model_selection import LeaveOneOut
from numpy import absolute
from sklearn.model_selection import cross_validate
from sklearn.linear_model import Ridge
import argparse
from pathlib import Path

def plot_difference_mae(name_regression,axs, title, data):
    lasso= data[data['Model_name'].str.contains(name_regression)]
    lasso=lasso[["Model_name","mae_test_20_m","data", "seed", "stratified"]]

    sns.boxplot(x="stratified", y='mae_test_20_m', data=lasso, palette=["lightblue", "lightgreen"], ax=axs)

    custom_ticks= [0,1]
    custom_labels=['Non Stratified', 'Stratified']
    axs.set_xticks(custom_ticks)
    axs.set_xticklabels(custom_labels)
    axs.set_xlabel("")

    axs.set_ylabel("MAE score")
    axs.set_title(title,fontsize=19)
    axs.set_ylim(2,21)
    seed=np.arange(1, 51)
    for i, seed in enumerate(seed):
        seed = lasso[lasso["seed"]==seed]
        zero= float(seed[seed["stratified"]==0]['mae_test_20_m'])
        one= float(seed[seed["stratified"]==1]['mae_test_20_m'])
        y_coord= [zero, one]
        x= [0, 1]
        sns.lineplot( x=x, y=y_coord, color= "gray", ax=axs)
        axs.scatter(x[0],y_coord[0], color= "lightblue")
        axs.scatter(x[1],y_coord[1], color= "lightgreen")

def plot_difference_rmae(name_regression,axs, title, data):
    lasso= data[data['Model_name'].str.contains(name_regression)]
    lasso=lasso[["Model_name","rmse_test_20_m","data", "seed", "stratified"]]
    #lasso = lasso.replace("no", 0)
    #lasso = lasso.replace("yes", 1)
    sns.boxplot(x="stratified", y='rmse_test_20_m', data=lasso, palette=["lightblue", "lightgreen"], ax=axs)

    custom_ticks= [0,1]
    custom_labels=['Non Stratified', 'Stratified']
    axs.set_xticks(custom_ticks)
    axs.set_xticklabels(custom_labels)
    axs.set_xlabel("")

    axs.set_ylabel("RMSE score")
    axs.set_title(title,fontsize=15.5)
    axs.set_ylim(2,15)
    seed=np.arange(1, 51)
    for i, seed in enumerate(seed):
        seed = lasso[lasso["seed"]==seed]
        zero= float(seed[seed["stratified"]==0]['rmse_test_20_m'])
        one= float(seed[seed["stratified"]==1]['rmse_test_20_m'])
        y_coord= [zero, one]
        x= [0, 1]
        sns.lineplot( x=x, y=y_coord, color= "gray", ax=axs)
        axs.scatter(x[0],y_coord[0], color= "lightblue")
        axs.scatter(x[1],y_coord[1], color= "lightgreen")