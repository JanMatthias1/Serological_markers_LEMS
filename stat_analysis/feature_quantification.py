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

data = pd.read_csv(r'')
marker = [x for x in data.columns if x.endswith("_mean")]

modified_list_range = [item.replace('_mean', '') for item in marker]
feature=pd.DataFrame(columns=["index"])
feature["index"]=modified_list_range

list=["mean", "median","min", "max", "range", "measured_7"]

for i, name in enumerate(list):
    path=""
    value= f""

    df_range = pd.read_csv(path + value)

    name_data= df_range["Unnamed: 0"].tolist()
    if name == "measured_7":
        print(i, name)
        print(name_data)
        modified_list_range = [item.replace(f'_{"times_measured_7"}', '') for item in name_data]
        print(modified_list_range)

    else:
        modified_list_range = [item.replace(f'_{name}', '') for item in name_data]

    features_range=pd.DataFrame(columns=["index", f"feature_seed_{name}"])
    features_range["index"]=modified_list_range
    features_range[f"feature_seed_{name}"]=df_range["feature_selected"]

    feature=pd.merge(feature, features_range , how='outer', on="index")