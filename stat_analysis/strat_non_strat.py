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
from scipy.stats import mannwhitneyu
from scipy.stats import ranksums
from scipy.stats import wilcoxon

p_value_corrected=0.05/160
results=pd.DataFrame(columns=["cohort", "model", "rmse t value", "rmse p value","significant", "mae t value", "mae p value ", "significant"])
model_names=df_average["Model_name"].unique().tolist()

for i, name in enumerate(df_average["data"].unique()):
    # choose a model and make the two data frames with only that specific model --> this chooses the cohort
    df_non_stratified= df_average.loc[df_average["data"]==name]
    df_stratified= df_average_strat.loc[df_average_strat["data"]==name]

    for i, model in enumerate(model_names):
        # iterate through the model names and compare them with RMSE and MAE
            significant=[]
            significant_rmse=[]

            ttest,pvalues= wilcoxon(df_non_stratified.loc[df_non_stratified["Model_name"]== model]["rmse_test_20_m"], df_stratified.loc[df_stratified["Model_name"]==model]["rmse_test_20_m"])

            print(len(df_non_stratified.loc[df_non_stratified["Model_name"]== model]["rmse_test_20_m"]),
                  len(df_stratified.loc[df_stratified["Model_name"]==model]["rmse_test_20_m"]))

            if pvalues < p_value_corrected:
                print(name, model, "the p_value is", pvalues)
                significant_rmse= 1
            if pvalues > p_value_corrected:
                print("NOT significant", name, model, "the p_value is", pvalues)

            ttest_mae,pvalues_mae= wilcoxon(df_non_stratified.loc[df_non_stratified["Model_name"]== model]["mae_test_20_m"], df_stratified.loc[df_stratified["Model_name"]==model]["mae_test_20_m"])

            if pvalues_mae < p_value_corrected:
                print(name,model, "the p_value is", pvalues_mae)
                significant=1
            if pvalues_mae > p_value_corrected:
                print("NOT significant", name,model, "the p_value is", pvalues_mae)
            results.loc[len(results)-1,:]= [name, model, ttest, pvalues,significant_rmse, ttest_mae, pvalues_mae, significant]


