""" Linear models, two part, script

The purpose of this script is to run the various linear models on the data. Here the data is divided based on very acute LEMS score.

    1) In the first part of the script all the functions are present to run the models

    2) In the second part there is the data and depending on the user terminal input different models and
       different dataframes are used

"""

import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
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
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import argparse
from pathlib import Path
from sklearn.linear_model import PoissonRegressor
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import randint
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
import lightgbm as lgb
from lightgbm.sklearn import LGBMClassifier, LGBMRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.feature_selection import SelectPercentile
import random

def get_scores(model,X_train,X_test, y_train,y_test, name,va_lems_test):
    """ Get scores from the respective model

            Parameters
            -------
            :param model: regression model thatt is being used
            :param X_train: feature rain set of the specific seed
            :param X_test: feature test set of the specific seed
            :param y_train: label train set of the specific seed
            :param y_test: label test set of the specific seed
            :param name: Name of the model, that will be printed
            :param va_lems_test: very acute score, test set

            Returns
            -------
            print of the score
        """

    y_pred_train =model.predict(X_train)
    y_pred_test =model.predict(X_test)

    not_modified_values= y_pred_test.copy()
    for i in range(len(y_test)):
        if y_pred_test[i] > 50:
            y_pred_test[i] = 50
            print("changed score, test",not_modified_values[i], "to", va_lems_test.iloc[i], "correct y values", y_test.iloc[i],model )

    r2_test = r2_score(y_test, y_pred_test)
    rmse_test = mean_squared_error(y_test, y_pred_test,squared=False)

    r2_train = r2_score(y_train, y_pred_train)
    rmse_train = mean_squared_error(y_train, y_pred_train,squared=False)

    # this is done with model predict
    print('Training set score', name, ': R2 score: {:.3f}, RMSE: {:.3f}'.format(r2_train, rmse_train))
    print('Test set score', name, ': R2 score: {:.3f}, RMSE: {:.3f}'.format(r2_test, rmse_test))
def r2_table(predicted_score_train,actual_scores_train,predicted_score, actual_scores, model_name, df_r2_performance, best_parameter, va_lems_test):
    """ Get scores of the model and cross validation scores and create a tabke

            Parameters
            -------
            :param predicted_score: combined predicted score from the two models (one on LEMS very acute =0, other on LEMS va >0)
            :param actual_scores: actual c_lems
            :param model_name: name of the model for the table
            :param df_r2_performance: data frame where the values will be inserted
            :param best_parameter: if a lasso, ridge regression, best parameters
            :param va_lems_test: very acute score, test set

            Returns
            -------
            data frame with the r-squared and RMSE scores from the model
        """

    r2_train = r2_score(actual_scores_train, predicted_score_train)
    rmse_train = mean_squared_error(actual_scores_train, predicted_score_train, squared=False)
    mae_train = mean_absolute_error(actual_scores_train, predicted_score_train)
    eval_metric_train = [r2_train, rmse_train, mae_train]

    r2_= r2_score(actual_scores, predicted_score)
    rmse  = mean_squared_error(actual_scores, predicted_score,squared=False)
    mae= mean_absolute_error(actual_scores, predicted_score)
    eval_metrics = [r2_, rmse, mae]

    # score correction
    predicted_score[predicted_score < 0] = 0
    predicted_score[predicted_score > 50] = 50

    predicted_score_train[predicted_score_train < 0] = 0
    predicted_score_train[predicted_score_train > 50] = 50

    r2_train_m = r2_score(actual_scores_train, predicted_score_train)
    rmse_train_m = mean_squared_error(actual_scores_train, predicted_score_train, squared=False)
    mae_train_m = mean_absolute_error(actual_scores_train, predicted_score_train)
    eval_metric_train_m = [r2_train_m, rmse_train_m, mae_train_m]

    r2_m = r2_score(actual_scores, predicted_score)
    rmse_m = mean_squared_error(actual_scores, predicted_score, squared=False)
    mae_m = mean_absolute_error(actual_scores, predicted_score)
    eval_metrics_modified = [r2_m, rmse_m, mae_m]

    print("non modified ",eval_metrics)
    print("modifief",eval_metrics_modified)
    df_r2_performance.loc[len(df_r2_performance)-1,:]= [model_name]+ eval_metric_train + eval_metrics + eval_metric_train_m + eval_metrics_modified + [best_parameter] # inserting the model name and the evaluation metrics
    #the .loc here enables us to filter the last row of the data frame
    # df_seed_performance = pd.DataFrame(columns=['Model_name', "r2", "rmse","mae","r2_m", "rmse_m","mae_m", "best_parameter", "seed"])

    return(df_r2_performance)
def add_features_all(model, model_name, seed, table):
    """
        Function to add feature importance to the respective data frame

        Parameters
        -------
        :param model: model that was trained
        :param model_name: imputed by the user, name on column
        :param seed: seed iteration (part of the column name)
        :param table: data frame to append new column

        Returns
        -------
        Data frame with an added column with the feature importance of that seed

        """
    if model_name=="LR_0" or model_name=="LR_25":
        table.loc[:,model_name + "_" + str(seed)]= model.coef_
        return table
    if model_name=="Lasso_0" or model_name=="Lasso_25" or model_name=="Ridge_0" or model_name=="Ridge_25":
        table[model_name + "_" + str(seed)]= model.best_estimator_.coef_
        return table
    if model_name=="RFR_0" or  model_name=="RFR_25"  or model_name=="GBR_0" or model_name=="GBR_25" or  model_name=="XGB_0" or model_name=="XGB_25" or model_name=="LGBM_0"or model_name=="LGBM_25" :
        table[model_name + "_" + str(seed)]= model.best_estimator_.feature_importances_
        return table
    if model_name== "SVR_0" or model_name=="SVR_25":
        table[model_name + "_" + str(seed)]= model.best_estimator_.coef_.reshape(-1, 1)
        return table
def difference_in_scores(predicted_score_2, actual_scores, index, score_difference_test, seed, model_name, va_lems_train, va_lems_test,path,seed_iteration):
    """ Difference in scores, from true vs predicted q6 lems values

            Function creates data frames for the test and train set, with the va_lems score, q6 score,
            predicted score from the model and difference between the predicted and q6 score. Additionally,
            creates an image with the distribution of the values (histogram).

            Parameters
            -------
            :param predicted_score: combined predicted score from the two models (one on LEMS very acute =0, other on LEMS va >0)
            :param actual_scores: actual c_lems
            :param index: index from the actual and predicted score subjects
            :param score_difference_test: data frame for difference of score/predicted values for the test set
            :param seed: seed iteration
            :param model_name: name of the model, can be customised
            :param va_lems_train: va_lems values from the train set
            :param va_lems_test: va_lems values from the test set
            :param data_type: specified by the user (for example: mean, max, min, range, abnormal,..), used for saving image with the correct name (here it is the name of data + seed iteration)
            :param path: where to save the output image
            :param seed_iteration: for the histogram output name

            Returns
            ------
            Two tables (score_difference_train and score_difference_test) and immage of the distribution of the particular seed
        """

    #train_difference = abs(y_pred_train - y_train)
    rmse_test = mean_squared_error(actual_scores, predicted_score_2, squared=False)

    y_pred_test_original = predicted_score_2.copy()

    predicted_score_2[predicted_score_2 < 0] = 0
    predicted_score_2[predicted_score_2 > 50] = 50

    test_difference_m= abs(predicted_score_2 - actual_scores)
    test_difference  = abs(y_pred_test_original - actual_scores)
    temporary_data_set_test = pd.DataFrame()

    temporary_data_set_test.loc[:, model_name + "_" + "index" + "_" + str(seed)] = index
    temporary_data_set_test.loc[:, model_name + "_" + "difference" + "_" + str(seed)] = test_difference.tolist()
    temporary_data_set_test.loc[:, model_name + "_" + "difference_modified" + "_" + str(seed)] = test_difference_m.tolist()
    temporary_data_set_test.loc[:, model_name + "_" + "va_lems" + "_" + str(seed)] = va_lems_test.tolist()
    temporary_data_set_test.loc[:, model_name + "_" + "q6_lems" + "_" + str(seed)] = actual_scores.tolist()
    temporary_data_set_test.loc[:, model_name + "_" + "y_pred_test" + "_" + str(seed)] = y_pred_test_original.tolist()
    temporary_data_set_test.loc[:, model_name + "_" + "y_pred_test_modified" + "_" + str(seed)] = predicted_score_2.tolist()

    score_difference_test = pd.concat([score_difference_test, temporary_data_set_test], axis=1)

    return (score_difference_test)
def linear_regression_RepeatedKFold (X_train_0, X_test_0, y_train_0, y_test_0, X_train, X_test, y_train, y_test,data_type,seed,path, df_r2_performance,va_lems_train,va_lems_test,score_difference_test,dfs):
    """ Linear regression function

        Parameters:
        --------
        :param X_train_0: train data set, va_lems == 0
        :param X_test_0: test data set, va_lems == 0
        :param y_train_0: label train data set, va_lems == 0
        :param y_test_0: label test data set, va_lems == 0
        :param X_train: train data set, va_lems > 0
        :param X_test: test data set, va_lems > 0
        :param y_train: label train data set, va_lems > 0
        :param y_test: label test data set, va_lems > 0
        :param data_type: specified by the user (for example: mean, max, min, range, abnormal,..), used for saving image with the correct name
        :param seed: seed iteration
        :param path: path for export
        :param df_r2_performance: data frame where the performance values will be inserted
        :param va_lems_train: va_lems values from the train set
        :param va_lems_test: va_lems values from the test set
        :param score_difference_test: data frame with the test score difference between predicted and acutal values
        :param dfs: dictionary containing the feature importance data frames

        Returns:
        ------
        df_r2_performance table and feature importance tables. Aditionally, plots the predicted values against the actual values and value distribution
    """
    # training the va_lems == 0 model
    np.random.seed(42)
    model_LR_0 = LinearRegression()
    model_LR_0.fit(X_train_0, y_train_0)

    y_pred_test_0 = model_LR_0.predict(X_test_0)

    np.random.seed(42)
    # training the va_lems > 0 model
    model_LR_25 = LinearRegression()
    model_LR_25.fit(X_train, y_train)

    y_pred_test_over = model_LR_25.predict(X_test)

    # concatenate the two y test scores (predicted, actual and index)
    predicted_score = np.concatenate([y_pred_test_0, y_pred_test_over])
    actual_scores = np.concatenate([y_test_0, y_test])
    index = y_test_0.index.to_list() + y_test.index.to_list()

    y_pred_train_0 = model_LR_0.predict(X_train_0)
    y_pred_train_over = model_LR_25.predict(X_train)
    predicted_score_train = np.concatenate([y_pred_train_0, y_pred_train_over])
    actual_scores_train = np.concatenate([y_train_0, y_train])

    predicted_score_2 = predicted_score.copy()
    seed_iteration = data_type + str(seed)

    score_difference_test = difference_in_scores(predicted_score_2, actual_scores, index, score_difference_test, seed,
                                                 "LR_two_factor", va_lems_train, va_lems_test, path, seed_iteration)

    df_r2_performance = r2_table(predicted_score_train,actual_scores_train,predicted_score, actual_scores, "LR_two_factor", df_r2_performance, 0, va_lems_test)

    pairwise_plot(predicted_score, actual_scores, "LR_two_factor", seed_iteration, path)


    dfs["feature_importance_LR_0"] [seed]= add_features_all(model_LR_0, "LR_0", seed, dfs["feature_importance_LR_0"][seed])

    dfs["feature_importance_LR_25"][seed] = add_features_all(model_LR_25, "LR_25", seed, dfs["feature_importance_LR_25"][seed])

    return df_r2_performance, score_difference_test,dfs
def Lasso_Regression_RepeatedKFold (X_train_0, X_test_0, y_train_0, y_test_0, X_train, X_test, y_train, y_test,data_type,seed,path, df_r2_performance,va_lems_train,va_lems_test,score_difference_test,dfs):
    """ Lasso regression function

        Parameters:
        --------
        :param X_train_0: train data set, va_lems == 0
        :param X_test_0: test data set, va_lems == 0
        :param y_train_0: label train data set, va_lems == 0
        :param y_test_0: label test data set, va_lems == 0
        :param X_train: train data set, va_lems > 0
        :param X_test: test data set, va_lems > 0
        :param y_train: label train data set, va_lems > 0
        :param y_test: label test data set, va_lems > 0
        :param data_type: specified by the user (for example: mean, max, min, range, abnormal,..), used for saving image with the correct name
        :param seed: seed iteration
        :param path: path for export
        :param df_r2_performance: data frame where the performance values will be inserted
        :param va_lems_train: va_lems values from the train set
        :param va_lems_test: va_lems values from the test set
        :param score_difference_test: data frame with the test score difference between predicted and acutal values
        :param dfs: dictionary containing the feature importance data frames

        Returns:
        ------
        df_r2_performance table and feature importance tables. Aditionally, plots the predicted values against the actual values and value distribution
    """

    hyper_params ={'alpha': np.arange(0.5, 1500, 0.5)}

    # training the va_lems == 0 model
    model_L_0 = Lasso(random_state=42)
    model_L_0 = GridSearchCV(model_L_0, hyper_params, cv=5, scoring="neg_root_mean_squared_error", n_jobs=-1)
    model_L_0.fit(X_train_0, y_train_0)

    print("lasso best estimator", model_L_0.best_estimator_)
    y_pred_test_0 = model_L_0.predict(X_test_0)

    # training the va_lems > 0 model
    model_L_25 = Lasso(random_state=42)
    model_L_25 = GridSearchCV(model_L_25, hyper_params, cv=5, scoring="neg_root_mean_squared_error", n_jobs=-1)
    model_L_25.fit(X_train, y_train)

    y_pred_test_over = model_L_25.predict(X_test)

    # concatenate the two y test scores (predicted, actual and index)
    predicted_score = np.concatenate([y_pred_test_0, y_pred_test_over])
    actual_scores = np.concatenate([y_test_0, y_test])
    index = y_test_0.index.to_list() + y_test.index.to_list()

    # train data
    y_pred_train_0 = model_L_0.predict(X_train_0)
    y_pred_train_over = model_L_25.predict(X_train)
    predicted_score_train = np.concatenate([y_pred_train_0, y_pred_train_over])
    actual_scores_train = np.concatenate([y_train_0, y_train])

    predicted_score_2 = predicted_score.copy()
    seed_iteration = data_type + str(seed)

    score_difference_test = difference_in_scores(predicted_score_2, actual_scores, index, score_difference_test, seed,
                                                 "Lasso_two_factor", va_lems_train, va_lems_test, path, seed_iteration)

    df_r2_performance = r2_table(predicted_score_train,actual_scores_train,predicted_score, actual_scores, "Lasso_two_factor", df_r2_performance, 0, va_lems_test)

    pairwise_plot(predicted_score, actual_scores, "Lasso_two_factor", seed_iteration, path)

    dfs["feature_importance_Lasso_0"][seed] = add_features_all(model_L_0, "Lasso_0", seed, dfs["feature_importance_Lasso_0"][seed])

    dfs["feature_importance_Lasso_25"][seed] = add_features_all(model_L_25, "Lasso_25", seed, dfs["feature_importance_Lasso_25"][seed])

    return df_r2_performance, score_difference_test,dfs
def Ridge_Regression_RepeatedKFold (X_train_0, X_test_0, y_train_0, y_test_0, X_train, X_test, y_train, y_test,data_type,seed,path, df_r2_performance,va_lems_train,va_lems_test,score_difference_test,dfs):
    """ Ridge regression function

        Parameters:
        --------
        :param X_train_0: train data set, va_lems == 0
        :param X_test_0: test data set, va_lems == 0
        :param y_train_0: label train data set, va_lems == 0
        :param y_test_0: label test data set, va_lems == 0
        :param X_train: train data set, va_lems > 0
        :param X_test: test data set, va_lems > 0
        :param y_train: label train data set, va_lems > 0
        :param y_test: label test data set, va_lems > 0
        :param data_type: specified by the user (for example: mean, max, min, range, abnormal,..), used for saving image with the correct name
        :param seed: seed iteration
        :param path: path for export
        :param df_r2_performance: data frame where the performance values will be inserted
        :param va_lems_train: va_lems values from the train set
        :param va_lems_test: va_lems values from the test set
        :param score_difference_test: data frame with the test score difference between predicted and acutal values
        :param dfs: dictionary containing the feature importance data frames

        Returns:
        ------
        df_r2_performance table and feature importance tables. Aditionally, plots the predicted values against the actual values and value distribution
    """
    hyper_params = {'alpha': np.arange(0.5, 1500, 0.5)}

    # training the va_lems == 0 model
    model_R_0 = Ridge(random_state=42)
    model_R_0 = GridSearchCV(model_R_0, hyper_params, cv=5, scoring="neg_root_mean_squared_error", n_jobs=-1)
    model_R_0.fit(X_train_0, y_train_0)

    y_pred_test_0 = model_R_0.predict(X_test_0)

    # training the va_lems > 0 model
    model_R_25 = Ridge(random_state=42)
    model_R_25 = GridSearchCV(model_R_25, hyper_params, cv=5, scoring="neg_root_mean_squared_error", n_jobs=-1)
    model_R_25.fit(X_train, y_train)

    y_pred_test_over = model_R_25.predict(X_test)

    # concatenate the two y test scores (predicted, actual and index)
    predicted_score = np.concatenate([y_pred_test_0, y_pred_test_over])
    actual_scores = np.concatenate([y_test_0, y_test])
    index = y_test_0.index.to_list() + y_test.index.to_list()

    y_pred_train_0 = model_R_0.predict(X_train_0)
    y_pred_train_over = model_R_25.predict(X_train)
    predicted_score_train = np.concatenate([y_pred_train_0, y_pred_train_over])
    actual_scores_train = np.concatenate([y_train_0, y_train])

    seed_iteration = data_type + str(seed)
    predicted_score_2 = predicted_score.copy()

    score_difference_test = difference_in_scores(predicted_score_2, actual_scores, index, score_difference_test, seed,
                                                 "Ridge_two_factor", va_lems_train, va_lems_test, path, seed_iteration)

    df_r2_performance = r2_table(predicted_score_train,actual_scores_train,predicted_score, actual_scores, "Ridge_two_factor", df_r2_performance, 0, va_lems_test)

    pairwise_plot(predicted_score, actual_scores, "Ridge_two_factor", seed_iteration, path)

    dfs["feature_importance_Ridge_0"][seed] = add_features_all(model_R_0, "Ridge_0", seed, dfs["feature_importance_Ridge_0"][seed])

    dfs["feature_importance_Ridge_25"][seed] = add_features_all(model_R_25, "Ridge_25", seed, dfs["feature_importance_Ridge_25"][seed])

    return df_r2_performance, score_difference_test, dfs
def Random_forest_regressor (X_train_0, X_test_0, y_train_0, y_test_0, X_train, X_test, y_train, y_test,data_type,seed,path, df_r2_performance,va_lems_train,va_lems_test,score_difference_test,dfs):
    """ Random Forest Regressor function

        Parameters:
        --------
        :param X_train_0: train data set, va_lems == 0
        :param X_test_0: test data set, va_lems == 0
        :param y_train_0: label train data set, va_lems == 0
        :param y_test_0: label test data set, va_lems == 0
        :param X_train: train data set, va_lems > 0
        :param X_test: test data set, va_lems > 0
        :param y_train: label train data set, va_lems > 0
        :param y_test: label test data set, va_lems > 0
        :param data_type: specified by the user (for example: mean, max, min, range, abnormal,..), used for saving image with the correct name
        :param seed: seed iteration
        :param path: path for export
        :param df_r2_performance: data frame where the performance values will be inserted
        :param va_lems_train: va_lems values from the train set
        :param va_lems_test: va_lems values from the test set
        :param score_difference_test: data frame with the test score difference between predicted and acutal values
        :param dfs: dictionary containing the feature importance data frames

        Returns:
        ------
        df_r2_performance table and feature importance tables. Aditionally, plots the predicted values against the actual values and value distribution
    """
    param_grid = {
        'n_estimators': [10, 25, 50, 100],
        'max_features': ['sqrt', 'log2', None],
        'max_depth': [3, 6, 9],
        'max_leaf_nodes': [3, 6, 9]}

    # training the va_lems == 0 model
    model_RFR_0 = RandomForestRegressor(random_state=42)
    model_RFR_0 = GridSearchCV(model_RFR_0, param_grid, cv=5, scoring="neg_root_mean_squared_error", n_jobs=-1)
    model_RFR_0.fit(X_train_0, y_train_0)

    y_pred_test_0 = model_RFR_0.predict(X_test_0)

    # training the va_lems > 0 model
    model_RFR = RandomForestRegressor(random_state=42)
    model_RFR = GridSearchCV(model_RFR, param_grid, cv=5, scoring="neg_root_mean_squared_error", n_jobs=-1)
    model_RFR.fit(X_train, y_train)

    y_pred_test_over = model_RFR.predict(X_test)

    # concatenate the two y test scores (predicted, actual and index)
    predicted_score = np.concatenate([y_pred_test_0, y_pred_test_over])
    actual_scores = np.concatenate([y_test_0, y_test])
    index = y_test_0.index.to_list() + y_test.index.to_list()

    y_pred_train_0 = model_RFR_0.predict(X_train_0)
    y_pred_train_over = model_RFR.predict(X_train)
    predicted_score_train = np.concatenate([y_pred_train_0, y_pred_train_over])
    actual_scores_train = np.concatenate([y_train_0, y_train])

    seed_iteration = data_type + str(seed)
    predicted_score_2 = predicted_score.copy()

    # only doing it with the 0 model
    get_scores(model_RFR_0, X_train_0, X_test_0, y_train_0, y_test_0, "RFR_0",va_lems_test)

    score_difference_test= difference_in_scores(predicted_score_2, actual_scores,index, score_difference_test, seed, "RFR_two_factor",va_lems_train,va_lems_test,path,seed_iteration)

    df_r2_performance= r2_table(predicted_score_train,actual_scores_train,predicted_score, actual_scores, "RFR_two_factor", df_r2_performance, 0, va_lems_test)

    pairwise_plot(predicted_score, actual_scores, "RFR_two_factor",seed_iteration, path)

    dfs["feature_importance_RFR_0"][seed] = add_features_all(model_RFR_0, "RFR_0", seed, dfs["feature_importance_RFR_0"][seed])

    dfs["feature_importance_RFR_25"][seed] = add_features_all(model_RFR, "RFR_25", seed, dfs["feature_importance_RFR_25"][seed])

    return df_r2_performance, score_difference_test,dfs
def Support_vector_regressor_linear (X_train_0, X_test_0, y_train_0, y_test_0, X_train, X_test, y_train, y_test,data_type,seed,path, df_r2_performance,va_lems_train,va_lems_test,score_difference_test, dfs):
    """ Random Forest Regressor function

        Parameters:
        --------
        :param X_train_0: train data set, va_lems == 0
        :param X_test_0: test data set, va_lems == 0
        :param y_train_0: label train data set, va_lems == 0
        :param y_test_0: label test data set, va_lems == 0
        :param X_train: train data set, va_lems > 0
        :param X_test: test data set, va_lems > 0
        :param y_train: label train data set, va_lems > 0
        :param y_test: label test data set, va_lems > 0
        :param data_type: specified by the user (for example: mean, max, min, range, abnormal,..), used for saving image with the correct name
        :param seed: seed iteration
        :param path: path for export
        :param df_r2_performance: data frame where the performance values will be inserted
        :param va_lems_train: va_lems values from the train set
        :param va_lems_test: va_lems values from the test set
        :param score_difference_test: data frame with the test score difference between predicted and acutal values
        :param dfs: dictionary containing the feature importance data frames

        Returns:
        ------
        df_r2_performance table and feature importance tables. Aditionally, plots the predicted values against the actual values and value distribution
    """

    np.random.seed(42)
    params = {'epsilon': np.arange(0, 1.5, 0.1)}

    # training the va_lems == 0 model
    model_SVR_0 = SVR(kernel = 'linear')
    model_SVR_0 = GridSearchCV(model_SVR_0, params, cv=5, scoring="neg_root_mean_squared_error", n_jobs=-1)
    model_SVR_0.fit(X_train_0, y_train_0)

    y_pred_test_0 = model_SVR_0.predict(X_test_0)

    # training the va_lems > 0 model
    model_SVR_25 = SVR(kernel = 'linear')
    model_SVR_25 = GridSearchCV(model_SVR_25, params, cv=5, scoring="neg_root_mean_squared_error", n_jobs=-1)
    model_SVR_25.fit(X_train, y_train)

    y_pred_test_over = model_SVR_25.predict(X_test)

    # concatenate the two y test scores (predicted, actual and index)
    predicted_score = np.concatenate([y_pred_test_0, y_pred_test_over])
    actual_scores = np.concatenate([y_test_0, y_test])
    index = y_test_0.index.to_list() + y_test.index.to_list()

    y_pred_train_0 = model_SVR_0.predict(X_train_0)
    y_pred_train_over = model_SVR_25.predict(X_train)
    predicted_score_train = np.concatenate([y_pred_train_0, y_pred_train_over])
    actual_scores_train = np.concatenate([y_train_0, y_train])

    seed_iteration = data_type + str(seed)
    predicted_score_2 = predicted_score.copy()

    score_difference_test = difference_in_scores(predicted_score_2, actual_scores, index, score_difference_test, seed,
                                                 "SVR_two_factor", va_lems_train, va_lems_test, path, seed_iteration)

    df_r2_performance = r2_table(predicted_score_train,actual_scores_train,predicted_score, actual_scores, "SVR_two_factor", df_r2_performance, 0, va_lems_test)

    pairwise_plot(predicted_score, actual_scores, "SVR_two_factor", seed_iteration, path)

    dfs["feature_importance_SVR_0"][seed] = add_features_all(model_SVR_0, "SVR_0", seed, dfs["feature_importance_SVR_0"][seed])

    dfs["feature_importance_SVR_25"][seed] = add_features_all(model_SVR_25, "SVR_25", seed, dfs["feature_importance_SVR_25"][seed])

    return df_r2_performance,score_difference_test,dfs

def gradient_boost_regressor (X_train_0, X_test_0, y_train_0, y_test_0, X_train, X_test, y_train, y_test,data_type,seed,path, df_r2_performance,va_lems_train,va_lems_test,score_difference_test,dfs):
    """ Gradient Boost regressor function

        Parameters:
        --------
        :param X_train_0: train data set, va_lems == 0
        :param X_test_0: test data set, va_lems == 0
        :param y_train_0: label train data set, va_lems == 0
        :param y_test_0: label test data set, va_lems == 0
        :param X_train: train data set, va_lems > 0
        :param X_test: test data set, va_lems > 0
        :param y_train: label train data set, va_lems > 0
        :param y_test: label test data set, va_lems > 0
        :param data_type: specified by the user (for example: mean, max, min, range, abnormal,..), used for saving image with the correct name
        :param seed: seed iteration
        :param path: path for export
        :param df_r2_performance: data frame where the performance values will be inserted
        :param va_lems_train: va_lems values from the train set
        :param va_lems_test: va_lems values from the test set
        :param score_difference_test: data frame with the test score difference between predicted and acutal values
        :param dfs: dictionary containing the feature importance data frames

        Returns:
        ------
        df_r2_performance table and feature importance tables. Aditionally plots the predicted values against the actual values and value distribution

    """
    hyper_params = {
        'n_estimators': [10, 25, 50, 100],
        'learning_rate': [0.001, 0.01, 0.05],
        'subsample': [0.5, 0.7, 0.8],
        'max_depth': [3,5, 6],
        'min_samples_split': [8, 10, 15],
        'min_samples_leaf': [5, 8, 10]
    }

    # training the va_lems == 0 model
    model_GBR_0 = GradientBoostingRegressor(random_state=42)
    model_GBR_0 = GridSearchCV(model_GBR_0, hyper_params, cv=5, scoring="neg_root_mean_squared_error", n_jobs=-1)
    model_GBR_0.fit(X_train_0, y_train_0)

    y_pred_test_0 = model_GBR_0.predict(X_test_0)

    # training the va_lems > 0 model
    model_GBR_25 = GradientBoostingRegressor(random_state=42)
    model_GBR_25 = GridSearchCV(model_GBR_25, hyper_params, cv=5, scoring="neg_root_mean_squared_error", n_jobs=-1)
    model_GBR_25.fit(X_train, y_train)

    y_pred_test_over = model_GBR_25.predict(X_test)

    # concatenate the two y test scores (predicted, actual and index)
    predicted_score = np.concatenate([y_pred_test_0, y_pred_test_over])
    actual_scores = np.concatenate([y_test_0, y_test])
    index = y_test_0.index.to_list() + y_test.index.to_list()

    y_pred_train_0 = model_GBR_0.predict(X_train_0)
    y_pred_train_over = model_GBR_25.predict(X_train)
    predicted_score_train = np.concatenate([y_pred_train_0, y_pred_train_over])
    actual_scores_train = np.concatenate([y_train_0, y_train])

    seed_iteration = data_type + str(seed)
    predicted_score_2 = predicted_score.copy()

    score_difference_test = difference_in_scores(predicted_score_2, actual_scores, index, score_difference_test, seed,
                                                 "GBR_two_factor", va_lems_train, va_lems_test, path, seed_iteration)

    df_r2_performance = r2_table(predicted_score_train,actual_scores_train,predicted_score, actual_scores, "GBR_two_factor", df_r2_performance, 0, va_lems_test)

    pairwise_plot(predicted_score, actual_scores, "GBR_two_factor", seed_iteration, path)

    dfs["feature_importance_GBR_0"][seed] = add_features_all(model_GBR_0, "GBR_0", seed, dfs["feature_importance_GBR_0"][seed])

    dfs["feature_importance_GBR_25"][seed] = add_features_all(model_GBR_25, "GBR_25", seed, dfs["feature_importance_GBR_25"][seed])

    return (df_r2_performance,score_difference_test,dfs)
def XGBoost (X_train_0, X_test_0, y_train_0, y_test_0, X_train, X_test, y_train, y_test,data_type,seed,path, df_r2_performance,va_lems_train,va_lems_test,score_difference_test,dfs):
    """ Extreme gradient boosting regressor function

        Parameters:
        --------
        :param X_train_0: train data set, va_lems == 0
        :param X_test_0: test data set, va_lems == 0
        :param y_train_0: label train data set, va_lems == 0
        :param y_test_0: label test data set, va_lems == 0
        :param X_train: train data set, va_lems > 0
        :param X_test: test data set, va_lems > 0
        :param y_train: label train data set, va_lems > 0
        :param y_test: label test data set, va_lems > 0
        :param data_type: specified by the user (for example: mean, max, min, range, abnormal,..), used for saving image with the correct name
        :param seed: seed iteration
        :param path: path for export
        :param df_r2_performance: data frame where the performance values will be inserted
        :param va_lems_train: va_lems values from the train set
        :param va_lems_test: va_lems values from the test set
        :param score_difference_test: data frame with the test score difference between predicted and acutal values
        :param dfs: dictionary containing the feature importance data frames

        Returns:
        ------
        df_r2_performance table and feature importance tables. Aditionally plots the predicted values against the actual values and value distribution
    """
    param_grid = {
        'n_estimators': [10, 25, 50, 100],
        'max_depth': [3, 5,6],
        'eta': [0.001 ,0.01, 0.05],
        'subsample': [0.3, 0.5, 0.9],
        'colsample_bytree': [0.5, 0.9],
        'gamma': [0.2, 0.3 ,0.4, 0.5]
    }

    # training the va_lems == 0 model
    model_XGB_0 = XGBRegressor(random_state=42)
    model_XGB_0 = GridSearchCV(model_XGB_0, param_grid, cv=5, scoring="neg_root_mean_squared_error", n_jobs=-1)
    model_XGB_0.fit(X_train_0, y_train_0)

    y_pred_test_0 = model_XGB_0.predict(X_test_0)

    # training the va_lems > 0 model
    model_XGB_25 = XGBRegressor(random_state=42)
    model_XGB_25 = GridSearchCV(model_XGB_25, param_grid, cv=5, scoring="neg_root_mean_squared_error", n_jobs=-1)
    model_XGB_25.fit(X_train, y_train)

    y_pred_test_over = model_XGB_25.predict(X_test)

    # concatenate the two y test scores (predicted, actual and index)
    predicted_score = np.concatenate([y_pred_test_0, y_pred_test_over])
    actual_scores = np.concatenate([y_test_0, y_test])
    index = y_test_0.index.to_list() + y_test.index.to_list()

    y_pred_train_0 = model_XGB_0.predict(X_train_0)
    y_pred_train_over = model_XGB_25.predict(X_train)
    predicted_score_train = np.concatenate([y_pred_train_0, y_pred_train_over])
    actual_scores_train = np.concatenate([y_train_0, y_train])

    seed_iteration = data_type + str(seed)
    predicted_score_2 = predicted_score.copy()

    score_difference_test = difference_in_scores(predicted_score_2, actual_scores, index, score_difference_test, seed,
                                                 "XGB_two_factor", va_lems_train, va_lems_test, path, seed_iteration)

    df_r2_performance = r2_table(predicted_score_train,actual_scores_train,predicted_score, actual_scores, "XGB_two_factor", df_r2_performance, 0, va_lems_test)

    pairwise_plot(predicted_score, actual_scores, "XGB_two_factor", seed_iteration, path)

    dfs["feature_importance_XGB_0"][seed] = add_features_all(model_XGB_0, "XGB_0", seed, dfs["feature_importance_XGB_0"][seed])

    dfs["feature_importance_XGB_25"][seed] = add_features_all(model_XGB_25, "XGB_25", seed, dfs["feature_importance_XGB_25"][seed])

    return (df_r2_performance,score_difference_test, dfs)

def lightGBM (X_train_0, X_test_0, y_train_0, y_test_0, X_train, X_test, y_train, y_test,data_type,seed,path, df_r2_performance,va_lems_train,va_lems_test,score_difference_test,dfs):
    """ Extreme gradient boosting regressor function

        Parameters:
        --------
        :param X_train_0: train data set, va_lems == 0
        :param X_test_0: test data set, va_lems == 0
        :param y_train_0: label train data set, va_lems == 0
        :param y_test_0: label test data set, va_lems == 0
        :param X_train: train data set, va_lems > 0
        :param X_test: test data set, va_lems > 0
        :param y_train: label train data set, va_lems > 0
        :param y_test: label test data set, va_lems > 0
        :param data_type: specified by the user (for example: mean, max, min, range, abnormal,..), used for saving image with the correct name
        :param seed: seed iteration
        :param path: path for export
        :param df_r2_performance: data frame where the performance values will be inserted
        :param va_lems_train: va_lems values from the train set
        :param va_lems_test: va_lems values from the test set
        :param score_difference_test: data frame with the test score difference between predicted and acutal values
        :param dfs: dictionary containing the feature importance data frames

        Returns:
        ------
        df_r2_performance table and feature importance tables. Aditionally plots the predicted values against the actual values and value distribution
    """
    param_grid = {
        'max_depth': [1, 2],
        'num_leaves': [2, 3],
        'metric': ['l2', 'l1', 'poisson'],
        'min_child_samples': [5],
        'learning_rate': [0.05, 0.1]
    }

    # training the va_lems == 0 model
    model_LGBM_0 = LGBMRegressor(force_col_wise=True,random_state=42, verbose=1)
    model_LGBM_0 = GridSearchCV(model_LGBM_0, param_grid, cv=5, scoring="neg_root_mean_squared_error", n_jobs=-1)
    model_LGBM_0.fit(X_train_0, y_train_0)

    y_pred_test_0 = model_LGBM_0.predict(X_test_0)

    # training the va_lems > 0 model
    model_LGBM_25 = LGBMRegressor(force_col_wise=True,random_state=42, verbose=1)
    model_LGBM_25 = GridSearchCV(model_LGBM_25, param_grid, cv=5, scoring="neg_root_mean_squared_error", n_jobs=-1)
    model_LGBM_25.fit(X_train, y_train)

    y_pred_test_over = model_LGBM_25.predict(X_test)

    # concatenate the two y test scores (predicted, actual and index)
    predicted_score = np.concatenate([y_pred_test_0, y_pred_test_over])
    actual_scores = np.concatenate([y_test_0, y_test])
    index = y_test_0.index.to_list() + y_test.index.to_list()

    y_pred_train_0 = model_LGBM_0.predict(X_train_0)
    y_pred_train_over = model_LGBM_25.predict(X_train)
    predicted_score_train = np.concatenate([y_pred_train_0, y_pred_train_over])
    actual_scores_train = np.concatenate([y_train_0, y_train])

    seed_iteration = data_type + str(seed)
    predicted_score_2 = predicted_score.copy()

    score_difference_test = difference_in_scores(predicted_score_2, actual_scores, index, score_difference_test, seed,
                                                 "LGBM_two_factor", va_lems_train, va_lems_test, path, seed_iteration)

    df_r2_performance = r2_table(predicted_score_train,actual_scores_train,predicted_score, actual_scores, "LGBM_two_factor", df_r2_performance, 0, va_lems_test)

    pairwise_plot(predicted_score, actual_scores, "LGBM_two_factor", seed_iteration, path)

    dfs["feature_importance_LGBM_0"][seed] = add_features_all(model_LGBM_0, "LGBM_0", seed, dfs["feature_importance_LGBM_0"][seed])

    dfs["feature_importance_LGBM_25"][seed] = add_features_all(model_LGBM_25, "LGBM_25", seed, dfs["feature_importance_LGBM_25"][seed])

    return (df_r2_performance,score_difference_test, dfs)
def fit_models (X_train_0, X_test_0, y_train_0, y_test_0, X_train, X_test, y_train, y_test,data_type,seed,path, df_r2_performance, va_lems_train,va_lems_test,score_difference_test,dfs,methods):
    """Fit the various regression models on the data all within one function

        Parameters:
        --------
        :param X_train_0: train data set, va_lems == 0
        :param X_test_0: test data set, va_lems == 0
        :param y_train_0: label train data set, va_lems == 0
        :param y_test_0: label test data set, va_lems == 0
        :param X_train: train data set, va_lems > 0
        :param X_test: test data set, va_lems > 0
        :param y_train: label train data set, va_lems > 0
        :param y_test: label test data set, va_lems > 0
        :param data_type: specified by the user (for example: mean, max, min, range, abnormal,..), used for saving image with the correct name
        :param path: where to save the various tables and images from the function
        :param df_r2_performance: data frame where the performance values will be inserted
        :param seed: seed iteration
        :param va_lems_train: va_lems values from the train set
        :param va_lems_test: va_lems values from the test set
        :param score_difference_test: data frame with the test score difference between predicted and acutal values
        :param dfs: dictionary containing the feature importance data frames
        :param methods: which regression models to run

        Returns
        -------
        df_r2_performance table and feature importance tables of the various models
        """
    for method in methods:
        df_r2_performance, score_difference_test, dfs = method(X_train_0, X_test_0, y_train_0, y_test_0, X_train, X_test, y_train, y_test, data_type, seed, path, df_r2_performance, va_lems_train, va_lems_test, score_difference_test, dfs)

    return df_r2_performance,score_difference_test,dfs

def pairwise_plot(predicted, actual, title,data_type,path):
    """ Plotting of predicted vs actual label values of the test set

        Parameters:
         --------
        :param X_train: feature rain set of the specific seed
        :param X_test: feature test set of the specific seed
        :param y_train: label train set of the specific seed
        :param y_test: label test set of the specific seed
        :param data_type: specified by the user (for example: mean, max, min, range, abnormal,..), used for saving image with the correct name
        :param path: where to save the various tables and images from the function

        Returns
        -------
        Plot of the test set
        """
    '''
    fig, ax = plt.subplots()
    sns.set(style="darkgrid")
    sns.regplot(x=predicted, y=actual,
                scatter_kws=dict(color='black', s=10, alpha=0.5)).set(title=title + data_type,
                                                                      xlabel='Predicted LEMS values',
                                                                      ylabel='True value LEMS values')

    path = path + "/Pairwise_plot/"
    # Create the output folder
    Path(path).mkdir(exist_ok=True)

    location = path + title + data_type + ".png"
    plt.savefig(location)
    # clear current image for the next one
    plt.clf()
    plt.close()  # due to memory warning of having all the pictures open
    '''

def seed_multiple_times (data_type, path,df_seed_performance,score_difference_test,methods, straify,name_data, model_names, data,feature_selection,feature_amount):
    """ Running the models with different seeds

            Parameters:
            -------
            :param X_0: features from the data set (va_lems=0), already filtered depending on correlation values
            :param y_0: labels from the data set (va_lems=0)
            :param X: features from the data set (va_lems > 0), already filtered depending on correlation values
            :param y: labels from the data set (va_lems > 0)
            :param data_type: specified by the user (for example: mean, max, min, range, abnormal,..), used for saving image with the correct name
            :param path: where to save the various tables and images from the function
            :param df_seed_performance: summary table of the performance of all models within different seeds
            :param dfs: dictionary containing the feature importance data frames
            :param methods: which regression models to run

            Returns
            -------
            df_seed_importance, and feature importance with mean values and standard deviation
        """

    variable = [i for i in range(1, 51)]

    reg_strengths = ['0', '25']

    dfs = {f'feature_importance_{model}_{strength}': {} for model in model_names for strength in reg_strengths}

    for seed in variable:

            df_r2_performance = pd.DataFrame(columns = ['Model_name',"r2_train", "rmse_train","mae_train", "r2", "rmse","mae","r2_m_train", "rmse_m_train","mae_m_train",
                                                        "r2_m", "rmse_m","mae_m", "best_parameter"])

            print(df_r2_performance)

            # ---------- FEATURE SELECTION
            if name_data == "mean" or name_data == "median" or name_data == "min" or name_data == "max" or name_data == "range" \
                    or name_data == "measured_7" or name_data == "measured_7_noise":

                # -------  Selecting all features for the specific cohort (mean,....)
                # 1) select features from our data set Mean features selection, selecting features and dropping all null values

                marker_all = [x for x in data.columns if x.endswith(name_data)]  # name data is inserted from below
                fixed_columns = ["age", 'va_lems_imputed', "lems_q6"]
                # only mean and other features needed --> dropping all null values
                data = data[marker_all + fixed_columns]

                # features and target creation
                X = data.drop("lems_q6", axis=1)
                y = data["lems_q6"]

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed,
                                                                    stratify=X["va_lems_imputed"] > 0)

                # -------  scaling X feature selection
                # correlation matrix for X_train with columns (defined above)
                columns = X_train.columns

                # Initialize the StandardScaler
                scaler = StandardScaler()
                # Fit and transform the data
                X_train = scaler.fit_transform(X_train)
                # Convert the scaled data back to a DataFrame
                X_train = pd.DataFrame(X_train, columns=columns)
                # correlation matrix for X_train with columns (defined above)

                if feature_selection == "pearson":
                    # -------  Correlation in matrix
                    # correlation matrix for X_train with columns (defined above)
                    correlation_matrix = X_train[marker_all].corr()

                    correlation_matrix_abs = correlation_matrix.abs()

                    # consider only one part of the correlation matrix
                    upper = correlation_matrix_abs.where(
                        np.triu(np.ones(correlation_matrix_abs.shape), k=1).astype(bool))

                    correlated_features = [column for column in upper.columns if any(upper[column] > 0.7)]

                    # -------  selecting the features for our analysis
                    selected_07 = [ele for ele in marker_all if
                                   ele not in correlated_features]  # excluding features that are correlated
                    # fixed_columns = ["age", 'va_lems_imputed']
                    X_train = X_train[selected_07]
                    X_test = X_test[selected_07]

                    # ------- check correlation (on scaled)
                    correlation_matrix = X_train.corr()
                    correlation_matrix_abs = correlation_matrix.abs()

                    # consider only one part of the correlation matrix
                    upper = correlation_matrix_abs.where(
                        np.triu(np.ones(correlation_matrix_abs.shape), k=1).astype(bool))

                    correlated_features_check = []
                    correlated_features_check = [column for column in upper.columns if any(upper[column] > 0.7)]

                    if len(correlated_features_check) > 0:
                        raise ValueError("Correlated features detected. This is not allowed.")

                    required_features = ["age", "va_lems_imputed"]
                    selected_07 = selected_07 + required_features

                if feature_selection == "MI":
                    # only select out of the serological markers

                    selector = SelectPercentile(lambda X, y: mutual_info_regression(X, y, random_state=42), percentile=float(feature_amount))
                    selector.fit(X_train[marker_all], y_train)

                    # Get the mask of selected features
                    selected_mask = selector.get_support()

                    # Extract the feature names of the selected features
                    selected_features = [feature for feature, selected in zip(marker_all, selected_mask) if selected]

                    if name_data == "measured_7_noise":
                        required_features = ["age", "va_lems_imputed","Noise_measured_7_noise"]
                    else:
                        required_features = ["age", "va_lems_imputed"]


                    selected_07 = selected_features + required_features

                # ----------- final split for task
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed,
                                                                        stratify=X["va_lems_imputed"] > 0)
                X_train = X_train[selected_07]
                X_test = X_test[selected_07]

                # -------  Feature importance data Sets
                # iterating in model_names to access the correct dictionary location --> overall dictionary already created above
                reg_strengths = ['0', '25']
                for model in model_names:
                    for strength in reg_strengths:
                        # Generate a DataFrame (replace this with your actual logic)
                        feature_importance = pd.DataFrame(index=X_train.columns)

                        # Add the DataFrame to the inner dictionary
                        dfs[f'feature_importance_{model}_{strength}'][seed] = feature_importance

            if name_data == "all_foward_feature":

                # features to select from
                marker = data.columns

                features = [col for col in marker if col != "lems_q6"]
                X = data[features]  # Include all your features here
                y = data['lems_q6']

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed,
                                                                    stratify=X["va_lems_imputed"] > 0)

                # -------  scaling X feature selection
                columns = X_train.columns

                scaler = StandardScaler()

                to_scale = [x for x in data.columns if x.endswith(("_mean", "_median", "_min", "_max", "_range", "_times_measured_7"))] + ['age', 'va_lems_imputed']

                columns_encoded = [feature for feature in columns if feature not in to_scale]

                X_train_scaled_columns = scaler.fit_transform(X_train[to_scale])

                X_train = X_train.drop(columns=to_scale)

                X_train_scaled = np.concatenate([X_train_scaled_columns, X_train], axis=1)

                # Convert the scaled data back to a DataFrame
                X_train = pd.DataFrame(X_train_scaled, columns=[to_scale + columns_encoded])

                if feature_selection == "RFS":
                    # -------
                    # Initial setup
                    p_value_threshold = 0.05  # Set your desired threshold
                    selected_features = ["age", "va_lems_imputed"]
                    remaining_features = [feature for feature in features if
                                          feature != "age" and feature != "va_lems_imputed"]

                    # Foward feature selection
                    while True:
                        if not remaining_features:
                            break  # No more features to add

                        # setting up the most_significant_feature and current_best_p_value --> here the most significant feature for that iteration is stored
                        most_significant_feature = None
                        current_best_p_value = float('inf')


                        for feature in remaining_features:
                            # iteratre trough the features
                            current_features = X_train[selected_features + [feature]].astype(np.float64)

                            # Resetting index to align with y_train
                            current_features.reset_index(drop=True, inplace=True)
                            y_train.reset_index(drop=True, inplace=True)

                            model = sm.OLS(y_train, current_features)
                            results = model.fit()
                            p_value = results.pvalues[feature]

                            if p_value < current_best_p_value:
                                # here it initially chooses one and we see if it can be beat
                                current_best_p_value = p_value
                                most_significant_feature = feature

                                print("best feature", most_significant_feature, feature, current_best_p_value)

                        # Check if the most significant feature is below the threshold --> establish above
                        if current_best_p_value < p_value_threshold:
                            # append new selected feature
                            selected_features.append(most_significant_feature)
                            remaining_features.remove(most_significant_feature)


                        else:
                            break  # Stop if no more features below the threshold

                if feature_selection == "MI":
                    remaining_features = [feature for feature in features if
                                          feature != "age" and feature != "va_lems_imputed"]

                    selector = SelectPercentile(lambda X, y: mutual_info_regression(X, y, random_state=42), percentile=float(feature_amount))
                    selector.fit(X_train[remaining_features], y_train)

                    # Get the mask of selected features
                    selected_mask = selector.get_support()

                    # Extract the feature names of the selected features
                    selected_features = [feature for feature, selected in zip(remaining_features, selected_mask) if
                                         selected]

                    required_features = ["age","va_lems_imputed"]

                    selected_features = selected_features + required_features

                # ----------- final split for task
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed,
                                                                            stratify=X["va_lems_imputed"] > 0)
                X_train = X_train[selected_features]
                X_test = X_test[selected_features]

                print("selected features END", selected_features, len(selected_features))

                reg_strengths = ['0', '25']

                to_scale = [x for x in data.columns if x.endswith(("_mean", "_median", "_min", "_max", "_range", "_times_measured_7"))] \
                           + ['age', 'va_lems_imputed']
                columns=X_train.columns

                columns_to_scale = [feature for feature in columns if feature in to_scale]
                columns_encoded = [feature for feature in columns if feature not in to_scale]
                columns_feature_importance= columns_to_scale + columns_encoded

                for model in model_names:
                    for strength in reg_strengths:
                        # Generate a DataFrame (replace this with your actual logic)
                        feature_importance = pd.DataFrame(index=columns_feature_importance)

                        # Add the DataFrame to the inner dictionary
                        dfs[f'feature_importance_{model}_{strength}'][seed] = feature_importance

            if name_data == "abnormal" or name_data=="lems_data" or name_data=="lems_missing_data" or \
                name_data=="range_lems":
                # here the features are already selected by the user
                X = data.drop("lems_q6", axis=1)
                y = data["lems_q6"]

                # -------  Feature importance data Sets

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed,
                                                                    stratify=X["va_lems_imputed"] > 0)
                # iterating in model_names to access the correct dictionary location

                reg_strengths = ['0', '25']
                for model in model_names:
                    for strength in reg_strengths:
                        # Generate a DataFrame (replace this with your actual logic)
                        feature_importance = pd.DataFrame(index=X_train.columns)

                        # Add the DataFrame to the inner dictionary
                        dfs[f'feature_importance_{model}_{strength}'][seed] = feature_importance


            # --------------- DIVISION OF TRAIN SET
            X_train_0 = X_train.loc[X_train["va_lems_imputed"] == 0]
            index_0 = X_train_0.index.to_numpy()
            y_train_0 = y_train.loc[index_0]

            x_train = X_train.loc[X_train["va_lems_imputed"] > 0]
            index_over = x_train.index.to_numpy()
            y_train = y_train.loc[index_over]

            # DIVISION OF TEST SET
            X_test_0 = X_test.loc[X_test["va_lems_imputed"] == 0]
            index_0 = X_test_0.index.to_numpy()
            y_test_0 = y_test.loc[index_0]

            X_test = X_test.loc[X_test["va_lems_imputed"] > 0]
            index_over = X_test.index.to_numpy()
            y_test = y_test.loc[index_over]

            # division of data sets
            va_lems_train= pd.concat([X_train_0["va_lems_imputed"], x_train["va_lems_imputed"]])
            va_lems_test = pd.concat([X_test_0["va_lems_imputed"], X_test["va_lems_imputed"]])

            # dropping va_lems_imputed as all of its values are 0 in the stratified 0 cohort
            #X_train_0 = X_train_0.drop(["va_lems_imputed"], axis=1)
            #X_test_0 = X_test_0.drop(["va_lems_imputed"], axis=1)

            # ------------------ Feature SCALING
            if name_data == "abnormal":
                columns_to_scale = ['age', 'va_lems_imputed']

                sc_0 = StandardScaler()
                X_train_0_scaled = sc_0.fit_transform(X_train_0[columns_to_scale])
                X_test_0_scaled = sc_0.transform(X_test_0[columns_to_scale])

                X_train_encoded_0 = X_train_0.drop(columns=columns_to_scale)
                X_test_encoded_0 = X_test_0.drop(columns=columns_to_scale)

                X_train_0 = np.concatenate([X_train_0_scaled, X_train_encoded_0], axis=1)
                X_test_0 = np.concatenate([X_test_0_scaled, X_test_encoded_0], axis=1)

                sc = StandardScaler()
                X_train_scaled = sc.fit_transform(x_train[columns_to_scale])
                X_test_scaled = sc.transform(X_test[columns_to_scale])

                X_train_encoded = x_train.drop(columns=columns_to_scale)
                X_test_encoded = X_test.drop(columns=columns_to_scale)

                X_train= np.concatenate([X_train_scaled, X_train_encoded], axis=1)
                X_test = np.concatenate([X_test_scaled, X_test_encoded], axis=1)

                print("Only age and va_lems_imputed are scaled ")

            if name_data == "all_foward_feature":
                # insert columns to scale
                to_scale = [x for x in data.columns if x.endswith(("_mean", "_median", "_min", "_max", "_range", "_times_measured_7"))] + ['age', 'va_lems_imputed']
                columns_to_scale = [feature for feature in selected_features if feature in to_scale]
                columns_encoded = [feature for feature in selected_features if feature not in to_scale]

                scaler = StandardScaler()

                sc_0 = StandardScaler()
                X_train_0_scaled = sc_0.fit_transform(X_train_0[columns_to_scale])
                X_test_0_scaled = sc_0.transform(X_test_0[columns_to_scale])

                X_train_encoded_0 = X_train_0.drop(columns=columns_to_scale)
                X_test_encoded_0 = X_test_0.drop(columns=columns_to_scale)

                X_train_0 = np.concatenate([X_train_0_scaled, X_train_encoded_0], axis=1)
                X_test_0 = np.concatenate([X_test_0_scaled, X_test_encoded_0], axis=1)

                sc = StandardScaler()

                X_train_scaled = sc.fit_transform(x_train[columns_to_scale])
                X_test_scaled = sc.transform(X_test[columns_to_scale])

                X_train_encoded = x_train.drop(columns=columns_to_scale)
                X_test_encoded = X_test.drop(columns=columns_to_scale)

                X_train = np.concatenate([X_train_scaled, X_train_encoded], axis=1)
                X_test = np.concatenate([X_test_scaled, X_test_encoded], axis=1)

            if name_data == "mean" or name_data == "median" or name_data == "min" or name_data == "max" or name_data == "range" or name_data == "measured_7" \
                    or name_data == "measured_7_noise" or name_data == "lems_data" or name_data == "lems_missing_data" or name_data == "range_lems":
                sc_0 = StandardScaler()
                X_train_0 = sc_0.fit_transform(X_train_0)
                X_test_0 = sc_0.transform(X_test_0)

                sc = StandardScaler()
                X_train = sc.fit_transform(x_train)
                X_test = sc.transform(X_test)

            df_r2_performance,score_difference_test,dfs= fit_models (X_train_0, X_test_0, y_train_0, y_test_0, X_train, X_test, y_train, y_test,data_type,seed,path, df_r2_performance, va_lems_train,va_lems_test,score_difference_test,dfs,methods)

            df_r2_performance["seed"]= seed

            df_seed_performance= pd.concat([df_seed_performance, df_r2_performance], axis=0) # adding it row by row

    merged_dfs = {}
    reg_strengths = ['0', '25']
    # ------- Merging feature importance data frames to export
    for model in model_names:
        for strength in reg_strengths:
            key = f'feature_importance_{model}_{strength}'
            print("key", key)
            print(dfs[key])

            # Extract the values (DataFrames) from the inner dictionary
            data_frames = list(dfs[key].values())

            # Concatenate the list of DataFrames along the columns
            concatenated_df = pd.concat(data_frames, axis=1, join='outer')
            feature_count=concatenated_df.copy()
            # Calculate mean and standard deviation for each row across seeds
            merged_dfs[key] = concatenated_df
            merged_dfs[key].loc['total_features_seed'] = merged_dfs[key].notnull().sum()
            merged_dfs[key]['importance_mean'] = merged_dfs[key].mean(axis=1)
            merged_dfs[key]['importance_std'] = feature_count.std(axis=1)
            merged_dfs[key]['feature_selected'] = feature_count.notnull().sum(axis=1)

    print(df_seed_performance)
    print(score_difference_test)

    return df_seed_performance,score_difference_test,merged_dfs
def average_seed (df_seed_performance):
    """ Mean and std seed value for the various models

        Parameters:
        -------
        :param df_seed_performance: dataframe of all model scores and different seeds

        Returns:
        -------
        Average performance values and std of all models

        """

    models= df_seed_performance["Model_name"].unique().tolist()
    # models to compute the average, all the model results are in the df_seed_performance data frame
    df_seed_average=pd.DataFrame(columns = ['Model_name',"r2_train", "rmse_train","mae_train", "r2", "rmse","mae","r2_m_train", "rmse_m_train","mae_m_train",
                                                        "r2_m", "rmse_m","mae_m", "best_parameter", "seed"])

    for i in range(len(models)):
        # creating a variable that stores of all column names, without the index
        selected_columns = df_seed_performance.columns[1:]
        # data frame, with the columns of that specific model
        average_values= df_seed_performance.loc[df_seed_performance['Model_name'] == models[i]][selected_columns]
        print(average_values)

        average_values_mean = (average_values.mean()).to_list()
        average_values_std = (average_values.std()).to_list()

        df_seed_average.loc[len(df_seed_average) - 1, :] = [models[i] + "_mean"] + average_values_mean
        df_seed_average.loc[len(df_seed_average) - 1, :] = [models[i] + "_std"] + average_values_std
    return df_seed_average
def which_functions_to_run(info):
    """ Which models to run, based on terminal input (model)

        Parameters:
        -------
        :param info: how many regression models to run (linear --> run linear_regression_RepeatedKfold

        Returns
        -------
        Models to run

    """

    method_mapping = {
        "linear": [linear_regression_RepeatedKFold],
        "lasso": [Lasso_Regression_RepeatedKFold],
        "ridge": [Ridge_Regression_RepeatedKFold],
        "RFR": [Random_forest_regressor],
        "SVR": [Support_vector_regressor_linear],
        "XGB": [XGBoost],
        "GBR": [gradient_boost_regressor],
        "LGBM": [lightGBM],
        "all": [linear_regression_RepeatedKFold, Lasso_Regression_RepeatedKFold, Ridge_Regression_RepeatedKFold,
                Random_forest_regressor, Support_vector_regressor_linear, XGBoost, gradient_boost_regressor, lightGBM]
    }

    return method_mapping.get(info, [])
def data_frames_to_create(info):
    """ Feature importance data frames we want to create

        Parameters:
        -------
        :param info: input from terminal, indicating how many models we want to run. Depending on the amount of
         models the user wants to run, a dataframe is created

        Returns
        -------
        Which feature importance data frames need to be created
    """
    method_mapping = {
        "linear": ['LR'],
        "lasso": ['Lasso'],
        "ridge": ['Ridge'],
        "RFR": ['RFR'],
        "SVR": ['SVR'],
        "XGB": ['XGB'],
        "GBR": ['GBR'],
        "LGBM": ['LGBM'],
        "all": ['LR', 'Lasso', 'Ridge', 'RFR', 'SVR', 'XGB', 'GBR', 'LGBM']
    }

    return method_mapping.get(info, [])
def run_models (results_dir, data,name_data,methods, info,feature_selection,feature_amount):
    """ Run the models, split the data, create the various data frames and export it

        Parameters:
        -------
        :param results_dir: path for output
        :param data: filtered data to run the models on
        :param name_data: mean, range, abnromal, used for title of graphs, tables,..
        :param methods: which models to run
        :param info: info regarding which data frames for feature importance to create (same parameter as which models to run)

        Returns:
        -------
        Feature importance data frames, train/test differences, summary data frame of model performance
        """

    df_seed_performance = pd.DataFrame(columns=['Model_name',"r2_train", "rmse_train","mae_train", "r2", "rmse","mae","r2_m_train", "rmse_m_train","mae_m_train",
                                                        "r2_m", "rmse_m","mae_m", "best_parameter", "seed"])

    model_names = data_frames_to_create(info)

    '''
    # creation of feature importance data frames
    for model in model_names:
        for strength in reg_strengths:
            # as the o cohort, doesn't have the va_lems as a predictor, need two different feature set sizes 
            if strength == "25":
                key = f'feature_importance_{model}_{strength}'
                dfs[key] = pd.DataFrame(index=X.columns)
            else:
                key = f'feature_importance_{model}_{strength}'
                columns = X.drop(["va_lems_imputed"], axis=1)
                dfs[key] = pd.DataFrame(index=columns.columns)
    '''
    score_difference_test = pd.DataFrame()

    path = results_dir + f"/{name_data}/{feature_selection}/{feature_amount}/"
    # Create the output folder
    Path(path).mkdir(exist_ok=True,parents=True)

    # now doing it with random states
    df_seed_performance, score_difference_test_seed, feature_importance = seed_multiple_times(f"_{name_data}_", path, df_seed_performance,
                                                                               score_difference_test,methods, (data["va_lems_imputed"] > 0),name_data,model_names, data,feature_selection,feature_amount)
    # creating the various performance and feature data frames across the various seeds

    df_average_seed = average_seed(df_seed_performance)
    # creating average values across

    # exporting the random states, one with all the values and one with only the averge
    df_seed_performance.to_csv(path + f"seed_values_{name_data}_{info}.csv")
    df_average_seed.to_csv(path + f"seed_average_{name_data}_{info}.csv")
    reg_strengths = ['0', '25']
    # export of the feature importance data frames
    for model in model_names:
        for strength in reg_strengths:
            key = f'feature_importance_{model}_{strength}'
            filename = f"{path}feature_importance_{model}_{strength}_{name_data}_seed.csv"
            feature_importance[key].to_csv(filename)

    score_difference_test_seed.to_csv(path + f"different_scores_mean_{info}.csv")

# importing data sets
data = pd.read_csv(r'')
data_missing= pd.read_csv(r'')
# output path creation, for whole document
results_dir = ''

# Create the output folder
Path(results_dir).mkdir(exist_ok=True)
# ture is euqal to 1, and false is equal to 0
if __name__ == '__main__':

    # initialise your parser
    parser = argparse.ArgumentParser(description="-")
    # Add the parameter positional/optional
    parser.add_argument("operation",
                        help="Which data set should we use?")  # add argument is what we are inputting in the function

    parser.add_argument('model',
                        help='Which regression model to use, options: linear, lasso, ridge, RFR, SVR, XGB, GBR, LGBM, all?')  # add argument is what we are inputting in the function

    parser.add_argument("selection", help="Which feature selection to choose: pearson, RFS, MI")

    parser.add_argument("feature_amount", help="Which feature selection to choose: 5,10,20,30,..., no")
    # Parse the arguments, using this function
    args = parser.parse_args()
    # --> will take the values that I inserted and make one argument
    # Perform operations with dataset1
    if args.operation == "mean":

        # filtering the _mean columns
        marker = [x for x in data.columns if x.endswith("_mean")]

        # adding columns that are necessary
        fixed_columns = ["age", 'va_lems_imputed', "lems_q6"]
        columns = marker + fixed_columns

        # copying the data set, with the columns for mean and age, va_lems and lems_q6
        data = data[columns].copy()
        data = data.dropna(subset=columns)

        methods = which_functions_to_run(args.model)

        run_models(results_dir, data, "mean", methods, args.model,args.selection, args.feature_amount)
        print("mean, done")

    if args.operation == "median":

        # filtering the _mean columns
        marker = [x for x in data.columns if x.endswith("_median")]

        # adding columns that are necessary
        fixed_columns = ["age", 'va_lems_imputed', "lems_q6"]
        columns = marker + fixed_columns

        # copying the data set, with the columns for mean and age, va_lems and lems_q6
        data = data[columns].copy()
        data = data.dropna(subset=columns)

        methods= which_functions_to_run(args.model)

        run_models(results_dir, data, "median", methods,args.model,args.selection, args.feature_amount)

        print("median, done")

    if args.operation == "min":
        # filtering the _mean columns
        marker = [x for x in data.columns if x.endswith("_min")]

        # adding columns that are necessary
        fixed_columns = ["age", 'va_lems_imputed', "lems_q6"]
        columns = marker + fixed_columns

        # copying the data set, with the columns for mean and age, va_lems and lems_q6
        data = data[columns].copy()
        # dropping the nan values in these columns
        data = data.dropna(subset=columns)

        methods = which_functions_to_run(args.model)

        run_models(results_dir, data, "min", methods, args.model,args.selection, args.feature_amount)

        print("min, done")

    if args.operation == "max":
        # filtering the _mean columns
        marker = [x for x in data.columns if x.endswith("_max")]

        # adding columns that are necessary
        fixed_columns = ["age", 'va_lems_imputed', "lems_q6"]
        columns = marker + fixed_columns

        # copying the data set, with the columns for mean and age, va_lems and lems_q6
        data = data[columns].copy()
        # dropping the nan values in these columns
        data = data.dropna(subset=columns)

        methods = which_functions_to_run(args.model)

        run_models(results_dir, data, "max", methods, args.model,args.selection, args.feature_amount)

        print("max, done")

    if args.operation == "range":
        # filtering the _mean columns
        marker = [x for x in data.columns if x.endswith("_range")]

        # adding columns that are necessary
        fixed_columns = ["age", 'va_lems_imputed', "lems_q6"]
        columns = marker + fixed_columns

        # copying the data set, with the columns for mean and age, va_lems and lems_q6
        data = data[columns].copy()
        # dropping the nan values in these columns
        data = data.dropna(subset=columns)

        methods = which_functions_to_run(args.model)

        run_models(results_dir, data, "range", methods, args.model,args.selection, args.feature_amount)

        print("range, done")

    if args.operation == "measured_7":
        # filtering the _mean columns
        marker_0 = [x for x in data_missing.columns if x.endswith("_measured_7")]

        # adding columns that are necessary
        fixed_columns = ["age", 'va_lems_imputed', "lems_q6"]
        columns = marker_0 + fixed_columns

        # copying the data set, with the columns for mean and age, va_lems and lems_q6
        data_missing = data_missing[columns].copy()
        data_missing = data_missing.dropna(subset=columns)

        methods = which_functions_to_run(args.model)

        run_models(results_dir, data_missing, "measured_7", methods, args.model,args.selection, args.feature_amount)
        print("measured_7, done")

    if args.operation == "abnormal":
        # filtering the _mean columns
        marker_7 = [x for x in data_missing.columns if x.endswith("_times_measured_7")]
        marker_3 = [x for x in data_missing.columns if x.endswith("_times_measured_3")]
        marker_mean = [x for x in data_missing.columns if x.endswith("_mean")]
        marker_count = [value for value in data_missing.columns if "_count" in value]
        marker = marker_7 + marker_3 + marker_count + marker_mean + ["Sex", "va_ais", "ai_ais", "aii_aiis",
                                                                     "aiii_aiiis", "c_cs", "asia_chronic_q6",
                                                                     "asia_chronic_q6_q3",
                                                                     "ai_lems", "aii_lems", "aiii_lems", "c_lems",
                                                                     "lems_q6_q3", "patient_number", "Unnamed: 0",
                                                                     "Unnamed: 0.1", "va_lems", "age", "lems_q6",
                                                                     "va_lems_imputed"]
        all_columns = data_missing.columns
        # only columns for serological markers
        final_columns = [ele for ele in all_columns if ele not in marker]

        # ---------------------- features needed for prediction task
        # adding columns that are necessary for our prediction
        fixed_columns = ["age", "va_lems_imputed", "lems_q6"]
        columns = final_columns + fixed_columns

        # copying the data set, with the columns for mean and age, va_lems and lems_q6
        data_missing = data_missing[columns].copy()
        # dropping the nan values in these columns
        data_missing = data_missing.dropna(subset=columns)

        # ---------------------- CHECK FOR COLUMNS WITH NO INFO
        # data set with only serological markers
        serological_data = data_missing.copy()

        # counting unique values within serological marker data set
        unique_counts = serological_data.nunique()

        # Identify columns where the number of unique values is 1
        columns_to_drop = unique_counts[unique_counts == 1].index

        # Drop the identified columns
        data_missing = serological_data.drop(columns=columns_to_drop)

        columns_after_checking = [ele for ele in final_columns if ele not in columns_to_drop]

        # ---------------------- One Hot Encoding

        # one-hot-encoding only final_columns
        data_missing = pd.get_dummies(data_missing, columns=columns_after_checking)
        # -------- CONTROL
        unique_counts = data_missing.nunique()
        # Identify columns where the number of unique values is 1
        columns_to_drop_check = unique_counts[unique_counts == 1].index
        if len(columns_to_drop_check) > 0:
            raise ValueError("filtered encoded did not work ")

        methods = which_functions_to_run(args.model)

        run_models(results_dir, data_missing, "abnormal", methods, args.model,args.selection, args.feature_amount)

        print("abnormal, done")

    if args.operation == "lems":

        # adding columns that are necessary
        fixed_columns = ["age", 'va_lems_imputed', "lems_q6"]
        columns =  fixed_columns

        # copying the data set, with the columns for mean and age, va_lems and lems_q6
        data = data[columns].copy()
        data = data.dropna(subset=columns)

        methods = which_functions_to_run(args.model)

        run_models(results_dir, data, "lems_data", methods, args.model,args.selection, args.feature_amount)
        print("lems, done")

    if args.operation == "lems_missing":

        # adding columns that are necessary
        fixed_columns = ["age", 'va_lems_imputed', "lems_q6"]
        columns =  fixed_columns

        # copying the data set, with the columns for mean and age, va_lems and lems_q6
        data_missing = data_missing[columns].copy()
        data_missing = data_missing.dropna(subset=columns)

        methods = which_functions_to_run(args.model)

        run_models(results_dir, data_missing, "lems_missing_data", methods, args.model,args.selection, args.feature_amount)
        print("lems, done")

    if args.operation == "range_lems":
        # filtering the _mean columns
        marker = [x for x in data.columns if x.endswith("_range")]

        # adding columns that are necessary
        fixed_columns = ["age", 'va_lems_imputed', "lems_q6"]
        columns = marker + fixed_columns

        # copying the data set, with the columns for mean and age, va_lems and lems_q6
        data = data[columns].copy()
        data = data.dropna(subset=columns)

        data = data[fixed_columns].copy()

        methods = which_functions_to_run(args.model)

        run_models(results_dir, data,"range_lems",methods,args.model,args.selection, args.feature_amount)

        print("range_lems, done")

    if args.operation == "measured_7_noise":
        # only marker's below 0.7
        seed = 42
        np.random.seed(seed)

        data_missing.columns = [col + '_noise' if col.endswith('_measured_7') else col for col in data_missing.columns]
        marker = [x for x in data_missing.columns if x.endswith("_measured_7_noise")]

        data_missing['Noise_measured_7_noise'] = np.random.randint(0, 2, size=len(data_missing))

        # adding columns that are necessary
        fixed_columns = ["age", 'va_lems_imputed', "lems_q6", "Noise_measured_7_noise"]
        columns = marker + fixed_columns

        # copying the data set, with the columns for mean and age, va_lems and lems_q6
        data_missing = data_missing[columns].copy()
        data_missing = data_missing.dropna(subset=columns)

        data_missing.to_csv(results_dir + "/measured_7_corr_noise.csv")
        methods = which_functions_to_run(args.model)

        run_models(results_dir, data_missing, "measured_7_noise", methods, args.model,args.selection, args.feature_amount)

        print("measured_7_corr, done")

    if args.operation == "all":
        marker_7 = [x for x in data_missing.columns if x.endswith("_times_measured_7")]
        marker_3 = [x for x in data_missing.columns if x.endswith("_times_measured_3")]
        marker_mean = [x for x in data_missing.columns if x.endswith("_mean")]
        marker_count = [value for value in data_missing.columns if "_count" in value]
        marker = marker_7 + marker_3 + marker_count + marker_mean + ["Sex", "va_ais", "ai_ais", "aii_aiis",
                                                                     "aiii_aiiis", "c_cs", "asia_chronic_q6",
                                                                     "asia_chronic_q6_q3",
                                                                     "ai_lems", "aii_lems", "aiii_lems", "c_lems",
                                                                     "lems_q6_q3", "patient_number", "Unnamed: 0",
                                                                     "Unnamed: 0.1", "va_lems", "age", "lems_q6",
                                                                     "va_lems_imputed"]
        all_columns = data_missing.columns
        # only columns for serological markers
        final_columns = [ele for ele in all_columns if ele not in marker]

        # ---------------------- features needed for prediction task
        # adding columns that are necessary for our prediction
        fixed_columns = ["age", "va_lems_imputed", "lems_q6"]
        columns_encoded = final_columns + fixed_columns

        marker_serological = [x for x in data.columns if x.endswith("_mean")] + [x for x in data.columns if x.endswith("_median")] \
                 + [x for x in data.columns if x.endswith("_min")] + [x for x in data.columns if x.endswith("_max")] + \
                 [x for x in data.columns if x.endswith("_range")]

        data_total = pd.concat([data_missing[columns_encoded], data[marker_serological], data_missing[marker_7]], axis=1)

        data_total = data_total.dropna()

        # -------- Filtering features with no info (all same values)
        # data set with only serological markers
        serological_data = data_total[final_columns]

        # counting unique values within serological marker data set
        unique_counts = serological_data[final_columns].nunique()

        # Identify columns where the number of unique values is 1
        columns_to_drop = unique_counts[unique_counts == 1].index

        # dropping from overall data frame
        data_total = data_total.drop(columns=columns_to_drop)

        # which encoded are still present?
        columns_after_checking = [ele for ele in final_columns if ele not in columns_to_drop]

        data_total = pd.get_dummies(data_total, columns=columns_after_checking)

        # -------- CONTROL
        unique_counts = data_total.nunique()
        # Identify columns where the number of unique values is 1
        columns_to_drop_check = unique_counts[unique_counts == 1].index
        if len(columns_to_drop_check) > 0:
            raise ValueError("filtered encoded did not work ")

        methods = which_functions_to_run(args.model)

        run_models(results_dir, data_total, "all_foward_feature", methods, args.model,args.selection, args.feature_amount)

        print("all foward selection, done")
