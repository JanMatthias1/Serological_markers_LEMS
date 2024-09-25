""" Linear models script

The purpose of this script is to run the various linear models on the data

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
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.feature_selection import SelectPercentile
import random


def get_scores(model,X_train,X_test, y_train,y_test, name,va_lems_train,va_lems_test):
    """ Get scores from the respective model

        Parameters
        -------
        :param model: regression model thatt is being used
        :param X_train: feature rain set of the specific seed
        :param X_test: feature test set of the specific seed
        :param y_train: label train set of the specific seed
        :param y_test: label test set of the specific seed
        :param name: Name of the model, that will be printed
        :param va_lems_train: very acute score, train set
        :param va_lems_test: very acute score, test set

        Returns
        -------
        print of the score
    """

    y_pred_train =model.predict(X_train)
    y_pred_test =model.predict(X_test)

    y_pred_original=y_pred_test.copy()

    # va lems,pandas, but y_pred_test numpy
    y_pred_test[y_pred_test < 0] = 0
    y_pred_test[y_pred_test > 50] = 50

    r2_test = r2_score(y_test, y_pred_test)
    rmse_test = mean_squared_error(y_test, y_pred_test,squared=False)

    r2_train = r2_score(y_train, y_pred_train)
    rmse_train = mean_squared_error(y_train, y_pred_train,squared=False)

    # this is done with model predict
    print('Training set score', name, ': R2 score: {:.3f}, RMSE: {:.3f}'.format(r2_train, rmse_train))
    print('Test set score', name, ': R2 score: {:.3f}, RMSE: {:.3f}'.format(r2_test, rmse_test))

def r2_table (model,X_train,X_test,y_train, y_test, model_name,df_r2_performance, best_parameter,va_lems_train,va_lems_test):
    """ Get scores of the model and cross validation scores and create a tabke

        Parameters
        -------
        :param model: regression model being used
        :param X_train: feature rain set of the specific seed
        :param X_test: feature test set of the specific seed
        :param y_train: label train set of the specific seed
        :param y_test: label test set of the specific seed
        :param model_name: name of the model for the table
        :param df_r2_performance: data frame where the values will be inserted
        :param best_parameter: if a lasso, ridge regression, best parameters
        :param va_lems_train: very acute score, train set
        :param va_lems_test: very acute score, test set

        Returns
        -------
        data frame with the r-squared and RMSE scores from the model
    """
    #X_test here is in a numpy array format
    # but va_lems is pandas data frame --> iloc use

    y_pred_train =model.predict(X_train)
    y_pred_test =model.predict(X_test)

    y_pred_original=y_pred_test.copy()

    r2_test = r2_score(y_test, y_pred_test)
    rmse_test = mean_squared_error(y_test, y_pred_test, squared=False)
    mae_test = mean_absolute_error(y_test, y_pred_test)

    r2_train = r2_score(y_train, y_pred_train)
    rmse_train = mean_squared_error(y_train, y_pred_train, squared=False)
    mae_train = mean_absolute_error(y_train, y_pred_train)

    eval_metrics_non_modified = [r2_train, rmse_train, mae_train, r2_test, rmse_test, mae_test]

    # score correction, if the predicted values is lower than the initial one (upper and lower limit)
    y_pred_test[y_pred_test < 0] = 0
    y_pred_test[y_pred_test > 50] = 50

    y_pred_train[y_pred_train < 0] = 0
    y_pred_train[y_pred_train > 50] = 50

    r2_test_m = r2_score(y_test, y_pred_test)
    rmse_test_m = mean_squared_error(y_test, y_pred_test,squared=False)
    mae_test_m = mean_absolute_error(y_test, y_pred_test)

    r2_train_m = r2_score(y_train, y_pred_train)
    rmse_train_m = mean_squared_error(y_train, y_pred_train,squared=False)
    mae_train_m = mean_absolute_error(y_train, y_pred_train)

    eval_metrics_modified = [r2_train_m, rmse_train_m, mae_train_m, r2_test_m, rmse_test_m, mae_test_m]

    df_r2_performance.loc[len(df_r2_performance)-1,:]= [model_name] + eval_metrics_non_modified + eval_metrics_modified  + [best_parameter]  # inserting the model name and the evaluation metrics
    #the .loc here enables us to filter the last row of the data frame

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
    if model_name=="LR":
        # ads the coeff. to the new column
        table.loc[:,model_name + "_" + str(seed)]= model.coef_
        return table
    if model_name=="Lasso_Regression" or model_name=="Ridge_Regression":
        table[model_name + "_" + str(seed)]= model.best_estimator_.coef_
        return table
    if model_name=="Random_Forest_Regressor" or model_name=="Gradient_boosting_regressor" or model_name=="XGBoost" or model_name=="LightGBM" :
        table[model_name + "_" + str(seed)]= model.best_estimator_.feature_importances_
        return table
    if model_name== "SVR_linear":
        table[model_name + "_" + str(seed)]= model.best_estimator_.coef_.reshape(-1, 1)
        return table
def difference_in_scores(model,X_train,X_test, y_train,y_test, score_difference_train, score_difference_test, seed, model_name,va_lems_train,va_lems_test,data_type,path):
    """ Difference in scores, from true vs predicted q6 lems values

        Function creates data frames for the test and train set, with the va_lems score, q6 score,
        predicted score from the model and difference between the predicted and q6 score. Additionally,
        creates an image with the distribution of the values (histogram).

        Parameters
        -------
        :param model: model of interest
        :param X_train: feature rain set of the specific seed
        :param X_test: feature test set of the specific seed
        :param y_train: label train set of the specific seed
        :param y_test: label test set of the specific seed
        :param score_difference_train: data frame for difference of score/predicted values for the train set
        :param score_difference_test: data frame for difference of score/predicted values for the test set
        :param seed: seed iteration
        :param model_name: name of the model, can be customised
        :param va_lems_train: va_lems values from the train set
        :param va_lems_test: va_lems values from the test set
        :param data_type: specified by the user (for example: mean, max, min, range, abnormal,..), used for saving image with the correct name (here it is the name of data + seed iteration)
        :param path: where to save the output image

        Returns
        ------
        Two tables (score_difference_train and score_difference_test) and immage of the distribution of the particular seed
    """
    y_pred_train =model.predict(X_train)
    y_pred_test =model.predict(X_test)

    # print(y_pred_test) --> np.series
    # x_test, x_train --> np.series
    # print(y_train) --> pd.data frame
    # print(va_lems_test) --> pd.data frame

    y_pred_test_original= y_pred_test.copy()
    y_pred_train_original = y_pred_train.copy()

    # score correction
    y_pred_test[y_pred_test < 0] = 0
    y_pred_test[y_pred_test > 50] = 50

    y_pred_train[y_pred_train < 0] = 0
    y_pred_train[y_pred_train > 50] = 50

    train_difference=abs(y_pred_train_original-y_train)
    train_difference_m = abs(y_pred_train - y_train)

    test_difference=abs(y_pred_test_original - y_test)
    test_difference_m= abs(y_pred_test - y_test)


    rmse_test = mean_squared_error(y_test, y_pred_test, squared=False)

    temporary_data_set_train = pd.DataFrame()
    temporary_data_set_test = pd.DataFrame()

    temporary_data_set_train.loc[:, model_name + "_" + "index" + "_" + str(seed)] = train_difference.index
    temporary_data_set_train.loc[:, model_name + "_" + "difference" + "_" + str(seed)] = train_difference.tolist()
    temporary_data_set_train.loc[:, model_name + "_" + "difference_m" + "_" + str(seed)] = train_difference_m.tolist()
    temporary_data_set_train.loc[:, model_name + "_" + "va_lems_train" + "_" + str(seed)] = va_lems_train.tolist()
    temporary_data_set_train.loc[:, model_name + "_" + "q6_lems" + "_" + str(seed)] = y_train.tolist()
    temporary_data_set_train.loc[:, model_name + "_" + "y_predicted" + "_" + str(seed)] = y_pred_train_original.tolist()
    temporary_data_set_train.loc[:, model_name + "_" + "y_predicted_modified" + "_" + str(seed)] = y_pred_train.tolist()

    # concatenating new data frame to output of function
    score_difference_train = pd.concat([score_difference_train, temporary_data_set_train], axis=1)

    temporary_data_set_test.loc[:, model_name + "_" + "index" + "_" + str(seed)] = test_difference.index
    temporary_data_set_test.loc[:, model_name + "_" + "difference_" + "_" + str(seed)] = test_difference.tolist()
    temporary_data_set_test.loc[:, model_name + "_" + "difference_m" + "_" + str(seed)] = test_difference_m.tolist()
    temporary_data_set_test.loc[:, model_name + "_" + "va_lems_train" + "_" + str(seed)] = va_lems_test.tolist()
    temporary_data_set_test.loc[:, model_name + "_" + "q6_lems" + "_" + str(seed)] = y_test.tolist()
    temporary_data_set_test.loc[:, model_name + "_" + "y_predicted" + "_" + str(seed)] = y_pred_test_original.tolist()
    temporary_data_set_test.loc[:, model_name + "_" + "y_predicted_modified" + "_" + str(seed)] = y_pred_test.tolist()

    # concatenating new data frame to output of function
    score_difference_test = pd.concat([score_difference_test, temporary_data_set_test], axis=1)
    '''
    # plotting distribution of scores
    data_list = [va_lems_test, y_test, y_pred_test_original, y_pred_test, test_difference_m]
    labels = ["va_lems", "True LEMS q6", "Predicted, LEMS q6", "Modified LEMS q6",
              "test_difference_modified"]
    colors = ["black", "blue", "red", "red", "yellow"]

    # y_axis limit
    unique, counts = np.unique(va_lems_test, return_counts=True)
    # y limit, corrisponds to the max count (amount) of one specific value
    y_limit = counts.max()

    # Create the subplots in a loop
    plt.figure(figsize=(20, 5))
    for i, (data, label, color) in enumerate(zip(data_list, labels, colors)):
        plt.subplot(1, 5, i + 1)
        sns.histplot(data, binwidth=3, label=label, color=color)
        plt.xlabel(label)
        plt.ylim([0, y_limit+1])
        plt.ylabel("Frequency")

    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    plt.title('The RMSE is {:.2f}'.format(rmse_test))

    path = path + "/Histogram/"
    # Create the output folder
    Path(path).mkdir(exist_ok=True)

    location = path + model_name + data_type + str(seed) + "_histogram" + ".png"
    plt.savefig(location)

    # clear current image for the next one
    plt.clf()
    plt.close()
    '''
    return score_difference_train, score_difference_test
def linear_regression_RepeatedKFold (X_train,X_test, y_train,y_test, data_type, path, df_r2_performance,seed,va_lems_train,va_lems_test,score_difference_train, score_difference_test,dfs):
    """ Linear Regression function

        Parameters:
        --------
        :param X_train: feature rain set of the specific seed
        :param X_test: feature test set of the specific seed
        :param y_train: label train set of the specific seed
        :param y_test: label test set of the specific seed
        :param data_type: specified by the user (for example: mean, max, min, range, abnormal,..), used for saving image with the correct name
        :param path: where to save the various tables and images from the function
        :param df_r2_performance: data frame where the performance values will be inserted
        :param seed: seed iteration
        :param va_lems_train: va_lems values from the train set
        :param va_lems_test: va_lems values from the test set
        :param score_difference_train: data frame with the train score difference between predicted and acutal values
        :param score_difference_test: data frame with the test score difference between predicted and acutal values
        :param dfs: dictionary containing the feature importance data frames

        Returns:
        ------
        df_r2_performance table and feature importance tables. Aditionally, plots the predicted values against the actual values and value distribution
    """
    # for the leave on out cross validation, the cross_validate function fits the model and cross-validates it
    model_LR = LinearRegression().fit(X_train, y_train)

    # Add feature scores to the appropriate dataframe within the dictionary dfs
    dfs["feature_importance_Linear"][seed]= add_features_all(model_LR, "LR", seed, dfs["feature_importance_Linear"][seed])

    # inert the performance values in the dataframe
    df_r2_performance= r2_table(model_LR, X_train, X_test, y_train, y_test, "Linear_regression",df_r2_performance,0,va_lems_train,va_lems_test)

    # print the scores in the output
    get_scores(model_LR,X_train,X_test, y_train,y_test, model_LR.__class__.__name__ ,va_lems_train,va_lems_test)

    # Plot the predicted vs actual scores
    pairwise_plot(model_LR,X_test, y_test, "Linear_Regression",data_type,seed, path)

    # va_lems score, predicted and actual label score (lemsq6)
    score_difference_train, score_difference_test = difference_in_scores(model_LR, X_train, X_test, y_train, y_test, score_difference_train, score_difference_test, seed,
                         "LR", va_lems_train, va_lems_test, data_type, path)

    return(df_r2_performance, dfs, score_difference_train, score_difference_test)
def Lasso_Regression_RepeatedKFold (X_train,X_test, y_train,y_test,data_type,path,df_r2_performance,seed,va_lems_train,va_lems_test,score_difference_train, score_difference_test,dfs):
    """ Lasso regression function, with GridSearchCV for an optimal alpha

        Parameters:
        --------
        :param X_train: feature rain set of the specific seed
        :param X_test: feature test set of the specific seed
        :param y_train: label train set of the specific seed
        :param y_test: label test set of the specific seed
        :param data_type: specified by the user (for example: mean, max, min, range, abnormal,..), used for saving image with the correct name
        :param path: where to save the various tables and images from the function
        :param df_r2_performance: data frame where the performance values will be inserted
        :param seed: seed iteration
        :param va_lems_train: va_lems values from the train set
        :param va_lems_test: va_lems values from the test set
        :param score_difference_train: data frame with the train score difference between predicted and acutal values
        :param score_difference_test: data frame with the test score difference between predicted and acutal values
        :param dfs: dictionary containing the feature importance data frames

        Returns:
        ------
        df_r2_performance table and feature importance tables. Aditionally plots the predicted values against the actual values and value distribution

    """
    model_lasso = Lasso(random_state=42)

    hyper_params ={'alpha': np.arange(0.5, 1500, 0.5)}

    model_lasso_cv = GridSearchCV(model_lasso, hyper_params, cv=5, scoring="neg_root_mean_squared_error", n_jobs=-1)
    # doing a grid_search_CV with stratified fold

    # fitting the data (training set, x and y)
    model_lasso_cv.fit(X_train, y_train)

    # printing the scores after input
    get_scores(model_lasso_cv,X_train,X_test, y_train,y_test, "Lasso Regression",va_lems_train,va_lems_test)

    # adding the various estimators to a table
    df_r2_performance= r2_table(model_lasso_cv, X_train, X_test, y_train, y_test, "Lasso_regression",df_r2_performance,model_lasso_cv.best_params_['alpha'],
                                va_lems_train,va_lems_test)

    # adding the features to the table + histogram
    dfs["feature_importance_Lasso"][seed]= add_features_all(model_lasso_cv, "Lasso_Regression", seed, dfs["feature_importance_Lasso"][seed])

    # Plot the predicted vs actual scores
    pairwise_plot(model_lasso_cv, X_test, y_test, "Lasso_Regression",data_type,seed, path)

    # va_lems score, predicted and actual label score (lemsq6)
    score_difference_train, score_difference_test= difference_in_scores(model_lasso_cv, X_train, X_test, y_train, y_test, score_difference_train, score_difference_test,
                         seed,
                         "Lasso", va_lems_train, va_lems_test, data_type, path)

    return(df_r2_performance,dfs, score_difference_train, score_difference_test)
def Ridge_Regression_RepeatedKFold (X_train,X_test, y_train,y_test, data_type,path,df_r2_performance,seed,va_lems_train,va_lems_test,score_difference_train, score_difference_test,dfs):
    """Ridge regression function, with GridSearchCV for an optimal alpha

        Parameters:
        --------
        :param: X,y: data split for the cross_validate scoring
        :param X_train: feature rain set of the specific seed
        :param X_test: feature test set of the specific seed
        :param y_train: label train set of the specific seed
        :param y_test: label test set of the specific seed
        :param data_type: specified by the user (for example: mean, max, min, range, abnormal,..), used for saving image with the correct name
        :param path: where to save the various tables and images from the function
        :param df_r2_performance: data frame where the performance values will be inserted
        :param seed: seed iteration
        :param va_lems_train: va_lems values from the train set
        :param va_lems_test: va_lems values from the test set
        :param score_difference_train: data frame with the train score difference between predicted and acutal values
        :param score_difference_test: data frame with the test score difference between predicted and acutal values
        :param dfs: dictionary containing the feature importance data frames

        Returns:
        ------
        df_r2_performance table and feature importance tables. Addutionally plots the predicted values against the actual values and value distribution

    """

    model_ridge = Ridge(random_state=42)

    hyper_params = {'alpha': np.arange(0.5, 1500, 0.5)}

    model_ridge_cv = GridSearchCV(model_ridge, hyper_params, cv=5, scoring="neg_root_mean_squared_error", n_jobs=-1)
    # doing a grid_search_CV with stratified fold
    # where values closer to zero represent less prediction error by the model --> negative root mean squared error

    # fitting the data (training set, x and y)
    model_ridge_cv.fit(X_train, y_train)

    # printing the scores after input
    get_scores(model_ridge_cv, X_train, X_test, y_train, y_test, "Ridge Regression",va_lems_train,va_lems_test)

    # adding the various estimators to a table
    df_r2_performance= r2_table(model_ridge_cv, X_train, X_test, y_train, y_test, "Ridge_regression",df_r2_performance, model_ridge_cv.best_params_['alpha'],
                                va_lems_train,va_lems_test)

    # adding the features to the table + histogram
    dfs["feature_importance_Ridge"][seed] = add_features_all(model_ridge_cv, "Ridge_Regression", seed, dfs["feature_importance_Ridge"][seed])

    # Plot the predicted vs actual scores
    pairwise_plot(model_ridge_cv, X_test, y_test, "Ridge_Regression", data_type,seed, path)

    # va_lems score, predicted and actual label score (lemsq6)
    score_difference_train, score_difference_test= difference_in_scores(model_ridge_cv, X_train, X_test, y_train, y_test, score_difference_train, score_difference_test,seed,"Ridge", va_lems_train, va_lems_test, data_type, path)

    return (df_r2_performance,dfs,score_difference_train, score_difference_test )
def Random_forest_regressor (X_train,X_test, y_train,y_test, data_type,path,df_r2_performance,seed,va_lems_train,va_lems_test,score_difference_train, score_difference_test,dfs):
    """Random Forest regression function, with GridSearchCV for an optimal number of estimators, features, depth and leaf_nodes

        Parameters:
        --------
        :param X_train: feature rain set of the specific seed
        :param X_test: feature test set of the specific seed
        :param y_train: label train set of the specific seed
        :param y_test: label test set of the specific seed
        :param data_type: specified by the user (for example: mean, max, min, range, abnormal,..), used for saving image with the correct name
        :param path: where to save the various tables and images from the function
        :param df_r2_performance: data frame where the performance values will be inserted
        :param seed: seed iteration
        :param va_lems_train: va_lems values from the train set
        :param va_lems_test: va_lems values from the test set
        :param score_difference_train: data frame with the train score difference between predicted and acutal values
        :param score_difference_test: data frame with the test score difference between predicted and acutal values
        :param dfs: dictionary containing the feature importance data frames

        Returns:
        ------
        df_r2_performance table and feature importance tables. Aditionally plots the predicted values against the actual values and value distribution
    """

    model_RFR = RandomForestRegressor(random_state=42)

    param_grid = {
        'n_estimators': [10, 25, 50, 100],
        'max_features': ['sqrt', 'log2', None],
        'max_depth': [3, 6, 9],
        'max_leaf_nodes': [3, 6, 9],
    }

    model_RFR_cv = GridSearchCV(model_RFR, param_grid, cv=5, scoring="neg_root_mean_squared_error", n_jobs=-1)
    # doing a grid_search_CV with stratified fold
    # where values closer to zero represent less prediction error by the model --> negative root mean squared error

    # fitting the data (training set, x and y)
    model_RFR_cv.fit(X_train, y_train)
    y_pred_test = model_RFR_cv.predict(X_test)

    # printing the scores after input
    get_scores(model_RFR_cv,X_train,X_test, y_train,y_test,"Random forest Regressor",va_lems_train,va_lems_test)

    # adding the various estimators to a table
    df_r2_performance= r2_table(model_RFR_cv, X_train, X_test, y_train, y_test, "Random_Forest_Regressor",df_r2_performance, 0,va_lems_train,va_lems_test)

    # adding the features to the table
    dfs["feature_importance_RFR"][seed] = add_features_all(model_RFR_cv, "Random_Forest_Regressor", seed, dfs["feature_importance_RFR"][seed])
    # Plot the predicted vs actual scores
    pairwise_plot(model_RFR_cv, X_test, y_test, "Random_Forest_Regressor", data_type,seed, path)

    # va_lems score, predicted and actual label score (lemsq6)
    score_difference_train, score_difference_test= difference_in_scores(model_RFR_cv, X_train, X_test, y_train, y_test, score_difference_train, score_difference_test,
                         seed,"RFR", va_lems_train, va_lems_test, data_type, path)

    return (df_r2_performance, dfs, score_difference_train, score_difference_test)
def Support_vector_regressor_linear (X_train,X_test, y_train,y_test, data_type,path,df_r2_performance,seed,va_lems_train,va_lems_test,score_difference_train, score_difference_test,dfs):
    """Support vector regressor, with linear kernel, with GridSearchCV for an optimal epsilon

        Parameters:
        --------
        :param X_train: feature rain set of the specific seed
        :param X_test: feature test set of the specific seed
        :param y_train: label train set of the specific seed
        :param y_test: label test set of the specific seed
        :param data_type: specified by the user (for example: mean, max, min, range, abnormal,..), used for saving image with the correct name
        :param path: where to save the various tables and images from the function
        :param df_r2_performance: data frame where the performance values will be inserted
        :param seed: seed iteration
        :param va_lems_train: va_lems values from the train set
        :param va_lems_test: va_lems values from the test set
        :param score_difference_train: data frame with the train score difference between predicted and acutal values
        :param score_difference_test: data frame with the test score difference between predicted and acutal values
        :param dfs: dictionary containing the feature importance data frames

        Returns:
        ------
        df_r2_performance table and feature importance tables. Aditionally plots the predicted values against the actual values
        """

    np.random.seed(42)
    model_RVR =  SVR(kernel = 'linear')

    # doing a grid_search_CV with stratified fold for epsilon
    params = {'epsilon': np.arange(0, 1.5, 0.1)}

    model_RVR = GridSearchCV(model_RVR, param_grid=params, cv=5, scoring="neg_root_mean_squared_error", verbose=1, n_jobs=-1)

    # fitting the data (training set, x and y)
    model_RVR.fit(X_train, y_train)

    # printing the scores after input
    get_scores(model_RVR,X_train,X_test, y_train,y_test,"Support_vector_regressor_linear",va_lems_train,va_lems_test)

    # adding the various estimators to a table
    df_r2_performance= r2_table(model_RVR, X_train, X_test, y_train, y_test, "Support_vector_regressor_linear",df_r2_performance, model_RVR.best_params_['epsilon'],va_lems_train,va_lems_test)

    # Plot the predicted vs actual scores
    pairwise_plot(model_RVR, X_test, y_test, "Support_vector_regressor_linear", data_type,seed, path)

    score_difference_train, score_difference_test = difference_in_scores(model_RVR, X_train, X_test, y_train, y_test, score_difference_train, score_difference_test,
                         seed,
                         "RVR", va_lems_train, va_lems_test, data_type, path)

    dfs["feature_importance_SVR"][seed] = add_features_all(model_RVR, "SVR_linear", seed, dfs["feature_importance_SVR"][seed])

    return (df_r2_performance,dfs, score_difference_train, score_difference_test)
def gradient_boost_regressor (X_train,X_test, y_train,y_test,data_type,path,df_r2_performance,seed,va_lems_train,va_lems_test,score_difference_train, score_difference_test,dfs):
    """Gradient boost regressor, with GridSearchCV for optimal number of estimators, learning rate, subsample and max depth

            Parameters:
            --------
            :param X_train: feature rain set of the specific seed
            :param X_test: feature test set of the specific seed
            :param y_train: label train set of the specific seed
            :param y_test: label test set of the specific seed
            :param data_type: specified by the user (for example: mean, max, min, range, abnormal,..), used for saving image with the correct name
            :param path: where to save the various tables and images from the function
            :param df_r2_performance: data frame where the performance values will be inserted
            :param seed: seed iteration
            :param va_lems_train: va_lems values from the train set
            :param va_lems_test: va_lems values from the test set
            :param score_difference_train: data frame with the train score difference between predicted and acutal values
            :param score_difference_test: data frame with the test score difference between predicted and acutal values
            :param dfs: dictionary containing the feature importance data frames

            Returns:
            ------
            df_r2_performance table and feature importance tables. Aditionally plots the predicted values against the actual values
            """

    model_GBR = GradientBoostingRegressor(random_state=42) # in order to have a deterministic behaviour during the fit

    hyper_params = {
        'n_estimators': [10, 25, 50, 100],
        'learning_rate': [0.001, 0.01, 0.05],
        'subsample': [0.5, 0.7, 0.8],
        'max_depth': [3,5, 6],
        'min_samples_split': [8, 10, 15],
        'min_samples_leaf': [5,8,10]
    }

    model_GBR_cv = GridSearchCV(model_GBR, hyper_params, cv=5, scoring="neg_root_mean_squared_error", n_jobs=-1)
    # doing a grid_search_CV with stratified fold
    #where values closer to zero represent less prediction error by the model --> negative root mean squared error

    # fitting the data (training set, x and y)
    model_GBR_cv.fit(X_train, y_train)

    # printing the scores after input
    get_scores(model_GBR_cv,X_train,X_test, y_train,y_test, "Gradient Boosting Regressor",va_lems_train,va_lems_test)

    # adding the various estimators to a table
    df_r2_performance= r2_table(model_GBR_cv, X_train, X_test, y_train, y_test, "Gradient_boost_regressor",df_r2_performance,0,va_lems_train,va_lems_test)

    # adding the features to the table
    dfs["feature_importance_GBR"][seed] = add_features_all(model_GBR_cv, "Gradient_boosting_regressor", seed, dfs["feature_importance_GBR"][seed])

    # Plot the predicted vs actual scores
    pairwise_plot(model_GBR_cv, X_test, y_test, "Gradient_boost_regressor",data_type,seed, path)

    # va_lems score, predicted and actual label score (lemsq6)
    score_difference_train, score_difference_test= difference_in_scores(model_GBR_cv, X_train, X_test, y_train, y_test, score_difference_train, score_difference_test,
                         seed, "GBR", va_lems_train, va_lems_test, data_type, path)

    return(df_r2_performance,dfs,score_difference_train, score_difference_test)
def XGboost (X_train,X_test, y_train,y_test,data_type,path,df_r2_performance,seed,va_lems_train,va_lems_test,score_difference_train, score_difference_test,dfs):
    """XGboost, with GridSearchCV for optimal number of estimators, max depth, eta, subsample, colsample_bytree

        Parameters:
        --------
        :param X_train: feature rain set of the specific seed
        :param X_test: feature test set of the specific seed
        :param y_train: label train set of the specific seed
        :param y_test: label test set of the specific seed
        :param data_type: specified by the user (for example: mean, max, min, range, abnormal,..), used for saving image with the correct name
        :param path: where to save the various tables and images from the function
        :param df_r2_performance: data frame where the performance values will be inserted
        :param seed: seed iteration
        :param va_lems_train: va_lems values from the train set
        :param va_lems_test: va_lems values from the test set
        :param score_difference_train: data frame with the train score difference between predicted and acutal values
        :param score_difference_test: data frame with the test score difference between predicted and acutal values
        :param dfs: dictionary containing the feature importance data frames

        Returns:
        ------
        df_r2_performance table and feature importance tables. Aditionally plots the predicted values against the actual values
    """

    model_XGB = XGBRegressor(random_state=42)

    param_grid = {
        'n_estimators': [10, 25, 50, 100],
        'max_depth': [3,5, 6],
        'eta':[0.001 ,0.01, 0.05],
        'subsample':[0.3, 0.5, 0.9],
        'colsample_bytree': [0.5, 0.9],
        'gamma': [0.2, 0.3 ,0.4, 0.5]
    }

    model_XGB_cv = GridSearchCV(model_XGB, param_grid, cv=5, scoring="neg_root_mean_squared_error", n_jobs=-1)
    # doing a grid_search_CV with stratified fold

    # fitting the data (training set, x and y)
    model_XGB_cv.fit(X_train, y_train)

    # printing the scores after input
    get_scores(model_XGB_cv,X_train,X_test, y_train,y_test, "XGBoost regression",va_lems_train,va_lems_test)

    # adding the various estimators to a table
    df_r2_performance= r2_table(model_XGB_cv, X_train, X_test, y_train, y_test, "XGBoost",df_r2_performance,0,va_lems_train,va_lems_test)

    # Plot the predicted vs actual scores
    pairwise_plot(model_XGB_cv, X_test, y_test, "XGBoost",data_type,seed, path)

    # va_lems score, predicted and actual label score (lemsq6)
    score_difference_train, score_difference_test= difference_in_scores(model_XGB_cv, X_train, X_test, y_train, y_test, score_difference_train, score_difference_test,
                         seed,
                         "XGB", va_lems_train, va_lems_test, data_type, path)

    dfs["feature_importance_XGB"][seed] = add_features_all(model_XGB_cv, "XGBoost", seed, dfs["feature_importance_XGB"][seed])
    # now can estimate the parameter
    print(model_XGB_cv.get_params)

    return(df_r2_performance,dfs , score_difference_train, score_difference_test) #feature_importance_GBR
def lightGBM (X_train,X_test, y_train,y_test,data_type,path,df_r2_performance,seed,va_lems_train,va_lems_test,score_difference_train, score_difference_test,dfs):
    """XGboost, with GridSearchCV for optimal number of estimators, max depth, eta, subsample, colsample_bytree

        Parameters:
        --------
        :param X_train: feature rain set of the specific seed
        :param X_test: feature test set of the specific seed
        :param y_train: label train set of the specific seed
        :param y_test: label test set of the specific seed
        :param data_type: specified by the user (for example: mean, max, min, range, abnormal,..), used for saving image with the correct name
        :param path: where to save the various tables and images from the function
        :param df_r2_performance: data frame where the performance values will be inserted
        :param seed: seed iteration
        :param va_lems_train: va_lems values from the train set
        :param va_lems_test: va_lems values from the test set
        :param score_difference_train: data frame with the train score difference between predicted and acutal values
        :param score_difference_test: data frame with the test score difference between predicted and acutal values
        :param dfs: dictionary containing the feature importance data frames

        Returns:
        ------
        df_r2_performance table and feature importance tables. Aditionally plots the predicted values against the actual values
    """

    param_grid = {
        'max_depth': [1, 2,3],
        'num_leaves': [2, 3],
        'metric': ['l2', 'l1', 'poisson'],
        'min_child_samples': [10],
        'learning_rate': [0.05, 0.1]
    }

    #lightGBM = lgb.LGBMRegressor(force_col_wise=True)

    lightGBM =LGBMRegressor(force_col_wise=True, random_state=42, verbose=1)

    lgb_model_cv = GridSearchCV(lightGBM, param_grid, scoring= 'neg_root_mean_squared_error', cv=5, n_jobs=-1)
    lgb_model_cv.fit(X_train, y_train)

    # printing the scores after input
    get_scores(lgb_model_cv,X_train,X_test, y_train,y_test, "LightGBM regression",va_lems_train,va_lems_test)

    # adding the various estimators to a table
    df_r2_performance= r2_table(lgb_model_cv, X_train, X_test, y_train, y_test, "LightGBM",df_r2_performance,0,va_lems_train,va_lems_test)

    # Plot the predicted vs actual scores
    pairwise_plot(lgb_model_cv, X_test, y_test, "LightGBM",data_type,seed, path)

    # va_lems score, predicted and actual label score (lemsq6)
    score_difference_train, score_difference_test= difference_in_scores(lgb_model_cv, X_train, X_test, y_train, y_test, score_difference_train, score_difference_test,
                         seed,
                         "LightGBM", va_lems_train, va_lems_test, data_type, path)

    dfs["feature_importance_LGBM"][seed] = add_features_all(lgb_model_cv, "LightGBM", seed, dfs["feature_importance_LGBM"][seed])

    return(df_r2_performance,dfs,score_difference_train, score_difference_test) #feature_importance_GBR
def fit_models(X_train, X_test, y_train, y_test, data_type,path,df_r2_performance,seed,va_lems_train,va_lems_test,score_difference_train, score_difference_test,dfs,methods):
    """Fit the various regression models on the data all within one function

        Parameters:
         --------
        :param X_train: feature rain set of the specific seed
        :param X_test: feature test set of the specific seed
        :param y_train: label train set of the specific seed
        :param y_test: label test set of the specific seed
        :param data_type: specified by the user (for example: mean, max, min, range, abnormal,..), used for saving image with the correct name
        :param path: where to save the various tables and images from the function
        :param df_r2_performance: data frame where the performance values will be inserted
        :param seed: seed iteration
        :param va_lems_train: va_lems values from the train set
        :param va_lems_test: va_lems values from the test set
        :param score_difference_train: data frame with the train score difference between predicted and acutal values
        :param score_difference_test: data frame with the test score difference between predicted and acutal values
        :param dfs: dictionary containing the feature importance data frames
        :param methods: which regression models to run

        Returns
        -------
        df_r2_performance table and feature importance tables of the various models
    """

    for method in methods:
        df_r2_performance, dfs, score_difference_train, score_difference_test= method(X_train, X_test, y_train, y_test, data_type, path, df_r2_performance, seed, va_lems_train, va_lems_test,score_difference_train, score_difference_test, dfs)

    return df_r2_performance,dfs, score_difference_train, score_difference_test

def pairwise_plot(model,X_test, y_test, title,data_type,seed, path):
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

    #predicting the values, model is imputed depending on the model choosen
    y_pred_test =model.predict(X_test)
    '''
    fig, ax = plt.subplots()
    sns.set(style="darkgrid")
    sns.regplot(x=y_pred_test, y=y_test,
                scatter_kws=dict(color='black', s=10, alpha=0.5)).set(title=title + data_type + str(seed),
                                                                      xlabel='Predicted LEMS values',
                                                                      ylabel='True value LEMS values')

    ax.set_xlim(0, 50)
    ax.set_ylim(-10, 60)

    path = path + "/Pairwise_plot/"
    # Create the output folder
    Path(path).mkdir(exist_ok=True)

    location = path + title + data_type + str(seed) + ".png"
    plt.savefig(location)
    # clear current image for the next one
    plt.clf()
    plt.close()  # due to memory warning of having all the pictures open
    '''
def seed_multiple_times(data_type, path, df_seed_performance, stratify,score_difference_train, score_difference_test,methods,name_data, model_names, data,feature_selection,feature_amount):
    """ Running the models with different seeds

        Parameters:
        -------
        :param X: features from the data set, already filtered depending on correlation values
        :param y: labels from the data set
        :param data_type: specified by the user (for example: mean, max, min, range, abnormal,..), used for saving image with the correct name
        :param path: where to save the various tables and images from the function
        :param df_seed_performance: summary table of the performance of all models within different seeds
        :param stratify: stratification for the split_train_test set
        :param score_difference_train: data frame with the train score difference between predicted and acutal values
        :param score_difference_test: data frame with the test score difference between predicted and acutal values
        :param dfs: dictionary containing the feature importance data frames
        :param methods: which regression models to run

        Returns
        -------
        df_seed_importance, and feature importance with mean values and standard deviation
    """

    variable = [i for i in range(1, 51)]
    # Initialize an empty dictionary with predefined key names
    dfs = {f'feature_importance_{model}': {} for model in model_names}

    for seed in variable:

        print(seed)

        df_r2_performance = pd.DataFrame(
            columns=['Model_name', "r2_train_80", "rmse_train_80", "mae_train_80", "r2_test_20", "rmse_test_20",
                     "mae_test_20",
                     "r2_train_80_m", "rmse_train_80_m", "mae_train_80_m", "r2_test_20_m", "rmse_test_20_m",
                     "mae_test_20_m", "best_parameters"])

        if name_data == "mean" or name_data=="median" or name_data=="min" or name_data=="max" or name_data=="range" \
                or name_data=="measured_7" or name_data=="measured_7_noise":

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
                upper = correlation_matrix_abs.where(np.triu(np.ones(correlation_matrix_abs.shape), k=1).astype(bool))

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
                upper = correlation_matrix_abs.where(np.triu(np.ones(correlation_matrix_abs.shape), k=1).astype(bool))

                correlated_features_check = []
                correlated_features_check = [column for column in upper.columns if any(upper[column] > 0.7)]

                if len(correlated_features_check) > 0:
                    raise ValueError("Correlated features detected. This is not allowed.")

                required_features = ["age", "va_lems_imputed"]
                selected_07 = selected_07 + required_features

            if feature_selection =="MI":
                # only select out of the serological markers

                selector = SelectPercentile(lambda X, y: mutual_info_regression(X, y, random_state=42), percentile=float(feature_amount))
                selector.fit(X_train[marker_all], y_train)

                # Get the mask of selected features
                selected_mask = selector.get_support()

                # Extract the feature names of the selected features
                selected_features = [feature for feature, selected in zip(marker_all, selected_mask) if selected]

                if name_data == "measured_7_noise":
                    required_features = ["age", "va_lems_imputed", "Noise_measured_7_noise"]

                else:
                    required_features = ["age", "va_lems_imputed"]

                selected_07 = selected_features + required_features

            # ----------- final split for task (on scaled)

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed,
                                                                    stratify=X["va_lems_imputed"] > 0)


            X_train = X_train[selected_07]
            X_test = X_test[selected_07]

            # -------  Feature importance data Sets
            # iterating in model_names to access the correct dictionary location --> overall dictionary already created above
            for model in model_names:
                # Generate a DataFrame (replace this with your actual logic)
                feature_importance = pd.DataFrame(index=X_train.columns)

                # Add the DataFrame to the inner dictionary
                dfs[f'feature_importance_{model}'][seed] = feature_importance

        if name_data == "all_foward_feature":

            # features to select from
            marker=data.columns

            features = [col for col in marker if col != "lems_q6"]
  
            X = data[features]  # Include all your features here
            y = data['lems_q6']

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed,
                                                                stratify=X["va_lems_imputed"] > 0)
            # -------  scaling X feature selection
            columns = X_train.columns
            scaler = StandardScaler()

            to_scale = [x for x in data.columns if x.endswith(("_mean", "_median", "_min", "_max", "_range", "_times_measured_7"))] \
                       + ['age', 'va_lems_imputed']

            columns_encoded = [feature for feature in columns if feature not in to_scale]

            X_train_scaled_columns = scaler.fit_transform(X_train[to_scale])

            X_train = X_train.drop(columns=to_scale)

            X_train_scaled = np.concatenate([X_train_scaled_columns, X_train], axis=1)

            # Convert the scaled data back to a DataFrame
            X_train = pd.DataFrame(X_train_scaled, columns=[to_scale + columns_encoded])

            if feature_selection== "RFS":
                # -------
                # Initial setup
                p_value_threshold = 0.05  # Set your desired threshold
                selected_features = ["age", "va_lems_imputed"]
                remaining_features = [feature for feature in features if feature != "age" and feature != "va_lems_imputed"]

                # Foward feature selection
                while True:
                    if not remaining_features:
                        break  # No more features to add

                    # setting up the most_significant_feature and current_best_p_value --> here the most significant feature for that iteration is stored
                    most_significant_feature = None
                    current_best_p_value = float('inf')
                    #print(selected_features)

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

                            #print("best feature", most_significant_feature, feature, current_best_p_value)

                    # Check if the most significant feature is below the threshold --> establish above
                    if current_best_p_value < p_value_threshold:
                        # append new selected feature
                        selected_features.append(most_significant_feature)
                        remaining_features.remove(most_significant_feature)
                    else:
                        break  # Stop if no more features below the threshold

            if feature_selection =="MI":

                remaining_features = [feature for feature in features if
                                      feature != "age" and feature != "va_lems_imputed"]

                selector = SelectPercentile(lambda X, y: mutual_info_regression(X, y, random_state=42), percentile=float(feature_amount))
                selector.fit(X_train[remaining_features], y_train)

                # Get the mask of selected features
                selected_mask = selector.get_support()

                # Extract the feature names of the selected features
                selected_features = [feature for feature, selected in zip(remaining_features, selected_mask) if selected]

                # Add required features to selected_features list
                selected_features.extend(["age", "va_lems_imputed"])


            # ----------- final split for task

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed,
                                                                stratify=X["va_lems_imputed"] > 0)
            X_train=X_train[selected_features]
            X_test=X_test[selected_features]

            to_scale = [x for x in data.columns if x.endswith(("_mean", "_median", "_min", "_max", "_range", "_times_measured_7"))] \
                       + ['age', 'va_lems_imputed']

            columns = X_train.columns

            columns_to_scale = [feature for feature in columns if feature in to_scale]
            columns_encoded = [feature for feature in columns if feature not in to_scale]
            columns_feature_importance = columns_to_scale + columns_encoded

            for model in model_names:
                # Generate a DataFrame (replace this with your actual logic)
                feature_importance = pd.DataFrame(index=columns_feature_importance)

                # Add the DataFrame to the inner dictionary
                dfs[f'feature_importance_{model}'][seed] = feature_importance
        # ----- Any data set where no feature selection takes place here (encoded, measured_7 and baseline models)

        if name_data == "abnormal" or name_data=="lems_data" or name_data=="lems_missing_data" or \
                name_data=="range_lems":
            # here the features are already selected by the user
            X = data.drop("lems_q6", axis=1)
            y = data["lems_q6"]

            # -------  Feature importance data Sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed,
                                                                stratify=data["va_lems_imputed"] > 0)

            # iterating in model_names to access the correct dictionary location
            for model in model_names:
                # Generate a DataFrame (replace this with your actual logic)
                feature_importance = pd.DataFrame(index=X_train.columns)

                # Add the DataFrame to the inner dictionary
                dfs[f'feature_importance_{model}'][seed] = feature_importance

        # here it is all in a pandas data frame, has the names still present
        va_lems_train= X_train["va_lems_imputed"]
        va_lems_test = X_test["va_lems_imputed"]

        # here it is all in a pandas data frame, has the names still present
        va_lems_train= X_train["va_lems_imputed"]
        va_lems_test = X_test["va_lems_imputed"]
        # after standard scaler it is in numpy array format, but Y remains in pd.dataframe (index present)

        # ------- Standardizing features

        if name_data == "abnormal":
            columns_to_scale = ['age', 'va_lems_imputed']

            scaler = StandardScaler()

            X_train_scaled_columns = scaler.fit_transform(X_train[columns_to_scale])
            X_test_scaled_columns= scaler.transform(X_test[columns_to_scale])
            print("Only age and va_lems_imputed are scaled ")

            X_train = X_train.drop(columns=columns_to_scale)
            X_test=X_test.drop(columns=columns_to_scale)

            X_train_scaled = np.concatenate([X_train_scaled_columns, X_train], axis=1)
            X_test_scaled = np.concatenate([X_test_scaled_columns, X_test], axis=1)

        if name_data == "all_foward_feature":
            # insert columns to scale

            to_scale = [x for x in data.columns if x.endswith(("_mean", "_median", "_min", "_max", "_range", "_times_measured_7"))] \
                       + ['age', 'va_lems_imputed']

            columns_to_scale = [feature for feature in selected_features if feature in to_scale]
            columns_encoded = [feature for feature in selected_features if feature not in to_scale]

            scaler = StandardScaler()
            X_train_scaled_columns = scaler.fit_transform(X_train[columns_to_scale])
            X_test_scaled_columns= scaler.transform(X_test[columns_to_scale])

            print(X_train_scaled_columns.shape,X_test_scaled_columns.shape)

            X_train = X_train.drop(columns=columns_to_scale)
            X_test=X_test.drop(columns=columns_to_scale)

            X_train_scaled = np.concatenate([X_train_scaled_columns, X_train], axis=1)
            X_test_scaled = np.concatenate([X_test_scaled_columns, X_test], axis=1)

        if name_data == "mean" or name_data == "median" or name_data == "min" or name_data == "max" or name_data == "range" or name_data == "measured_7" \
                or name_data == "measured_7_noise" or name_data=="lems_data" or name_data=="lems_missing_data" or name_data=="range_lems":
            sc = StandardScaler()

            X_train_scaled = sc.fit_transform(X_train)
            X_test_scaled = sc.transform(X_test)

        df_r2_performance, dfs ,score_difference_train, score_difference_test =fit_models(X_train_scaled, X_test_scaled, y_train, y_test,
                                                                                          data_type,path,df_r2_performance,seed,va_lems_train,
                                                                                          va_lems_test,score_difference_train, score_difference_test,
                                                                                          dfs,methods)

        df_r2_performance["seed"]=seed
        df_seed_performance= pd.concat([df_seed_performance, df_r2_performance], axis=0)

    merged_dfs = {}
    # ------- Merging feature importance data frames to export
    for model in model_names:
        key = f'feature_importance_{model}'

        # Extract the values (DataFrames) from the inner dictionary
        data_frames = list(dfs[key].values())

        # Concatenate the list of DataFrames along the columns
        concatenated_df = pd.concat(data_frames, axis=1, join='outer')
        feature_count = concatenated_df.copy()

        # Calculate mean and standard deviation for each row across seeds
        merged_dfs[key] = concatenated_df
        merged_dfs[key].loc['total_features_seed'] = merged_dfs[key].notnull().sum()
        merged_dfs[key]['importance_mean'] = merged_dfs[key].mean(axis=1)
        merged_dfs[key]['importance_std'] = feature_count.std(axis=1)
        merged_dfs[key]['feature_selected'] = feature_count.notnull().sum(axis=1)

    return df_seed_performance, merged_dfs, score_difference_train, score_difference_test

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
    df_seed_average=pd.DataFrame(columns = ['Model_name',"r2_train_80", "rmse_train_80","mae_train_80" ,"r2_test_20", "rmse_test_20", "mae_test_20",
                                                        "r2_train_80_m", "rmse_train_80_m","mae_train_80_m", "r2_test_20_m", "rmse_test_20_m", "mae_test_20_m", "best_parameters" ,"seed"])

    for i in range(len(models)):
        # creating a variable that stores of all column names of df_seed_performance, without the index
        selected_columns = df_seed_performance.columns[1:]

        # data frame, with all the columns of that specific model
        average_values= df_seed_performance.loc[df_seed_performance['Model_name'] == models[i]][selected_columns]

        print(average_values)

        average_values_mean = (average_values.mean()).to_list()
        average_values_std = (average_values.std()).to_list()

        # add to last row the mean and std
        df_seed_average.loc[len(df_seed_average) - 1, :] = [models[i] + "_mean"] + average_values_mean
        df_seed_average.loc[len(df_seed_average) - 1, :] = [models[i] + "_std"] + average_values_std

        print(df_seed_average)
    return df_seed_average

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
        "linear": ['Linear'],
        "lasso": ['Lasso'],
        "ridge": ['Ridge'],
        "RFR": ['RFR'],
        "SVR": ['SVR'],
        "XGB": ['XGB'],
        "GBR": ['GBR'],
        "LGBM": ['LGBM'],
        "all": ['Linear', 'Lasso', 'Ridge', 'RFR', 'SVR', 'XGB', 'GBR', 'LGBM']
    }

    return method_mapping.get(info, [])
def run_models(results_dir, data,name_data,methods,info,feature_selection, feature_amount):
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

    df_seed_performance = pd.DataFrame(
        columns=['Model_name', "r2_train_80", "rmse_train_80", "mae_train_80", "r2_test_20", "rmse_test_20",
                 "mae_test_20",
                 "r2_train_80_m", "rmse_train_80_m", "mae_train_80_m", "r2_test_20_m", "rmse_test_20_m",
                 "mae_test_20_m", "best_parameters","seed"])


    model_names = data_frames_to_create(info) # will only contain the abbreviations for the needed feature importance data frame
    # if you insert linear, will only create linear feature table
    # Construct features and labels

    score_difference_train = pd.DataFrame()
    score_difference_test = pd.DataFrame()

    path = results_dir + f"/{name_data}/{feature_selection}/{feature_amount}/"
    # Create the output folder
    Path(path).mkdir(exist_ok=True,parents=True)

    # now doing it with random states
    df_seed_performance, feature_importance, score_difference_train_seed, \
    score_difference_test_seed = seed_multiple_times( f"_{name_data}_", path, df_seed_performance,
                                                             (data["va_lems_imputed"] > 0), score_difference_train,
                                                             score_difference_test, methods, name_data,model_names, data,feature_selection,feature_amount)

    df_average_seed = average_seed(df_seed_performance)
    # creating average values across

    # exporting the random states, one with all the values and one with only the averge
    df_seed_performance.to_csv(path + f"seed_values_{name_data}_{info}.csv") # by adding info, will export different based on what we want, otherwsie will overwrite the past data frame
    # for example, seed_values_mean_linear or seed_values_mean_all
    df_average_seed.to_csv(path + f"seed_average_{name_data}_{info}.csv")

    # for the export of the feature data frames, iterate in model names, which can be just one or all and based on that export the tables present in the dictionary dfs
    for model in model_names:
            key = f'feature_importance_{model}'
            filename = f"{path}feature_importance_{model}_{name_data}_seed.csv"
            feature_importance[key].to_csv(filename)

    score_difference_train_seed.to_csv(path + f"different_train_scores_{name_data}_{info}.csv")
    score_difference_test_seed.to_csv(path + f"different_test_scores_{name_data}_{info}.csv")
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
        "XGB": [XGboost],
        "GBR": [gradient_boost_regressor],
        "LGBM": [lightGBM],
        "all": [linear_regression_RepeatedKFold, Lasso_Regression_RepeatedKFold, Ridge_Regression_RepeatedKFold,
                Random_forest_regressor, Support_vector_regressor_linear, XGboost, gradient_boost_regressor, lightGBM]
    }

    return method_mapping.get(info, []) # put in info (in our case args.model and it will assign to methods the correct model to run

# import dataframes from feature creation
data = pd.read_csv(r'')
data_missing= pd.read_csv(r'')
# output path creation, for whole document
results_dir = ''

# Create the output folderpa
Path(results_dir).mkdir(exist_ok=True)

if __name__ == '__main__':

    # initialise your parser
    parser = argparse.ArgumentParser(description="-")
    # Add the parameter positional/optional
    parser.add_argument("operation",
                        help="Which data set should we use?")  # add argument is what we are inputting in the function
    parser.add_argument('model', help='Which regression model to use, options: linear, lasso, ridge, RFR, SVR, XGB, GBR, LGBM, all?')

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

        # inputes which models we want to run
        methods= which_functions_to_run(args.model)

        run_models(results_dir, data,"mean", methods,args.model,args.selection, args.feature_amount)

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

        run_models(results_dir, data,"min",methods,args.model,args.selection, args.feature_amount)

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

        run_models(results_dir, data,"max",methods,args.model,args.selection, args.feature_amount)

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

        run_models(results_dir, data,"range",methods,args.model,args.selection, args.feature_amount)

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

        run_models(results_dir, data_missing,"measured_7",methods,args.model,args.selection, args.feature_amount)

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
        print("total features", len(columns))
        print(columns, len(columns))

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
        print(columns_to_drop)
        # Drop the identified columns
        data_missing = serological_data.drop(columns=columns_to_drop)

        columns_after_checking = [ele for ele in final_columns if ele not in columns_to_drop]

        # ---------------------- One Hot Encoding
        print("columns after dropping re-repeats", len(data_missing.columns),
              "serological marker features with info", len(columns_after_checking))
        print(columns_after_checking, len(columns_after_checking))

        # one-hot-encoding only final_columns
        data_missing = pd.get_dummies(data_missing, columns=columns_after_checking)

        print(len(data_missing.columns))
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

        run_models(results_dir, data, "lems_data",methods,args.model,args.selection, args.feature_amount)
        print("lems, done")

    if args.operation == "lems_missing":

        # adding columns that are necessary
        fixed_columns = ["age", 'va_lems_imputed', "lems_q6"]
        columns =  fixed_columns

        # copying the data set, with the columns for mean and age, va_lems and lems_q6
        data_missing = data_missing[columns].copy()
        data_missing = data_missing.dropna(subset=columns)
        print(len(data_missing))
        methods = which_functions_to_run(args.model)

        run_models(results_dir, data_missing, "lems_missing_data",methods,args.model,args.selection, args.feature_amount)

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

        print(len(data))
        methods = which_functions_to_run(args.model)

        run_models(results_dir, data,"range_lems",methods,args.model,args.selection, args.feature_amount)

        print("range_lems, done")

    if args.operation == "measured_7_noise":

        seed = 42
        np.random.seed(seed)
        # only marker's below 0.7
        data_missing.columns = [col + '_noise' if col.endswith('_measured_7') else col for col in data_missing.columns]
        marker = [x for x in data_missing.columns if x.endswith("_measured_7_noise")]

        data_missing['Noise_measured_7_noise'] = np.random.randint(0, 2, size=len(data_missing))

        # adding columns that are necessary
        fixed_columns = ["age", 'va_lems_imputed', "lems_q6", "Noise_measured_7_noise"]
        columns = marker + fixed_columns

        # copying the data set, with the columns for mean and age, va_lems and lems_q6
        data_missing = data_missing[columns].copy()
        data_missing = data_missing.dropna(subset=columns)

        methods = which_functions_to_run(args.model)

        run_models(results_dir, data_missing,"measured_7_noise",methods,args.model,args.selection, args.feature_amount)

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
        print(columns_encoded)
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
        print("encoded columns", final_columns, "to one hot encode after selection", len(columns_to_drop))
        print("encoded columns", final_columns, "to one hot encode after selection", len(columns_after_checking))
        data_total = pd.get_dummies(data_total, columns=columns_after_checking)

        print(data_total.columns.tolist())
        # -------- CONTROL
        unique_counts = data_total.nunique()
        # Identify columns where the number of unique values is 1
        columns_to_drop_check = unique_counts[unique_counts == 1].index

        if len(columns_to_drop_check) > 0:
            raise ValueError("filtered encoded did not work ")

        print(len(data_total.columns))

        methods = which_functions_to_run(args.model)

        run_models(results_dir, data_total,"all_foward_feature",methods,args.model,args.selection, args.feature_amount)

        print("all foward selection, done")