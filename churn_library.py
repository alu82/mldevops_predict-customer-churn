"""
Wrapper file that forwards call to churn module

Author: alu82
Date: February 2023
"""

from churn import churn_data_prep as cdp
from churn import churn_modelling as cmod

LOG_PTH = "./logs/results.log"
DOC_PTH = "./docs"
MODEL_PTH = "./models"
churn_data_prep_step = cdp.ChurnDataPreparation(LOG_PTH, DOC_PTH)
churn_modelling_step = cmod.ChurnModelling(LOG_PTH, DOC_PTH, MODEL_PTH)


def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    '''
    return churn_data_prep_step.import_data(pth)


def perform_eda(churn_df):
    '''
    perform eda on df and save figures to images folder
    input:
            churn_df: pandas dataframe

    output:
            None
    '''
    churn_data_prep_step.perform_eda(churn_df)


def encoder_helper(churn_df, category_lst, response):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            churn_df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could be
            used for naming variables or index y column]

    output:
            df: pandas dataframe with new columns for
    '''
    return churn_data_prep_step.encoder_helper(
        churn_df, category_lst, response)


def perform_feature_engineering(churn_df, response):
    '''
    input:
              df: pandas dataframe
              response: string of response name
              [optional argument that could be used for naming variables or index y column]

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''
    return churn_data_prep_step.perform_feature_engineering(churn_df, response)


def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    '''
    churn_modelling_step.classification_report_image(
        y_train,
        y_test,
        y_train_preds_lr,
        y_train_preds_rf,
        y_test_preds_lr,
        y_test_preds_rf)


def feature_importance_plot(model, x_data):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            x_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    churn_modelling_step.feature_importance_plot(model, x_data)


def train_models(x_train, x_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''
    churn_modelling_step.train_models(x_train, x_test, y_train, y_test)
