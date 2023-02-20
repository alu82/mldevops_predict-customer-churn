'''
Module for testing and logging the churn solution.

Author: alu82
Date: February 2023
'''

# Logger configuration is used from the churn module
import logging
import pandas as pd
import churn_library as cls


def test_import():
    '''
    test data import - this example is completed for you to assist with the other test functions
    '''
    try:
        churn_df = cls.import_data("./data/bank_data.csv")
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert churn_df.shape[0] > 0
        assert churn_df.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err
    return churn_df


def test_eda(churn_df):
    '''
    test perform eda function
    '''
    logging.info("Performing eda for churn.")
    try:
        assert isinstance(churn_df, pd.DataFrame)
    except AssertionError as err:
        logging.error("The input type is not of type pd.DataFrame")
        raise err

    cls.perform_eda(churn_df)
    logging.info("Performing eda for churn successful.")


def test_encoder_helper(churn_df, cat_columns, response):
    '''
    test encoder helper
    '''
    logging.info("Performing encoding for churn.")
    old_length = len(churn_df.columns)
    try:
        assert isinstance(churn_df, pd.DataFrame)
        assert isinstance(cat_columns, list)
        assert isinstance(response, str)
    except AssertionError as err:
        logging.error("The input has the wrong type.")
        raise err

    encoded_churn_df = cls.encoder_helper(churn_df, cat_columns, response)

    try:
        assert old_length < len(encoded_churn_df.columns)
    except AssertionError as err:
        logging.error("No new columns, encoding failed. %s %s",
                      old_length, len(encoded_churn_df.columns))
        raise err

    logging.info("Performing encoding for churn successful.")


def test_perform_feature_engineering(churn_df):
    '''
    test perform_feature_engineering
    '''
    logging.info("Performing feature engineering for churn.")
    try:
        assert isinstance(churn_df, pd.DataFrame)
    except AssertionError as err:
        logging.error("The input has the wrong type.")
        raise err

    x_train, x_test, y_train, y_test = cls.perform_feature_engineering(
        churn_df, response="Churn")

    try:
        assert len(x_train) > 0
        assert len(x_test) > 0
        assert len(y_train) > 0
        assert len(y_test) > 0
    except AssertionError as err:
        logging.error("Feature engineering produced empty data: %s %s %s %s.",
                      len(x_train), len(x_test), len(y_train), len(y_test))
        raise err

    return x_train, x_test, y_train, y_test


def test_train_models(x_train, x_test, y_train, y_test):
    '''
    test train_models
    '''
    logging.info("Training model now.")
    try:
        assert isinstance(x_train, pd.DataFrame)
        assert isinstance(x_test, pd.DataFrame)
        assert isinstance(y_train, pd.Series)
        assert isinstance(y_test, pd.Series)
    except AssertionError as err:
        logging.error("The input has the wrong type: %s %s %s %s:",
                      type(x_train), type(x_test), type(y_train), type(y_test))
        raise err

    cls.train_models(x_train, x_test, y_train, y_test)
    logging.info("Training model successful.")


if __name__ == "__main__":
    CAT_COLUMNS = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category'
    ]

    CHURN_DF = test_import()
    test_eda(CHURN_DF)

    # to test the encoding function, we have to create a dummy dataset.
    # this is because the Churn column is only added in the feature
    # engineering function but it is needed when encoding
    CHURN_DF_DUMMY = CHURN_DF.copy()
    CHURN_DF_DUMMY["Churn"] = 1
    test_encoder_helper(CHURN_DF_DUMMY, CAT_COLUMNS, "Churn")

    X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = test_perform_feature_engineering(
        CHURN_DF)
    test_train_models(X_TRAIN, X_TEST, Y_TRAIN, Y_TEST)
