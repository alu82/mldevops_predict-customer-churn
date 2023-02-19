"""
Module for retrieving, cleaning and preprocessing the data used for predict customer churn.
"""

import pandas as pd
import logging


class ChurnDataPreparation:
    def __init__(self, log_pth):
        logging.basicConfig(
            filename=log_pth,
            level=logging.INFO,
            filemode='w',
        )
        self.logger = logging.getLogger(__name__)

    def import_data(self, pth):
        '''
        returns dataframe for the csv found at pth

        input:
                pth: the path to the csv
        output:
                df: pandas dataframe or None if the file at the given path could not be found.
        '''
        self.logger.info(f"Reading file at path {pth}.")
        
        df = None
        try:
            df = pd.read_csv(f"{pth}")
            self.logger.info(f"Reading file at path {pth} successful.")
        except FileNotFoundError as err:
            self.logger.error(f"Reading file at path {pth} failed.")
        return df


def perform_eda(df):
    '''
    perform eda on df and save figures to images folder
    input:
            df: pandas dataframe

    output:
            None
    '''
    pass


def encoder_helper(df, category_lst, response):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
            df: pandas dataframe with new columns for
    '''
    pass


def perform_feature_engineering(df, response):
    '''
    input:
              df: pandas dataframe
              response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''
