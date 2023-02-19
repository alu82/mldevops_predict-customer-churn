'''
Module for retrieving, cleaning and preprocessing the data used for predict customer churn.
'''

import logging
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns


class ChurnDataPreparation:
    '''
    Class that encapsulates all the data transformation logic.
    '''

    def __init__(self, log_pth, doc_pth):
        sns.set()
        logging.basicConfig(
            filename=log_pth,
            level=logging.INFO,
            filemode='w',
        )
        self.logger = logging.getLogger(__name__)
        self.doc_pth = doc_pth

    def import_data(self, pth):
        '''
        returns dataframe for the csv found at pth

        input:
                pth: the path to the csv
        output:
                df: pandas dataframe or None if the file at the given path could not be found.
        '''
        self.logger.info("Reading file at path %s.", pth)

        imported_df = None
        try:
            imported_df = pd.read_csv(f"{pth}")
            self.logger.info("Reading file at path %s successful.", pth)
        except FileNotFoundError:
            self.logger.error("Reading file at path %s failed.", pth)
        return imported_df

    def perform_eda(self, churn_df):
        '''
        performs eda on churn dataframe and saves figures to docs folder
        Input:
            churn_df (dataframe): raw churn data
        '''
        plt.figure(figsize=(20, 10))
        churn_df['Customer_Age'].hist()
        plt.savefig(f"{self.doc_pth}/customer_age.png")

        plt.figure(figsize=(20, 10))
        sns.histplot(churn_df['Total_Trans_Ct'], stat='density', kde=True)
        plt.savefig(f"{self.doc_pth}/transaction_count.png")

        plt.figure(figsize=(20, 10))
        sns.heatmap(churn_df.corr(), annot=False, cmap='Dark2_r', linewidths=2)
        plt.savefig(f"{self.doc_pth}/correlations.png")


#
# def encoder_helper(df, category_lst, response):
#    '''
#    helper function to turn each categorical column into a new column with
#    propotion of churn for each category - associated with cell 15 from the notebook
#
#    input:
#            df: pandas dataframe
#            category_lst: list of columns that contain categorical features
#            response: string of response name [optional argument that could be used
#            for naming variables or index y column]
#
#    output:
#            df: pandas dataframe with new columns for
#    '''
#    pass
#
#
# def perform_feature_engineering(df, response):
#    '''
#    input:
#              df: pandas dataframe
#              response: string of response name [optional argument that could be used
#               for naming variables or index y column]
#
#    output:
#              X_train: X training data
#              X_test: X testing data
#              y_train: y training data
#              y_test: y testing data
#    '''
#
