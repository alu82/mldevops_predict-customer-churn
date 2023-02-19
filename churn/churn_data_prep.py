'''
Module for retrieving, cleaning and preprocessing the data used for predict customer churn.
'''

import logging
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split


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
        self.logger.info("Histogram for customer created in %s.", self.doc_pth)

        plt.figure(figsize=(20, 10))
        sns.histplot(churn_df['Total_Trans_Ct'], stat='density', kde=True)
        plt.savefig(f"{self.doc_pth}/transaction_count.png")
        self.logger.info(
            "Histogram for Total_Trans_Ct created in %s.", self.doc_pth)

        plt.figure(figsize=(20, 10))
        sns.heatmap(churn_df.corr(), annot=False, cmap='Dark2_r', linewidths=2)
        plt.savefig(f"{self.doc_pth}/correlations.png")
        self.logger.info("Correlation matrix created in %s.", self.doc_pth)

    def encoder_helper(self, churn_df, category_lst, response):
        '''
        helper function to turn each categorical column into a new column with
        propotion of churn for each category - associated with cell 15 from the notebook
        input:
                churn_df: pandas dataframe
                category_lst: list of columns that contain categorical features
                response: string of response name [optional argument that could be used
                for naming variables or index y column]
        output:
                churn_df: pandas dataframe with new columns
        '''
        for category in category_lst:
            self.logger.info("Creating proportion for column %s.", category)
            try:
                cat_group = churn_df.groupby(category).mean()[response]
                churn_df[f"{category}_{response}"] = churn_df.apply(
                    lambda row: cat_group.loc[row[category]], axis=1)
                self.logger.info(
                    "Creating proportion for column %s. successful", category)
            except KeyError:
                self.logger.warning(
                    "Creating proportion for column %s. failed. No such column found.", category)

        return churn_df

    def perform_feature_engineering(self, churn_df, response="Churn"):
        '''
        input:
             churn_df: pandas dataframe
             response: string of response name [optional argument that could be used
             for naming variables or index y column]
         output:
             X_train: X training data
             X_test: X testing data
             y_train: y training data
             y_test: y testing data
        '''

        self.logger.info("Creating label column %s.", response)
        churn_df[response] = churn_df['Attrition_Flag'].apply(
            lambda val: 0 if val == "Existing Customer" else 1)
        churn_label = churn_df[response]
        self.logger.info("Creating label column %s successful", response)

        cat_columns = [
            'Gender',
            'Education_Level',
            'Marital_Status',
            'Income_Category',
            'Card_Category'
        ]
        churn_df = self.encoder_helper(churn_df, cat_columns, response)

        keep_cols = [
            'Customer_Age', 'Dependent_count', 'Months_on_book',
            'Total_Relationship_Count', 'Months_Inactive_12_mon',
            'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
            'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
            'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio',
            f'Gender_{response}', f'Education_Level_{response}', f'Marital_Status_{response}',
            f'Income_Category_{response}', f'Card_Category_{response}'
        ]
        self.logger.info("Keeping the following columns %s.", keep_cols)

        return train_test_split(churn_df[keep_cols], churn_label, test_size=0.3, random_state=42)
