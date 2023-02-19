'''
Module that contains all unit test for the churn module.
'''

from os.path import exists
import pytest
import pandas as pd
from churn.churn_data_prep import ChurnDataPreparation


@pytest.fixture(name="cdp")
def cdp_fix(tmp_path):
    '''
    Churn Data Processing object to be used in the tests.
    Input tmp_path: pytest buildin temp path
    '''
    return ChurnDataPreparation(tmp_path, tmp_path)


@pytest.fixture(name="correct_pth")
def correct_pth_fix():
    '''
    Test input data: correct path to a data file.
    '''
    return "./data/bank_data.csv"


@pytest.fixture(name="incorrect_pth")
def incorrect_pth_fix():
    '''
    Test input data: incorrect path to a data file.
    '''
    return "does_not_exist.csv"


@pytest.fixture(name="sample_churn_df")
def sample_churn_df_fix():
    '''
    Test input data: sample churn df
    '''
    sample_data = {
        "CLIENTNUM": [768805383, 818770008],
        "Attrition_Flag": ["Existing Customer", "Existing Customer"],
        "Customer_Age": [45, 49],
        "Gender": ["M", "F"],
        "Dependent_count": [3, 5],
        "Education_Level": ["High School", "Graduate"],
        "Marital_Status": ["Married", "Single"],
        "Income_Category": ["$60K - $80K", "Less than $40K"],
        "Card_Category": ["Blue", "Blue"],
        "Months_on_book": [39, 44],
        "Total_Relationship_Count": [5, 6],
        "Months_Inactive_12_mon": [1, 1],
        "Contacts_Count_12_mon": [3, 2],
        "Credit_Limit": [12691.0, 8256.0],
        "Total_Revolving_Bal": [777, 864],
        "Avg_Open_To_Buy": [11914.0, 7392.0],
        "Total_Amt_Chng_Q4_Q1": [1.335, 1.541],
        "Total_Trans_Amt": [1144, 1291],
        "Total_Trans_Ct": [42, 33],
        "Total_Ct_Chng_Q4_Q1": [1.625, 3.714],
        "Avg_Utilization_Ratio": [0.061, 0.105]
    }
    return pd.DataFrame(data=sample_data)


def test_import(cdp, correct_pth):
    '''
    test data import - this example is completed for you to assist with the other test functions
    '''
    churn_df = cdp.import_data(correct_pth)
    assert churn_df is not None


def test_import_incorrect_pth(cdp, incorrect_pth):
    '''
    test data import - this example is completed for you to assist with the other test functions
    '''
    churn_df = cdp.import_data(incorrect_pth)
    assert churn_df is None


def test_eda(cdp, sample_churn_df):
    '''
    test data import - this example is completed for you to assist with the other test functions
    '''
    cdp.perform_eda(sample_churn_df)
    assert exists(f"{cdp.doc_pth}/customer_age.png")
    assert exists(f"{cdp.doc_pth}/transaction_count.png")
    assert exists(f"{cdp.doc_pth}/correlations.png")


#
# def blatest_encoder_helper(encoder_helper):
#    '''
#    test encoder helper
#    '''
#
#
# def blatest_perform_feature_engineering(perform_feature_engineering):
#    '''
#    test perform_feature_engineering
#    '''
#
#
# def blatest_train_models(train_models):
#    '''
#    test train_models
#    '''
