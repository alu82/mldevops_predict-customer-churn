import pytest
from churn.churn_data_prep import ChurnDataPreparation


@pytest.fixture
def cdp():
    log_pth = "./logs/test_results.log" # in the current implementation nothing will be logged, because pytest catches all log messages
    return ChurnDataPreparation(log_pth)


@pytest.fixture
def correct_pth():
    return "./data/bank_data.csv"


@pytest.fixture
def incorrect_pth():
    return "does_not_exist.csv"


def test_import(cdp, correct_pth):
    '''
    test data import - this example is completed for you to assist with the other test functions
    '''
    df = cdp.import_data(correct_pth)
    assert df is not None


def test_import_incorrect_pth(cdp, incorrect_pth):
    '''
    test data import - this example is completed for you to assist with the other test functions
    '''
    df = cdp.import_data(incorrect_pth)
    assert df is None


def blatest_eda(perform_eda):
    '''
    test perform eda function
    '''


def blatest_encoder_helper(encoder_helper):
    '''
    test encoder helper
    '''


def blatest_perform_feature_engineering(perform_feature_engineering):
    '''
    test perform_feature_engineering
    '''


def blatest_train_models(train_models):
    '''
    test train_models
    '''
