'''
Defines methods to create a churn pipeline.
Can be called via the main method directly for testing reasons.

Author: alu82
Date: February 2023
'''
from churn import churn_data_prep as cdp
from churn import churn_modelling as cmod


def do_data_preparation(log_pth, data_pth, doc_pth):
    '''
    Method that does the data preparation
    '''
    churn_data_prep_step = cdp.ChurnDataPreparation(log_pth, doc_pth)
    churn_df = churn_data_prep_step.import_data(data_pth)
    if churn_df is not None:
        churn_data_prep_step.perform_eda(churn_df)
        return churn_data_prep_step.perform_feature_engineering(churn_df)
    return None


def do_modelling(log_pth, doc_pth, model_pth, x_train, x_test, y_train, y_test):
    '''
    Method that does the modelling
    '''
    churn_modelling_step = cmod.ChurnModelling(log_pth, doc_pth, model_pth)
    churn_modelling_step.train_models(x_train, x_test, y_train, y_test)


if __name__ == '__main__':
    DATA_PTH = "./data/bank_data.csv"
    LOG_PTH = "./logs/results.log"
    DOC_PTH = "./docs"
    MODEL_PTH = "./models"
    X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = do_data_preparation(
        LOG_PTH, DATA_PTH, DOC_PTH)

    do_modelling(LOG_PTH, DOC_PTH, MODEL_PTH, X_TRAIN, X_TEST, Y_TRAIN, Y_TEST)
