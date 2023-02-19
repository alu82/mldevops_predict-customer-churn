'''
Defines methods to create a churn pipeline.
Can be called via the main method directly for testing reasons.
'''
from churn import churn_data_prep as cdp


def do_data_preparation(log_pth, data_pth, doc_pth):
    '''
    Method that does the data preparation
    '''
    churn_data_prep_step = cdp.ChurnDataPreparation(log_pth, doc_pth)
    churn_df = churn_data_prep_step.import_data(data_pth)
    if churn_df is not None:
        churn_data_prep_step.perform_eda(churn_df)


if __name__ == '__main__':
    DATA_PTH = "./data/bank_data.csv"
    LOG_PTH = "./logs/results.log"
    DOC_PTH = "./docs"
    do_data_preparation(LOG_PTH, DATA_PTH, DOC_PTH)
