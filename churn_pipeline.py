from churn import churn_data_prep as cdp


def do_data_preparation(log_pth, data_pth):
    churn_data_prep_step = cdp.ChurnDataPreparation(log_pth)
    df = churn_data_prep_step.import_data(data_pth)
    if df is not None:
        print(df.head())


if __name__ == '__main__':
    churn_data_prep_step = cdp.ChurnDataPreparation("./logs/results.log")
    data_pth = "./data/bank_data.csv"
    log_pth = "./log/results.log"
    do_data_preparation(log_pth, data_pth)
