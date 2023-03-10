'''
Module for model training and save model metrics used for predict customer churn.

Author: alu82
Date: February 2023
'''

import logging
import shap
import joblib
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import plot_roc_curve, classification_report


class ChurnModelling:
    '''
    Class that encapuslates model training and documenting model metrics
    '''

    def __init__(self, log_pth, doc_pth, model_pth):
        sns.set()
        logging.basicConfig(
            filename=log_pth,
            level=logging.INFO,
            filemode='w',
        )
        self.logger = logging.getLogger(__name__)
        self.doc_pth = doc_pth
        self.model_pth = model_pth

    def train_models(self, x_train, x_test, y_train, y_test):
        '''
        train, store model results: images + scores, and store models
        input:
               x_train: X training data
               x_test: X testing data
               y_train: y training data
               y_test: y testing data
        output:
                None
        '''
        # Train and evaluate different models (using some helper methods
        rf_estimator, y_train_preds_rf, y_test_preds_rf = self.train_test_predict_rf(
            x_train, y_train, x_test)
        lr_estimator, y_train_preds_lr, y_test_preds_lr = self.train_test_predict_lr(
            x_train, y_train, x_test)

        # document some metrics and model training outputs
        self.classification_report_image(
            y_train, y_test, y_train_preds_lr, y_train_preds_rf, y_test_preds_lr, y_test_preds_rf)
        self.roc_image(x_test, y_test, lr_estimator, rf_estimator)
        self.feature_importance_plot(rf_estimator.best_estimator_, x_test)

        # save the best models
        joblib.dump(rf_estimator.best_estimator_,
                    f"{self.model_pth}/rfc_model.pkl")
        joblib.dump(lr_estimator, f"{self.model_pth}/logistic_model.pkl")

    def classification_report_image(
            self, y_train, y_test,
            y_train_preds_lr, y_train_preds_rf,
            y_test_preds_lr, y_test_preds_rf):
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
        self.save_classification_report(
            y_test, y_test_preds_rf, "rf_test_classification_report")
        self.save_classification_report(
            y_train, y_train_preds_rf, "rf_train_classification_report")
        self.save_classification_report(
            y_test, y_test_preds_lr, "lr_test_classification_report")
        self.save_classification_report(
            y_train, y_train_preds_lr, "lr_train_classification_report")

    def roc_image(self, x_test, y_test, lr_estimator, rf_estimator):
        '''
        Saves roc curve of both estimators as image.
        Input:
            x_test: X testing data
            y_test: y testing data
            lr_estimator: trained logistic regression estimator
            rf_estimator: trained random forrest estimator
        '''
        self.logger.info("Creating roc curves and saving them.")
        plt.figure(figsize=(15, 8))
        axs = plt.gca()
        plot_roc_curve(rf_estimator, x_test, y_test, ax=axs, alpha=0.8)
        plot_roc_curve(lr_estimator, x_test, y_test, ax=axs, alpha=0.8)
        plt.savefig(f"{self.doc_pth}/roc_curve.png")
        self.logger.info("Creating roc curves and saving them successful.")

    def feature_importance_plot(self, model, x_data):
        '''
        creates and stores the feature importances in pth
        input:
                model: model object containing feature_importances_
                X_data: pandas dataframe of X values
        output:
                 None
        '''
        self.logger.info("Creating feature importance image.")
        plt.figure(figsize=(15, 8))
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(x_data)
        shap.summary_plot(shap_values, x_data, plot_type="bar", show=False)
        plt.savefig(f"{self.doc_pth}/feature_importance.png")

    def train_test_predict_rf(self, x_train, y_train, x_test):
        '''
        Helper method, that trains a Random Forrest classifier using a grid search.
        Input:
            x_train: X training data
            x_test: X testing data
            y_train: y training data
            y_test: y testing data
        Return:
            y_train_preds: predictions made with train data
            y_test_preds: predictions made with (unseen) test data
        '''
        self.logger.info("Training a random forest classifier.")
        rfc = RandomForestClassifier(random_state=42)
        param_grid = {
            'n_estimators': [200, 500],
            'max_features': ['auto', 'sqrt'],
            'max_depth': [4, 5, 100],
            'criterion': ['gini', 'entropy']
        }
        cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
        return self.train_classifier(cv_rfc, x_train, y_train, x_test)

    def train_test_predict_lr(self, x_train, y_train, x_test):
        '''
        Helper method, that trains a Logistic Regression Classifier.
        Input:
            x_train: X training data
            x_test: X testing data
            y_train: y training data
            y_test: y testing data
        Return:
            y_train_preds: predictions made with train data
            y_test_preds: predictions made with (unseen) test data
        '''
        # Use a different solver if the default 'lbfgs' fails to converge
        # Reference: https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
        self.logger.info("Training a rlogistic regression classifier.")
        lrc = LogisticRegression(solver='lbfgs', max_iter=3000)
        return self.train_classifier(lrc, x_train, y_train, x_test)

    def train_classifier(self, clf, x_train, y_train, x_test):
        '''
        Helper method, that takes a "fittable" classifier. It fits the classifier and does some
        predictions.
        Input:
            x_train: X training data
            x_test: X testing data
            y_train: y training data
            y_test: y testing data
        Return:
            y_train_preds: predictions made with train data
            y_test_preds: predictions made with (unseen) test data
        '''
        clf.fit(x_train, y_train)
        self.logger.info("Training successful. Starting predicting.")
        y_train_preds = clf.predict(x_train)
        y_test_preds = clf.predict(x_test)
        self.logger.info("Predicting successful.")
        return clf, y_train_preds, y_test_preds

    def save_classification_report(self, y_correct, y_pred, name):
        '''
        Helper method to create a classification report and save it to the docs folder
        Input:
            name: Name of the classification report (filename where the report is stored)
            y_correct: correct labels
            y_pred: predicted labels
        '''
        # Inspired from here:
        # https://stackoverflow.com/questions/28200786/how-to-plot-scikit-learn-classification-report
        self.logger.info("Creating classification report %s.", name)
        clf_report = classification_report(y_correct, y_pred, output_dict=True)
        plt.figure(figsize=(20, 10))
        sns.heatmap(pd.DataFrame(
            clf_report).iloc[:-1, :].T, annot=True, vmin=0, vmax=1)
        plt.savefig(f"{self.doc_pth}/{name}.png")
        self.logger.info("Creating classification report %s successful.", name)
