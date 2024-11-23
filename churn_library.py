# library doc string
"""
Refactored library for churn prediction
Author: Ivan Gorban
Date: 2024-11-22
"""

# import libraries
import os
import logging
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import RocCurveDisplay, classification_report
import shap
from config import KEEP_COLUMNS, CAT_COLUMNS, QUANT_COLUMNS, PARAM_GRID

os.environ['QT_QPA_PLATFORM']='offscreen'

logging.basicConfig(
        filename='logs/churn_library.log',
        level=logging.INFO,
        format='%(asctime)s:%(levelname)s:%(message)s'
)

def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    '''
    logging.info('INFO: Importing data')
    try:
        data = pd.read_csv(pth)
        logging.info('SUCCESS: Data imported successfully')
        return data
    except FileNotFoundError:
        logging.error('ERROR: File not found')
        return None


def target_creation(df):
    '''
    creates target column for churn prediction

    input:
            df: pandas dataframe
    output:
            df: pandas dataframe with new column 'churn' based on 'Attrition_Flag'
    '''
    logging.info('INFO: Creating target column')
    df['Churn'] = df['Attrition_Flag'].apply(lambda val: 0 if val == "Existing Customer" else 1)
    df.drop('Attrition_Flag', axis=1, inplace=True)
    return df


def perform_eda(df):
    '''
    perform eda on df and save figures to images folder
    input:
            df: pandas dataframe

    output:
            None
    '''
    logging.info('INFO: Performing EDA')
    logging.info('INFO: Shape of the data: %s',df.shape)

    missing_data = df.isnull().sum()

    logging.info('INFO: Missing data columns: %s', missing_data[missing_data > 0])
    logging.info('INFO: Data description: %s', df.describe())
    # create images folder if it does not exist
    if not os.path.exists('./images'):
        os.makedirs('./images')
    logging.info('INFO: Creating images of EDA')
    #Churn distribution
    plt.figure(figsize=(20,10))
    df['Churn'].hist()
    plt.savefig('./images/churn_distribution.png')
    # Customer age distribution
    plt.figure(figsize=(20,10))
    df['Customer_Age'].hist()
    plt.savefig('./images/customer_age_distribution.png')
    # Marital status distribution
    plt.figure(figsize=(20,10))
    df.Marital_Status.value_counts('normalize').plot(kind='bar')
    plt.savefig('./images/marital_status_distribution.png')
    # Total trans cat distribution
    plt.figure(figsize=(20,10))
    sns.histplot(df['Total_Trans_Ct'], stat='density', kde=True)
    plt.savefig('./images/total_trans_ct_distribution.png')
    # Correlation heatmap
    plt.figure(figsize=(20,10))
    sns.heatmap(df[QUANT_COLUMNS+['Churn']].corr(), annot=False, cmap='Dark2_r', linewidths=2)
    plt.savefig('./images/correlation_heatmap.png')


def encoder_helper(df, category_lst):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features

    output:
            df: pandas dataframe with new columns for
    '''
    logging.info('INFO: Encoding categorical columns')
    for category in category_lst:
        churn_rate = df.groupby(category)['Churn'].mean()
        df[f'{category}_encoded'] = df[category].map(churn_rate)

    return df

def perform_feature_engineering(df):
    '''
    input:
              df: pandas dataframe

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''

    y = df['Churn']
    x = df[KEEP_COLUMNS]

    return train_test_split(x, y, test_size= 0.3, random_state=42)


def classification_report_image(targets, predictions_lr, predictions_rf):
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
    y_tr, y_tst = targets
    y_tr_preds_lr, y_tst_preds_lr = predictions_lr
    y_tr_preds_rf, y_tst_preds_rf = predictions_rf
    plt.rc('figure', figsize=(5, 5))
    plt.text(0.01, 1.25, str('Random Forest Train'), {'fontsize': 10}, fontproperties = 'monospace')
    plt.text(0.01, 0.05, str(classification_report(y_tst, y_tst_preds_rf)),
             {'fontsize': 10}, fontproperties = 'monospace')
    plt.text(0.01, 0.6, str('Random Forest Test'), {'fontsize': 10}, fontproperties = 'monospace')
    plt.text(0.01, 0.7, str(classification_report(y_tr, y_tr_preds_rf)),
             {'fontsize': 10}, fontproperties = 'monospace')
    plt.axis('off')
    plt.savefig('./images/rfc_classification_report.png')

    plt.rc('figure', figsize=(5, 5))
    plt.text(0.01, 1.25, str('Logistic Regression Train'),
             {'fontsize': 10}, fontproperties = 'monospace')
    plt.text(0.01, 0.05, str(classification_report(y_tr, y_tr_preds_lr)),
             {'fontsize': 10}, fontproperties = 'monospace')
    plt.text(0.01, 0.6, str('Logistic Regression Test'),
             {'fontsize': 10}, fontproperties = 'monospace')
    plt.text(0.01, 0.7, str(classification_report(y_tst, y_tst_preds_lr)),
             {'fontsize': 10}, fontproperties = 'monospace')
    plt.axis('off')
    plt.savefig('./images/lrc_classification_report.png')


def feature_importance_plot(model, x_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    explainer = shap.Explainer(model, x_data)
    shap_values = explainer(x_data)
    shap.summary_plot(shap_values, x_data, plot_type="bar")
    plt.savefig(output_pth)

def train_models(x_tr, x_tst, y_tr, y_tst):
    '''
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''
    logging.info('INFO: Training models')
    rfc = RandomForestClassifier(random_state=42)
    lrc = LogisticRegression(solver='lbfgs', max_iter=3000)

    cv_rfc = GridSearchCV(estimator=rfc, param_grid=PARAM_GRID, cv=5)
    cv_rfc.fit(x_tr, y_tr)

    lrc.fit(x_tr, y_tr)
    logging.info('INFO: Best parameters for Random Forest: %s', cv_rfc.best_params_)
    logging.info('INFO: Predicting on training and testing data RF')
    y_train_preds_rf = cv_rfc.best_estimator_.predict(x_tr)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(x_tst)

    logging.info('INFO: Predicting on training and testing data LR')
    y_train_preds_lr = lrc.predict(x_tr)
    y_test_preds_lr = lrc.predict(x_tst)

    logging.info('Saving metrics')

    if not os.path.exists('./images'):
        os.makedirs('./images')

    classification_report_image((y_tr, y_tst), (y_train_preds_lr, y_test_preds_lr),
                                (y_train_preds_rf,  y_test_preds_rf))

    plt.figure(figsize=(15, 8))
    ax = plt.gca()
    RocCurveDisplay.from_estimator(lrc, x_tst, y_tst, ax=ax)
    RocCurveDisplay.from_estimator(cv_rfc.best_estimator_, x_tst, y_tst, ax=ax)
    plt.savefig('./images/roc_curve.png')

    save_models(cv_rfc.best_estimator_, 'rfc_model.pkl')
    save_models(lrc, 'lrc_model.pkl')


def save_models(model, pth):
    '''
    save model to pth
    input:
            model: trained model object
            pth: path to save model
    output:
            None
    '''
    logging.info('INFO: Saving model in %s', pth)
    models_path = './models'
    if not os.path.exists(models_path):
        os.makedirs(models_path)

    with open(models_path + '/' + pth, 'wb') as file:
        pickle.dump(model, file)

if __name__ == '__main__':
    logging.info('INFO: Running main function')
    dat = import_data('data/BankChurners.csv')
    dat = target_creation(dat)
    perform_eda(dat)
    dat = encoder_helper(dat, CAT_COLUMNS)
    X_train, X_test, y_train, y_test = perform_feature_engineering(dat)
    train_models(X_train, X_test, y_train, y_test)
    logging.info('INFO: Finished running main function')
