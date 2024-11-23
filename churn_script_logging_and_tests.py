"""
Author: Ivan Gorban
Date: 2024-11-22

Testing module for churn_library.py
"""
import os
import logging
import pytest
import churn_library as cls
from config import KEEP_COLUMNS, CAT_COLUMNS

logging.basicConfig(
    filename='logs/churn_library.log',
    level = logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')

def test_import():
    '''
	test data import - this example is completed for you to assist with the other test functions
	'''
    try:
        data = cls.import_data("./data/bank_data.csv")
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert data.shape[0] > 0
        assert data.shape[1] > 0
    except AssertionError as err:
        logging.error("Testing import_data: The file doesn't appear to have rows and columns")
        raise err

@pytest.fixture
def dat():
    '''
    returns a dataframe
    '''
    data = cls.import_data('./data/bank_data.csv')
    data = cls.target_creation(data)
    return data

@pytest.fixture
def categories():
    '''
    returns a list of columns
    '''
    return CAT_COLUMNS

def test_eda(dat):
    '''
    test perform eda function
    '''
    df = dat
    try:
        cls.perform_eda(df)
        logging.info("INFO: Testing perform_eda: SUCCESS")
        assert os.path.exists('./images')
        logging.info("INFO: Testing perform_eda: The images folder was created")
        assert os.path.exists('./images/churn_distribution.png')
        logging.info("INFO: Testing perform_eda: The age histogram was created")
        assert os.path.exists('./images/customer_age_distribution.png')
        logging.info("INFO: Testing perform_eda: The heatmap was created")
        assert os.path.exists('./images/marital_status_distribution.png')
        logging.info("INFO: Testing perform_eda: The heatmap target was created")
        assert os.path.exists('./images/total_trans_ct_distribution.png')
        logging.info("INFO: Testing perform_eda: The heatmap target was created")
        assert os.path.exists('./images/correlation_heatmap.png')
        logging.info("INFO: Testing perform_eda: The heatmap target was created")
    except Exception as err:
        logging.error("ERROR: Testing perform_eda: The function didn't run successfully")
        raise err

def test_encoder_helper(dat, categories):
    '''
    test encoder helper
    '''
    df = dat
    category_lst = categories
    try:
        data = cls.encoder_helper(df, category_lst)
        logging.info("INFO: Testing encoder_helper: SUCCESS")
        logging.info("INFO: dataframe contains columns: %s", data.columns)
        assert data.shape[0] == df.shape[0]
        logging.info("INFO: Testing encoder_helper: The number of rows is the same")
        assert set(colname+'_encoded' for colname in category_lst).issubset(data.columns)
        logging.info("INFO: Testing encoder_helper: The columns were encoded")
    except Exception as err:
        logging.error("ERROR: Testing encoder_helper: The function didn't run successfully")
        raise err


def test_perform_feature_engineering(dat, categories):
    '''
    test perform_feature_engineering
    '''
    df = dat
    category_lst = categories
    try:
        df = cls.encoder_helper(df, category_lst)
        result = cls.perform_feature_engineering(df)
        logging.info("INFO: Testing perform_feature_engineering: SUCCESS")
        assert len(result) == 4
        logging.info("INFO: Testing perform_feature_engineering: \
                     The number of dataframes is correct")
        assert result[0].shape[0] + result[1].shape[0] == df.shape[0]
        logging.info("INFO: Testing perform_feature_engineering: \
                     The number of rows is the same")
        assert result[0].shape[1] == result[1].shape[1] == len(KEEP_COLUMNS)
        logging.info("INFO: Testing perform_feature_engineering: \
                     The number of columns is correct")
        assert result[1].shape[0] + result[2].shape[0] == df.shape[0]
        logging.info("INFO: Testing perform_feature_engineering: \
                     The number of rows is the same in target")
    except Exception as err:
        logging.error("ERROR: Testing perform_feature_engineering: \
                      The function didn't run successfully")
        raise err


def test_train_models(dat, categories):
    '''
    test train_models
    '''
    df = dat
    cat_columns = categories
    try:
        df = cls.encoder_helper(df, cat_columns)
        x_train, x_test, y_train, y_test = cls.perform_feature_engineering(df)
        cls.train_models(x_train, x_test, y_train, y_test)
        logging.info("INFO: Testing train_models: SUCCESS")
        assert os.path.exists('./images')
        logging.info("INFO: Testing train_models: The images folder was created")
        assert os.path.exists('./images/roc_curve.png')
        logging.info("INFO: Testing train_models: The roc curve was created")
        assert os.path.exists('./images/rfc_classification_report.png')
        logging.info("INFO: Testing train_models: The classification report \
                     was created for random forest")
        assert os.path.exists('./images/lrc_classification_report.png')
        logging.info("INFO: Testing train_models: The classification report \
                     was created for logistic regression")
        assert os.path.exists('./models')
        logging.info("INFO: Testing train_models: The models folder was created")
        assert os.path.exists('./models/rfc_model.pkl')
        logging.info("INFO: Testing train_models: The random forest model was created")
        assert os.path.exists('./models/lrc_model.pkl')
        logging.info("INFO: Testing train_models: The logistic regression model was created")
    except Exception as err:
        logging.error("ERROR: Testing train_models: The function didn't run successfully")
        raise err


if __name__ == "__main__":
    pass
