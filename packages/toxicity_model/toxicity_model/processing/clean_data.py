import pandas as pd
import warnings
import time
from tqdm.auto import tqdm
import os
import numpy as np
from toxicity_model.config import config

from toxicity_model.config import logging_config

_logger = logging_config.get_logger(__name__)


warnings.simplefilter(action='ignore', category=FutureWarning)
pd.options.mode.chained_assignment = None
tqdm.pandas()

from nltk.tokenize.treebank import TreebankWordTokenizer
tokenizer = TreebankWordTokenizer()


def train_val_test(df, train_prob=0.8):
    np.random.seed(1234)
    mask = np.random.rand(len(df)) < train_prob
    train_data = df[mask]
    temp_data = df[~mask]
    mask = np.random.rand(len(temp_data)) <= 0.5
    test_data = temp_data[mask]
    val_data = temp_data[~mask]
    
    return train_data, val_data, test_data


def handle_punctuation(x):
    x = x.translate(config.REMOVE_DICT)
    x = x.translate(config.ISOLATE_DICT)
    return x

def handle_contractions(x):
    x = tokenizer.tokenize(x)
    return x

def fix_quote(x):
    x = [x_[1:] if x_.startswith("'") else x_ for x_ in x]
    x = ' '.join(x)
    return x

def preprocess(x):
    x = handle_punctuation(x)
    x = handle_contractions(x)
    x = fix_quote(x)
    return x

def load_dataset(*, filename: str) -> pd.DataFrame:
    _data = pd.read_csv(f'{config.DATASETS_DIR}/{filename}', keep_default_na=False)
    return _data

def clean_data():
    '''

    :param input_file:
    :param output_file:
    :return:
    '''
    start = time.time()
    _logger.info(f'Reading data ... ')
    data = pd.read_csv(os.path.join(config.DATASETS_DIR, config.DATA_FILE))
    _logger.info('Done reading')
    _logger.info('Preprocessing text ...')
    
    data[config.TEXT] = data[config.TEXT].progress_apply(lambda x: preprocess(x))
    _logger.info('Preprocess target ...')
    data[config.TARGET] = data[config.TARGET].progress_apply(lambda x: 1 if x >= 0.5 else 0)
    new_data = data[[config.TEXT] + config.IDENTITY_COLUMNS + config.AUX_COLUMNS]
    
    new_data[config.IDENTITY_COLUMNS] = new_data[config.IDENTITY_COLUMNS].fillna(0)
    new_data[config.AUX_COLUMNS] = new_data[config.AUX_COLUMNS].fillna(0)
    new_data.dropna(axis=0, inplace=True)
    output_file = os.path.join(config.DATASETS_DIR, 'processed_train.csv')
    new_data.to_csv(output_file, index=False)
    
    train_file = os.path.join(config.DATASETS_DIR, config.TRAINING_DATA_FILE)
    validation_file = os.path.join(config.DATASETS_DIR, config.VALIDATION_DATA_FILE)
    test_file = os.path.join(config.DATASETS_DIR, config.TESTING_DATA_FILE)
    train_data, val_data, test_data = train_val_test(new_data)
    
    train_data.to_csv(train_file, index=False)
    val_data.to_csv(validation_file, index=False)
    test_data.to_csv(test_file, index=False)
    minutes, seconds = divmod(time.time() - start, 60)
    _logger.info(f'{int(minutes):0>2}:{int(seconds):05.2f}')
    
if __name__ == "__main__":
    clean_data()
    
    
    