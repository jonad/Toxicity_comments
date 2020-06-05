import pandas as pd
import pickle

from toxicity_model.config import config
from keras.preprocessing import text, sequence
import torch
from torch.utils import data
from toxicity_model.models import lstm, bilstm, attention
from toxicity_model import utils
import numpy as np
import os
from tqdm import tqdm
from toxicity_model.processing.clean_data import load_dataset
from toxicity_model import __version__ as _version
from toxicity_model.processing.validate_data import validate_input


def make_prediction(*, input_data):
    pass
    # function to evaluate my model
    # 1 - read the test.cs
    test_data = pd.read_json(input_data)
    test_data = validate_input(input_data=test_data)
    # 2 - vectorize it
    with open(os.path.join(config.TOKENIZER_DIR, config.TOKENIZER_FILE), 'rb') as handle:
        tokenizer = pickle.load(handle)
    
    with open(os.path.join(config.EMBEDDING_DIR, config.GLOVE_EMBEDDING_MODEL_FILE), 'rb') as handle:
        embeddings = pickle.load(handle)
    # read validation data
    x_test = tokenizer.texts_to_sequences(test_data[config.TEXT])
    x_test = sequence.pad_sequences(x_test, maxlen=config.MAX_LEN)
    # 3 - call the evaluate function
    x_test_torch = torch.tensor(x_test, dtype=torch.long).to('cpu')
    
    test_dataset = data.TensorDataset(x_test_torch)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)
    # LSTM models
    lstm_model = lstm.Lstm(embeddings)
    lstm_model.load_state_dict(torch.load(os.path.join(config.TRAINED_MODEL_DIR, f'state_dict_lstm{_version}.pt'), map_location="cpu"))
    lstm_model.eval()
    
    # BILSTM models
    bilstm_model = bilstm.BILSTM(embeddings)
    bilstm_model.load_state_dict(
        torch.load(os.path.join(config.TRAINED_MODEL_DIR, f'state_dict_bilstm{_version}.pt'), map_location="cpu"))
    bilstm_model.eval()
    
    # Attention
    attention_model = attention.AttentionNet(embeddings)
    attention_model.load_state_dict(
        torch.load(os.path.join(config.TRAINED_MODEL_DIR, f'state_dict_attention{_version}.pt'), map_location="cpu"))
    attention_model.eval()
    results = {}
    
    for model in [lstm_model, bilstm_model, attention_model]:
        
        test_preds = np.zeros((len(test_dataset), 7))
    
        for i, val_data in enumerate(tqdm(test_loader, disable=False)):
            x_batch = val_data[0]
            if model == attention_model:
                y_pred, _ = model(x_batch)
            else:
                y_pred = model(x_batch)
            y_pred = utils.sigmoid(y_pred.detach().cpu().numpy())
            test_preds[i * 1:i * 1 + y_pred.shape[0], :] = y_pred[:, :1]
        
        results[type(model).__name__.lower()] = test_preds.tolist()
    return {'predictions': results, 'version': _version}


if __name__ == '__main__':
    test_data = load_dataset(filename=config.TESTING_DATA_FILE)
    single_test_json = test_data[0:10].to_json(orient='records')
    test_preds = make_prediction(input_data=single_test_json)
    print(test_preds)



