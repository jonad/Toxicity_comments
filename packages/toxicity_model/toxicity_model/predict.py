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
from toxicity_model.processing.clean_data import preprocess


def make_prediction(*, input_data):
    pass
    # function to evaluate my model
    # 1 - read the test.cs
    test_data = pd.read_json(input_data)
    test_data[config.TEXT] = test_data[config.TEXT].apply(lambda x: preprocess(x))
    test_data = validate_input(input_data=test_data)
    # 2 - vectorize it
    with open(os.path.join(config.TOKENIZER_DIR, config.TOKENIZER_FILE), 'rb') as handle:
        tokenizer = pickle.load(handle)
    
    with open(os.path.join(config.EMBEDDING_DIR, config.GLOVE_EMBEDDING_MODEL_FILE), 'rb') as handle:
        embeddings = pickle.load(handle)
    # read validation data
    x_test = tokenizer.texts_to_sequences(test_data[config.TEXT])
    #x_test = sequence.pad_sequences(x_test, maxlen=config.MAX_LEN)
    # 3 - call the evaluate function
    lstm_predictions = []
    bilstm_predictions = []
    attention_predictions = []
    weight_predictions = []
    for i, elt in enumerate(tqdm(x_test)):
        x_test_torch = torch.tensor(elt, dtype=torch.long).to('cpu')
        len_elt = len(elt)
    
    
    #test_dataset = data.TensorDataset(x_test_torch)
    #test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)
    #LSTM models
        lstm_model = lstm.Lstm(embeddings)
        lstm_model.load_state_dict(torch.load(os.path.join(config.TRAINED_MODEL_DIR, f'state_dict_lstm{_version}.pt'), map_location="cpu"))
        lstm_model.eval()

    #BILSTM models
        bilstm_model = bilstm.BILSTM(embeddings)
        bilstm_model.load_state_dict(
            torch.load(os.path.join(config.TRAINED_MODEL_DIR, f'state_dict_bilstm{_version}.pt'), map_location="cpu"))
        bilstm_model.eval()

    # Attention
        attention_model = attention.AttentionNet(embeddings)
        attention_model.load_state_dict(
            torch.load(os.path.join(config.TRAINED_MODEL_DIR, f'state_dict_attention.pt'), map_location="cpu"))
        attention_model.eval()
        results = {}
    
        for model in [lstm_model, bilstm_model, attention_model]:
        
            #test_preds = np.zeros((1, 7))
    
        #for i, val_data in enumerate(tqdm(test_loader, disable=False)):
            #x_batch = val_data[0]
            if model == attention_model:
                #print(len(*x_batch))
                y_pred, weight = model(x_test_torch.view(-1, len_elt), len_elt)
                pred = utils.sigmoid(y_pred.detach().cpu().numpy())
                weight = weight.detach().cpu().numpy().flatten().tolist()
                attention_predictions.append(pred.tolist()[0])
                weight_predictions.append(weight)
                #print(weight.detach().cpu().numpy().flatten().argsort()[-1:][::-1])
                #print(np.sort(weight.detach().cpu().numpy().flatten())[:20])
                
                #print(np.sum(weight.detach().cpu().numpy().flatten()))
            elif model == bilstm_model:
                y_pred = model(x_test_torch.view(-1, len_elt))
                pred = utils.sigmoid(y_pred.detach().cpu().numpy())
                bilstm_predictions.append(pred.tolist()[0])
            else:
                y_pred = model(x_test_torch.view(-1, len_elt))
                pred = utils.sigmoid(y_pred.detach().cpu().numpy())
                lstm_predictions.append(pred.tolist()[0])
                # print('weights')
                # print(weight)
                # print('Predictions')
                # print(pred)
            #test_preds[i * 1:i * 1 + pred.shape[0], :] = pred
        
        #results[type(model).__name__.lower()] = test_preds.tolist()
    


    results = {'lstm': lstm_predictions, 'bilstm':bilstm_predictions,
               'attention':attention_predictions, 'weights': weight_predictions,
               'comments': test_data[config.TEXT].tolist(),
               #'target': test_data[config.TARGET].tolist(),
               #'severe_toxicity': test_data['severe_toxicity'].tolist(),
               #'obscene': test_data['obscene'].tolist(),
               #'identity_attack': test_data['identity_attack'].tolist(),
               #'insult': test_data['insult'].tolist(),
               #'threat': test_data['threat'].tolist()
               }
    df = pd.DataFrame(results)
    df.to_csv('result_trump.csv')
    #return results
    


if __name__ == '__main__':
    test_data = load_dataset(filename=config.TESTING_DATA_FILE)
    single_test_json = test_data.to_json(orient='records')
    make_prediction(input_data=single_test_json)
    #print(test_preds)



