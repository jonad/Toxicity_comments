import torch
import time
from tqdm import tqdm
import pandas as pd
from toxicity_model.config import config
import pickle
from keras.preprocessing import text, sequence
import numpy as np
from torch.utils import data
from toxicity_model import utils
import os
from toxicity_model.models import lstm, bilstm, attention
from torch import nn
from toxicity_model.config import logging_config
from toxicity_model import __version__ as _version

_logger = logging_config.get_logger('__name__')


def train_model(model, train, test, model_file, model_name, loss_fn, lr=0.001, batch_size=512, n_epochs=10):
    param_lrs = [{'params': param, 'lr': lr} for param in model.parameters()]
    optimizer = torch.optim.Adam(param_lrs, lr=lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 0.6 ** epoch)
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False)
    training_loss = []
    validation_loss = []
    
    best_loss = float("inf")
    for epoch in range(n_epochs):
        start_time = time.time()
        
        model.train()
        avg_loss = 0
        
        for data in tqdm(train_loader, disable=False):
            x_batch = data[:-1]
            y_batch = data[-1]
            if model_name != 'attention':
                y_pred = model(*x_batch)
            else:
                
                y_pred, _ = model(*x_batch, config.MAX_LEN)
            loss = loss_fn(y_pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            avg_loss += loss.item() / len(train_loader)
            
        training_loss.append(avg_loss)
        model.eval()
        _logger.info(f'... Validating {model_name} ... ')
        avg_val_loss = 0
        for val_data in tqdm(test_loader, disable=False):
            x_batch = val_data[:-1]
            y_batch = val_data[-1]
            if model_name != 'attention':
                y_pred = model(*x_batch)
            else:
                y_pred, _ = model(*x_batch, config.MAXLEN)
            
            val_loss = loss_fn(y_pred, y_batch)
            avg_val_loss += val_loss.item() / len(test_loader)
        
        elapsed_time = time.time() - start_time
        validation_loss.append(avg_val_loss)
        if avg_val_loss < best_loss:
            _logger.info('saving the best model so far')
            best_loss = avg_val_loss
            torch.save(model.state_dict(), model_file)
        _logger.info(
            f'Epoch {epoch + 1}/{n_epochs}\t training_loss={avg_loss:.4f} \t validation_loss={avg_val_loss: 4f} \t time={elapsed_time:.2f}s')
        scheduler.step()
    return training_loss, validation_loss


def train(model_name):
    utils.seed_everything()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # read the train data
    train = pd.read_csv(os.path.join(config.DATASETS_DIR, config.TRAINING_DATA_FILE), keep_default_na=False)
    validation = pd.read_csv(os.path.join(config.DATASETS_DIR, config.VALIDATION_DATA_FILE), keep_default_na=False)
    
    # compute weights
    loss_weights, training_weights = utils.compute_weights(train)
    _, validation_weights = utils.compute_weights(validation)
    
    # read the tokenizer and the embeddings
    with open(os.path.join(config.TOKENIZER_DIR, config.TOKENIZER_FILE), 'rb') as handle:
        tokenizer = pickle.load(handle)
    
    with open(os.path.join(config.EMBEDDING_DIR, config.GLOVE_EMBEDDING_MODEL_FILE), 'rb') as handle:
        embeddings = pickle.load(handle)
    # read validation data
    x_train = tokenizer.texts_to_sequences(train[config.TEXT])
    x_val = tokenizer.texts_to_sequences(validation[config.TEXT])
    x_train = sequence.pad_sequences(x_train, maxlen=config.MAX_LEN)
    x_val = sequence.pad_sequences(x_val, maxlen=config.MAX_LEN)
    
    # create y data
    y_train = np.vstack([(train[config.TARGET].values >= 0.5).astype(np.int), training_weights]).T
    y_val = np.vstack([(validation[config.TARGET].values >= 0.5).astype(np.int), validation_weights]).T
    
    y_aux_train = train[config.AUX_COLUMNS]
    y_aux_val = validation[config.AUX_COLUMNS]
    
    x_train_torch = torch.tensor(x_train, dtype=torch.long).to(device)
    x_val_torch = torch.tensor(x_val, dtype=torch.long).to(device)
    y_train_torch = torch.tensor(np.hstack([y_train, y_aux_train]), dtype=torch.float32).to(device)
    y_val_torch = torch.tensor(np.hstack([y_val, y_aux_val]), dtype=torch.float32).to(device)
    
    train_dataset = data.TensorDataset(x_train_torch, y_train_torch)
    val_dataset = data.TensorDataset(x_val_torch, y_val_torch)
    model_file = None
    model = None
    if model_name == 'lstm':
        model = lstm.Lstm(embeddings)
        model_file = config.TRAINED_MODEL_DIR/f'state_dict_lstm{_version}.pt'
    elif model_name == 'bilstm':
        model = bilstm.BILSTM(embeddings)
        model_file = os.path.join(config.TRAINED_MODEL_DIR, f'state_dict_bilstm{_version}.pt')
    elif model_name == 'attention':
        model = attention.AttentionNet(embeddings)
        model_file = os.path.join(config.TRAINED_MODEL_DIR, f'state_dict_attention.pt')
    model.to(device)
    
    def custom_loss(data, targets):
        ''' Define custom loss function for weighted BCE on 'target' column '''
        bce_loss_1 = nn.BCEWithLogitsLoss(weight=targets[:, 1:2])(data[:, :1], targets[:, :1])
        bce_loss_2 = nn.BCEWithLogitsLoss()(data[:, 1:], targets[:, 2:])
        return (bce_loss_1 * loss_weights) + bce_loss_2
    
    training_loss, validation_loss = train_model(model, train_dataset, val_dataset, model_file, model_name,
                                                 n_epochs=1,loss_fn=custom_loss)
    
    
    return training_loss, validation_loss
    
   

if __name__ == '__main__':
    models = ['attention']
    for model_name in models:
        _logger.info(f'.... Training .... {model_name}')
        
        training_loss, validation_loss = train(model_name)
        df = pd.DataFrame({'training_loss': training_loss, 'validation_loss':validation_loss})
        df.to_csv(model_name + '.csv')
    _logger.info('Training ended')
