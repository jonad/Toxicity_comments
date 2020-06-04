from torch import nn
from toxicity_model.config import config
import torch
import torch.nn.functional as F
from toxicity_model.models.dropout import SpatialDropout


class Lstm(nn.Module):
    def __init__(self, embedding_matrix):
        super(Lstm, self).__init__()
        embed_size = embedding_matrix.shape[1]
        self.embedding = nn.Embedding(config.MAX_FEATURES, embed_size)
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = False
        self.embedding_dropout = SpatialDropout(0.3)
        self.lstm1 = nn.LSTM(embed_size, config.LSTM_UNITS, bidirectional=False, batch_first=True)
        self.lstm2 = nn.LSTM(config.LSTM_UNITS, config.LSTM_UNITS, bidirectional=False, batch_first=True)
        
        self.linear1 = nn.Linear(256, config.LSTM_UNITS)
        self.linear2 = nn.Linear(config.LSTM_UNITS * 2, config.LSTM_UNITS)
        
        self.linear_out = nn.Linear(config.LSTM_UNITS * 2, 1)
        self.linear_aux_out = nn.Linear(config.LSTM_UNITS * 2, config.NUM_AUX_TARGETS)
    
    def forward(self, x):
        h_embedding = self.embedding(x)
        h_embedding = self.embedding_dropout(h_embedding)
        h_lstm1, _ = self.lstm1(h_embedding)
        h_lstm2, _ = self.lstm2(h_lstm1)
        
        # global average pooling
        avg_pool = torch.mean(h_lstm2, 1)
        
        # global max pooling
        max_pool, _ = torch.max(h_lstm2, 1)
        h_conc = torch.cat((max_pool, avg_pool), 1)
        h_conc_linear1 = F.relu(self.linear1(h_conc))
        h_conc_linear2 = F.relu(self.linear2(h_conc))
        h_conc_linear = torch.cat((h_conc_linear1, h_conc_linear2), 1)
        
        hidden = h_conc + h_conc_linear
        result = self.linear_out(hidden)
        aux_result = self.linear_aux_out(hidden)
        out = torch.cat([result, aux_result], 1)
        return out


