import torch.nn as nn
import torch
from toxicity_model.config import config
from toxicity_model.models import dropout
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, feature_dim, step_dim, bias=True, **kwargs):
        super(Attention, self).__init__(**kwargs)
        
        self.supports_masking = True
        
        self.bias = bias
        self.feature_dim = feature_dim
        self.step_dim = step_dim
        
        weight = torch.zeros(feature_dim, 1)
        nn.init.xavier_uniform_(weight)
        self.weight = nn.Parameter(weight)
        
        if bias:
            self.b = nn.Parameter(torch.zeros(step_dim))
    
    def forward(self, x, mask=None):
        feature_dim = self.feature_dim
        step_dim = self.step_dim
        
        eij = torch.mm(
            x.contiguous().view(-1, feature_dim), self.weight
        ).view(-1, step_dim)
        
        eij = torch.tanh(eij)
        a = torch.exp(eij)
        
        if mask is not None:
            a = a * mask
        a = a / torch.sum(a, 1, keepdim=True) + 1e-10
        weighted_input = x * torch.unsqueeze(a, -1)
        return torch.sum(weighted_input, 1), torch.unsqueeze(a, -1)


class AttentionNet(nn.Module):
    def __init__(self, embedding_matrix):
        super(AttentionNet, self).__init__()
        embed_size = embedding_matrix.shape[1]
        self.embedding = nn.Embedding(config.MAX_FEATURES, embed_size)
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = False
        
        self.embedding_dropout = dropout.SpatialDropout(0.3)
        
        self.lstm1 = nn.LSTM(embed_size, config.LSTM_UNITS, bidirectional=True, batch_first=True)
        
        self.lstm2 = nn.LSTM(config.LSTM_UNITS * 2, config.LSTM_UNITS, bidirectional=True, batch_first=True)
        
        self.lstm_attention = Attention(config.LSTM_UNITS * 2, config.MAX_LEN)
        
        self.linear1 = nn.Linear(768, 768)
        self.linear2 = nn.Linear(768, 768)
        
        self.linear_out = nn.Linear(768, 1)
        self.linear_aux_out = nn.Linear(768, config.NUM_AUX_TARGETS)
    
    def forward(self, x):
        h_embedding = self.embedding(x)
        h_embedding = self.embedding_dropout(h_embedding)
        h_lstm1, _ = self.lstm1(h_embedding)
        h_lstm2, _ = self.lstm2(h_lstm1)
        
        h_lstm_atten, weights = self.lstm_attention(h_lstm2)
        
        # Attention layer
        
        # global average pooling
        avg_pool = torch.mean(h_lstm2, 1)
        
        # global max pooling
        max_pool, _ = torch.max(h_lstm2, 1)
        h_conc = torch.cat((h_lstm_atten, max_pool, avg_pool), 1)
        h_conc_linear1 = F.relu(self.linear1(h_conc))
        h_conc_linear2 = F.relu(self.linear2(h_conc))
        
        hidden = h_conc + h_conc_linear1 + h_conc_linear2
        result = self.linear_out(hidden)
        aux_result = self.linear_aux_out(hidden)
        out = torch.cat([result, aux_result], 1)
        return out, weights


if __name__ == '__main__':
    pass






