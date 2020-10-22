import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader,TensorDataset



class LAnn_prediction(nn.Module):
    """
    Params:
        def __init__():
            diction_size: number of all chars
            embedding_dim: dim. of one char
            hidden_size: Feature dim. of output of GRU
            target_size: dim. of binary classification

        def forward():
            data: data[0] is phrase recognized by ocr, dim. of it is changeable;
                  data[1] is all phrase recognized by ocr, dim. of it is fixed.

        def attentionDot():
            gru_out: final vector of GRU for one phrase
            gru_out_env: final vector of GRU for all phrases
    Return:
        def forward():
            score: score of classification
        
        def attentionDot():
            Hl: output of scaled-dot attention
    """
    def __init__(self, diction_size, embedding_dim, hidden_size, target_size):
        super(LAnn_prediction, self).__init__()
        self.input = nn.Embedding(diction_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_size, 1, bidirectional=True, batch_first=True)
        self.h2target = nn.Linear(2*hidden_size, target_size)
        self.activation = nn.SELU()
        self.scaling_factor = torch.sqrt(torch.tensor(hidden_size).float())
        
    def forward(self, data):
        t = data[0]
        env = data[1]

        embedded = self.input(t)
        embedded_env = self.input(env)
        _, hidden = self.gru(embedded)
        _, hidden_env = self.gru(embedded_env)
        gru_out = torch.cat([hidden[0],hidden[1]], dim=1)
        gru_out_env = torch.cat([hidden_env[0],hidden_env[1]], dim=1)
        
        Hl = self.attentionDot(gru_out, gru_out_env)
        score = self.activation(self.h2target(Hl))
        return score

    def attentionDot(self, gru_out, gru_out_env):
        # scaled-dot attention, (q, k, v) are (query, key, value)
        q = gru_out_env
        k = gru_out
        v = gru_out
        k_trans = k.transpose(1, 0)
        Hl = torch.mm(torch.softmax(torch.mm(q, k_trans) / self.scaling_factor, dim=-1), v)
        return Hl
