import torch.nn as nn
import torch.nn.functional as F
from pygcn.layers import GraphConvolution
import numpy as np
import torch

def l2norm(x):
    norm = torch.pow(x, 2).sum(dim=1, keepdim=True).sqrt()
    x = torch.div(x, norm)
    return x


class GCN(nn.Module):
    # nfeat：300 nhid：16 nclass：300 dropout：0.5
    def __init__(self, nfeat, embed_size, nhid, nclass, dropout):
        super(GCN, self).__init__()

        # (300,16)    (16,300)
        self.fc = nn.Linear(nfeat, embed_size)
        self.init_weights()

        self.gc1 = GraphConvolution(embed_size, embed_size)
        self.gc2 = GraphConvolution(embed_size, embed_size)
        self.word_rnn = nn.GRU(embed_size, embed_size, 1, batch_first=True)

        self.dropout = dropout

    def init_weights(self):
        r = np.sqrt(6.) / np.sqrt(self.fc.in_features +
                                  self.fc.out_features)
        self.fc.weight.data.uniform_(-r, r)
        self.fc.bias.data.fill_(0)

    def forward(self, word, adj):
        fc_word_emd = self.fc(word)
        fc_word_emd = l2norm(fc_word_emd)

        x = F.relu(self.gc1(word, adj))  # (300,16)
        x = F.dropout(x, self.dropout, training=self.training)  # (300,16)
        x = self.gc2(x, adj)
        x = F.log_softmax(x, dim=1)
        x = l2norm(x)

        x = x.unsqueeze(0)
        rnn_word, hidden_state = self.word_rnn(x)
        rnn_word = rnn_word.squeeze(0)
        word_GCN = rnn_word

        return word_GCN