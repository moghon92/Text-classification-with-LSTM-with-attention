import torch

torch.manual_seed(10)
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class Attention(nn.Module):
    def __init__(self, vocab, num_classes):

        super(Attention, self).__init__()
        self.vocab = vocab
        self.num_classes = num_classes

        self.embed_len = 50  # embedding_dim default value for embedding layer
        self.hidden_dim = 75  # hidden_dim default value for rnn layer
        self.n_layers = 1  # num_layers default value for rnn

        self.embedding_layer = nn.Embedding(len(self.vocab), self.embed_len)
        self.lstm = nn.LSTM(self.embed_len, self.hidden_dim, self.n_layers, bidirectional=True, batch_first=True, dropout=0.2)
        self.lin = nn.Linear(2 * self.hidden_dim, self.num_classes)
        self.dropout = nn.Dropout(0.2)


    def forward(self, inputs, inputs_len):

        x = self.embedding_layer(inputs) # [N, L, embed_len] -> [1024, 136, 50]
        x = self.dropout(x)

        #print(x.shape)
        x = pack_padded_sequence(x, inputs_len, batch_first=True, enforce_sorted=False)

        output, (hidden, cell) = self.lstm(x)  # LSTM
        output, _ = pad_packed_sequence(output, batch_first=True)
        # output [N, L, 2*hidden_dim] -> [1024, 25, 150]
        # hidden [2, N, hidden_dim] -> [1024, 2, 75]
        #print(output.shape)
        #print(hidden.shape)

        hidden = hidden.view(-1, self.hidden_dim * 2,  1)  # [N, 2*hidden_dim, 1(n_layer)]  -> [1024, 150, 1]
        #print(hidden.shape)
        attn_weights = torch.bmm(output, hidden)  # [batch_size, L, 1] -> [1024,25,150]x[1024,150,1] = [1024,25,1]
        attn_weights = attn_weights.squeeze(2)  # [batch_size, L] -> [1024,25]
        attn_weights = F.softmax(attn_weights, 1)

        context = torch.bmm(output.transpose(1, 2), attn_weights.unsqueeze(2))  # [N, 2*hidden_dim, 1] -> [1024,150,25]x[1024,25,1] = [1024,150,1]
        context = context.squeeze(2)  # [N, 2*hidden_dim]

        x = self.lin(context)

        return x
