import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import torch.nn.functional as F
from config import ACTIVATION_FUNCTIONS, device

def conv1d_block_audio(in_channels, out_channels, kernel_size=3, stride=1, padding='same'):
    return nn.Sequential(nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size,stride=stride, padding='valid'),nn.BatchNorm1d(out_channels),
                                   nn.ReLU(inplace=True), nn.MaxPool1d(2,1))

class Model(nn.Module):
    """
    source: https://github.com/gphuang/multimodal-emotion-recognition-ravdess/tree/main
    """

    def __init__(self, params):
        super(Model, self).__init__()

        self.params = params
        input_channels=params.d_in
        num_classes=params.n_targets

        self.conv1d_0 = conv1d_block_audio(input_channels, 64)
        self.conv1d_1 = conv1d_block_audio(64, 128)
        self.conv1d_2 = conv1d_block_audio(128, 256)
        self.conv1d_3 = conv1d_block_audio(256, 128)

        self.classifier_1 = nn.Sequential(nn.Linear(128, num_classes),)
        self.final_activation = ACTIVATION_FUNCTIONS[params.task]()
            
    def forward(self, x, x_len):
        """Input (av feats): (batch_size, seq_len, feat_size)"""

        """print(x.shape)
        print(self.params)
        import sys
        sys.exit(0)"""

        # Conv 
        x = x.transpose(1, 2)     # (batch_size x feat_size x seq_len) torch.Size([59, 512, 81])
        x = self.conv1d_0(x)
        x = self.conv1d_1(x) # torch.Size([59, 128, 75])
        x = self.conv1d_2(x)
        x = self.conv1d_3(x) # torch.Size([59, 128, 69])
        x = x.mean([-1]) # torch.Size([59, 128]) pooling accross temporal dimension
        x = self.classifier_1(x) # torch.Size([59, 1])
        activation = self.final_activation(x) #   torch.Size([59, 1])
        return activation, x

class RNN(nn.Module):
    def __init__(self, d_in, d_out, n_layers=1, bi=True, dropout=0.2, n_to_1=False):
        super(RNN, self).__init__()
        self.rnn = nn.GRU(input_size=d_in, hidden_size=d_out, bidirectional=bi, num_layers=n_layers, dropout=dropout)
        self.n_layers = n_layers
        self.d_out = d_out
        self.n_directions = 2 if bi else 1
        self.n_to_1 = n_to_1

    def forward(self, x, x_len):
        x_packed = pack_padded_sequence(x, x_len.cpu(), batch_first=True, enforce_sorted=False)
        rnn_enc = self.rnn(x_packed)

        if self.n_to_1:
            # hiddenstates, h_n, only last layer
            return last_item_from_packed(rnn_enc[0], x_len)

        else:
            x_out = rnn_enc[0]
            x_out = pad_packed_sequence(x_out, total_length=x.size(1), batch_first=True)[0]

        return x_out

#https://discuss.pytorch.org/t/get-each-sequences-last-item-from-packed-sequence/41118/7
def last_item_from_packed(packed, lengths):
    sum_batch_sizes = torch.cat((
        torch.zeros(2, dtype=torch.int64),
        torch.cumsum(packed.batch_sizes, 0)
    )).to(device)
    sorted_lengths = lengths[packed.sorted_indices].to(device)
    last_seq_idxs = sum_batch_sizes[sorted_lengths] + torch.arange(lengths.size(0)).to(device)
    last_seq_items = packed.data[last_seq_idxs]
    last_seq_items = last_seq_items[packed.unsorted_indices]
    return last_seq_items

class OutLayer(nn.Module):
    def __init__(self, d_in, d_hidden, d_out, dropout=.0, bias=.0):
        super(OutLayer, self).__init__()
        self.fc_1 = nn.Sequential(nn.Linear(d_in, d_hidden), nn.ReLU(True), nn.Dropout(dropout))
        self.fc_2 = nn.Linear(d_hidden, d_out)
        nn.init.constant_(self.fc_2.bias.data, bias)

    def forward(self, x):
        y = self.fc_2(self.fc_1(x))
        return y

class RNNModel(nn.Module):
    def __init__(self, params):
        super(RNNModel, self).__init__()
        self.params = params

        self.inp = nn.Linear(params.d_in, params.model_dim, bias=False)

        self.encoder = RNN(params.model_dim, params.model_dim, n_layers=params.rnn_n_layers, bi=params.rnn_bi,
                           dropout=params.rnn_dropout, n_to_1=params.n_to_1)

        d_rnn_out = params.model_dim * 2 if params.rnn_bi and params.rnn_n_layers > 0 else params.model_dim
        self.out = OutLayer(d_rnn_out, params.d_fc_out, params.n_targets, dropout=params.linear_dropout)
        self.final_activation = ACTIVATION_FUNCTIONS[params.task]()

    def forward(self, x, x_len):
        """
        x: torch.Size([bs, t_step, feat_dim]), e.g. feat_dim=20 faus
        x_len: torch.Size([bs])
        """
        x = self.inp(x) # torch.Size([59, 81, 256])
        x = self.encoder(x, x_len) # torch.Size([59, 256])
        x = self.out(x) # torch.Size([59, 1])
        activation = self.final_activation(x) # torch.Size([59, 1])
        return activation, x

    def set_n_to_1(self, n_to_1):
        self.encoder.n_to_1 = n_to_1


