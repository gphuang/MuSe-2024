import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import torch.nn.functional as F
from config import ACTIVATION_FUNCTIONS, device

class Attention(nn.Module):
    def __init__(self, in_dim_k, in_dim_q, out_dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = out_dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(in_dim_q, out_dim, bias=qkv_bias)
        self.kv = nn.Linear(in_dim_k, out_dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(out_dim, out_dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.qkmatrix = None

    def forward(self, x, x_q):
        B, Nk, Ck = x.shape
        B, Nq, Cq = x_q.shape
        q = self.q(x_q).reshape(B, Nq, 1, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        kv = self.kv(x).reshape(B, Nk, 2, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]   # make torchscript happy (cannot use tensor as tuple)
        q = q.squeeze(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
       
        attn = attn.softmax(dim=-1)
        
        self.qkmatrix = attn
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, Nq, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, self.qkmatrix

def conv1d_block(in_channels, out_channels, kernel_size=3, stride=1, padding='same'):
    return nn.Sequential(nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size,stride=stride, padding='valid'),nn.BatchNorm1d(out_channels),
                                   nn.ReLU(inplace=True), nn.MaxPool1d(2,1))

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

def last_item_from_packed(packed, lengths):
    """
    #https://discuss.pytorch.org/t/get-each-sequences-last-item-from-packed-sequence/41118/7
    """
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

class Model(nn.Module):
    def __init__(self, params):
        super(Model, self).__init__()
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

class CnnModel(nn.Module):
    """
    source: https://github.com/gphuang/multimodal-emotion-recognition-ravdess/tree/main
    """

    def __init__(self, params):
        super(CnnModel, self).__init__()

        self.params = params
        input_channels=params.d_in # feat_size
        num_classes=params.n_targets

        self.conv1d_0 = conv1d_block(input_channels, 64)
        self.conv1d_1 = conv1d_block(64, 128)
        self.conv1d_2 = conv1d_block(128, 256)
        self.conv1d_3 = conv1d_block(256, 128)
        self.out = OutLayer(128, params.d_fc_out, params.n_targets, dropout=params.linear_dropout)
        self.final_activation = ACTIVATION_FUNCTIONS[params.task]()
            
    def forward(self, x, x_len):
        """Input (av feats): (batch_size, seq_len, input_channels)"""

        # Conv 
        x = x.transpose(1, 2) # (batch_size, input_channels, seq_len)  
        x = self.conv1d_0(x)
        x = self.conv1d_1(x) # torch.Size([batch_size, conv_dim, conv_seq_len'])
        x = self.conv1d_2(x)
        x = self.conv1d_3(x) # (batch_size, conv_dim, conv_seq_len*)  
        x = x.mean([-1]) # torch.Size([batch_size, conv_dim]) pooling accross temporal dimension
        x = self.out(x) # torch.Size([batch_size, 1])
        activation = self.final_activation(x) #   torch.Size([batch_size, 1])
        return activation, x
    
class AttnCnnModel(nn.Module):
    """
    source: https://github.com/gphuang/multimodal-emotion-recognition-ravdess/tree/main/models/multimodalcnn.py
    """

    def __init__(self, params):
        super(AttnCnnModel, self).__init__()

        self.params = params
        input_channels=params.d_in 
        num_classes=params.n_targets

        self.attention = Attention(in_dim_k=input_channels, in_dim_q=input_channels, out_dim=256)
        self.conv1d = conv1d_block(256, 128)
        self.out = OutLayer(128, params.d_fc_out, params.n_targets, dropout=params.linear_dropout)
        self.final_activation = ACTIVATION_FUNCTIONS[params.task]()
            
    def forward(self, x, x_len):
        """Input (av feats): (batch_size, seq_len, input_channels)"""

        x = x.transpose(1, 2) # (batch_size, input_channels, seq_len) 

        # Attn
        proj_x = x.permute(0,2,1) # (batch_size, seq_len, input_channels)
        x, _ = self.attention(proj_x, proj_x) # (batch_size, input_channels, attn_out_dim)

        # Conv
        x = x.permute(0,2,1)
        x = self.conv1d(x) # (batch_size, conv_dim, conv_seq_len*)

        # Classifier
        x = x.mean([-1]) # torch.Size([batch_size, conv_dim]) pooling accross temporal dimension
        x = self.out(x) # torch.Size([batch_size, 1])

        activation = self.final_activation(x) #   torch.Size([batch_size, 1])
        return activation, x

class CrnnAttnModel(nn.Module):
    """
    source: https://github.com/gphuang/multimodal-emotion-recognition-ravdess/tree/main/models/multimodalcnn.py
    """

    def __init__(self, params):
        super(CrnnAttnModel, self).__init__()

        self.params = params
        input_channels=params.d_in 

        self.conv1d_0 = conv1d_block(input_channels, 64)
        self.conv1d_1 = conv1d_block(64, 128)
        self.conv1d_2 = conv1d_block(128, 256)
        self.conv1d_3 = conv1d_block(256, 128)
        
        self.attention = Attention(in_dim_k=128, in_dim_q=128, out_dim=128)

        self.inp = nn.Linear(128, params.model_dim, bias=False)

        self.gru = nn.GRU(input_size=params.model_dim, 
                          hidden_size=params.model_dim, 
                          bidirectional=params.rnn_bi, 
                          num_layers=params.rnn_n_layers, 
                          dropout=params.rnn_dropout)

        d_rnn_out = params.model_dim * 2 if params.rnn_bi and params.rnn_n_layers > 0 else params.model_dim
        self.out = OutLayer(d_rnn_out, params.d_fc_out, params.n_targets, dropout=params.linear_dropout)
        self.final_activation = ACTIVATION_FUNCTIONS[params.task]()

    def forward(self, x, x_len):
        """Input (av feats): (batch_size, seq_len, input_channels)"""

        # Conv 
        x = x.transpose(1, 2) # (batch_size, input_channels, seq_len)  
        x = self.conv1d_0(x)
        x = self.conv1d_1(x) # (batch_size, conv_dim, conv_seq_len')

        # Attn
        proj_x = x.permute(0,2,1) # (batch_size, conv_seq_len', conv_dim)
        _, h_matrix = self.attention(proj_x, proj_x) # (batch_size, attn_heads, attn_seq_len, attn_seq_len)
        if h_matrix.size(1) > 1: #if more than 1 head, take average
            h_matrix = torch.mean(h_matrix, axis=1).unsqueeze(1)
        h_matrix = h_matrix.sum([-2]) # (batch_size, 1, attn_seq_len)
        x = h_matrix*x # (batch_size, conv_dim, attn_seq_len)
    
        # Conv
        x = self.conv1d_2(x)
        x = self.conv1d_3(x) # (batch_size, conv_dim, conv_seq_len*)

        # GRU
        x = x.transpose(1, 2) # (batch_size, conv_seq_len*, conv_dim)  
        x = self.inp(x) # (batch_size, conv_seq_len*, rnn_dim)  
        x, _hidden_states = self.gru(x) # (batch_size, conv_seq_len*, rnn_dim)  
        x = x.transpose(1, 2)
        x = x.mean([-1]) # pooling accross temporal dimension  
        x = self.out(x) # torch.Size([batch_size, 1])  
        activation = self.final_activation(x) # torch.Size([batch_size, 1]) 
        return activation, x

class CnnAttnModel(nn.Module):
    """
    source: https://github.com/gphuang/multimodal-emotion-recognition-ravdess/tree/main/models/multimodalcnn.py
    """

    def __init__(self, params):
        super(CnnAttnModel, self).__init__()

        self.params = params
        input_channels=params.d_in 
        num_classes=params.n_targets

        self.conv1d_0 = conv1d_block(input_channels, 64)
        self.conv1d_1 = conv1d_block(64, 128)
        self.conv1d_2 = conv1d_block(128, 256)
        self.conv1d_3 = conv1d_block(256, 128)
        
        self.attention = Attention(in_dim_k=128, in_dim_q=128, out_dim=128)
        self.out = OutLayer(128, params.d_fc_out, params.n_targets, dropout=params.linear_dropout)        
        self.final_activation = ACTIVATION_FUNCTIONS[params.task]()
            
    def forward(self, x, x_len):
        """Input (av feats): (batch_size, seq_len, input_channels)"""

        # Conv 
        x = x.transpose(1, 2) # (batch_size, input_channels, seq_len)  
        x = self.conv1d_0(x)
        x = self.conv1d_1(x) # (batch_size, conv_dim, conv_seq_len')

        # Attn
        proj_x = x.permute(0,2,1) # (batch_size, conv_seq_len', conv_dim)
        _, h_matrix = self.attention(proj_x, proj_x) # (batch_size, attn_heads, attn_seq_len, attn_seq_len)
        if h_matrix.size(1) > 1: #if more than 1 head, take average
            h_matrix = torch.mean(h_matrix, axis=1).unsqueeze(1)
        h_matrix = h_matrix.sum([-2]) # (batch_size, 1, attn_seq_len)
        x = h_matrix*x # (batch_size, conv_dim, attn_seq_len)
    
        # Conv
        x = self.conv1d_2(x)
        x = self.conv1d_3(x) # (batch_size, conv_dim, conv_seq_len*)

        # Classifier
        x = x.mean([-1]) # torch.Size([batch_size, conv_dim]) pooling accross temporal dimension
        x = self.out(x) # torch.Size([batch_size, 1])
        activation = self.final_activation(x) #   torch.Size([batch_size, 1])
        return activation, x

class CrnnModel(nn.Module):
    """
    source: https://github.com/gphuang/multimodal-emotion-recognition-ravdess/tree/main
    """

    def __init__(self, params):
        super(CrnnModel, self).__init__()

        self.params = params
        input_channels=params.d_in # feat_size

        self.conv1d_0 = conv1d_block(input_channels, 64)
        self.conv1d_1 = conv1d_block(64, 128)
        self.conv1d_2 = conv1d_block(128, 256)
        self.conv1d_3 = conv1d_block(256, 128)

        self.inp = nn.Linear(128, params.model_dim, bias=False)

        self.gru = nn.GRU(input_size=params.model_dim, 
                          hidden_size=params.model_dim, 
                          bidirectional=params.rnn_bi, 
                          num_layers=params.rnn_n_layers, 
                          dropout=params.rnn_dropout)

        d_rnn_out = params.model_dim * 2 if params.rnn_bi and params.rnn_n_layers > 0 else params.model_dim
        self.out = OutLayer(d_rnn_out, params.d_fc_out, params.n_targets, dropout=params.linear_dropout)
        self.final_activation = ACTIVATION_FUNCTIONS[params.task]()
            
    def forward(self, x, x_len):
        """Input (av feats): (batch_size, seq_len, input_channels)"""

        # Conv 
        x = x.transpose(1, 2) # (batch_size, input_channels, seq_len)   ([59, 88, 80])
        x = self.conv1d_0(x)
        x = self.conv1d_1(x) # torch.Size([batch_size, conv_dim, conv_seq_len']) ([59, 128, 74])
        x = self.conv1d_2(x)
        x = self.conv1d_3(x) # (batch_size, conv_dim, conv_seq_len*)  ([59, 128, 68])

        # GRU
        x = x.transpose(1, 2) # (batch_size, conv_seq_len*, conv_dim) ([59, 68, 128])
        x = self.inp(x) # (batch_size, conv_seq_len*, rnn_dim) ([59, 68, 256])
        x, _hidden_states = self.gru(x) # (batch_size, conv_seq_len*, rnn_dim) ([59, 68, 256])
        x = x.transpose(1, 2)
        x = x.mean([-1]) # pooling accross temporal dimension ([59, 256])
        x = self.out(x) # torch.Size([batch_size, 1]) ([59, 1])
        activation = self.final_activation(x) # torch.Size([batch_size, 1]) ([59, 1])
        return activation, x
