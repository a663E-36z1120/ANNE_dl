import torch.nn as nn
import torch
from torch.nn import TransformerEncoderLayer, TransformerEncoder
import math


# TODO: Add more residual for recurrent layers
# TODO: Plot confusion matrix


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        # Create a long enough 'pe' matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x
    

class CRNN(nn.Module):
    def __init__(self, num_classes, in_channels, in_channels_f, in_channels_s):
        super(CRNN, self).__init__()

        n_slope = 0.01

        # Convolutional layers, time domain
        self.pool0 = nn.MaxPool1d(kernel_size=16)
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=in_channels, kernel_size=7, stride=1,
                               padding=3)
        torch.nn.init.xavier_uniform_(self.conv1.weight)
        self.bn1 = nn.BatchNorm1d(num_features=in_channels)
        self.relu1 = nn.LeakyReLU(n_slope)
        self.pool1 = nn.MaxPool1d(kernel_size=4)

        self.conv2 = nn.Conv1d(in_channels=in_channels, out_channels=in_channels, kernel_size=7, stride=1,
                               padding=3)
        torch.nn.init.xavier_uniform_(self.conv2.weight)
        self.bn2 = nn.BatchNorm1d(num_features=in_channels)
        self.relu2 = nn.LeakyReLU(n_slope)
        self.pool2 = nn.MaxPool1d(kernel_size=4)

        self.conv3 = nn.Conv1d(in_channels=in_channels + in_channels, out_channels=in_channels + in_channels,
                               stride=1, kernel_size=7)
        torch.nn.init.xavier_uniform_(self.conv3.weight)
        self.bn3 = nn.BatchNorm1d(num_features=in_channels + in_channels)
        self.relu3 = nn.LeakyReLU(n_slope)
        self.pool3 = nn.MaxPool1d(kernel_size=16)

        # Convolutional layers, freq domain
        self.pool0f = nn.MaxPool1d(kernel_size=4)

        self.conv1f = nn.Conv1d(in_channels=in_channels_f, out_channels=in_channels_f, kernel_size=7, stride=1,
                               padding=3)
        torch.nn.init.xavier_uniform_(self.conv1f.weight)
        self.bn1f = nn.BatchNorm1d(num_features=in_channels_f)
        self.relu1f = nn.LeakyReLU(n_slope)
        self.pool1f = nn.MaxPool1d(kernel_size=2)

        self.conv2f = nn.Conv1d(in_channels=in_channels_f, out_channels=in_channels_f, kernel_size=7, stride=1,
                               padding=3)
        torch.nn.init.xavier_uniform_(self.conv2f.weight)
        self.bn2f = nn.BatchNorm1d(num_features=in_channels_f)
        self.relu2f = nn.LeakyReLU(n_slope)
        self.pool2f = nn.MaxPool1d(kernel_size=2)

        self.conv3f = nn.Conv1d(in_channels=in_channels_f + in_channels_f, out_channels=in_channels_f + in_channels_f,
                               stride=1, kernel_size=7)
        torch.nn.init.xavier_uniform_(self.conv3f.weight)
        self.bn3f = nn.BatchNorm1d(num_features=in_channels_f + in_channels_f)
        self.relu3f = nn.LeakyReLU(n_slope)
        self.pool3f = nn.MaxPool1d(kernel_size=16)

        # Linear encoding layers

        self.flatten = nn.Flatten()
        self.dropout1 = torch.nn.Dropout(0.6)

        # self.relui = nn.LeakyReLU(n_slope)
        # self.fci = nn.Linear(in_features=192, out_features=128)
        # torch.nn.init.xavier_uniform_(self.fci.weight)
        #
        # self.reluii = nn.LeakyReLU(n_slope)
        # self.fcii = nn.Linear(in_features=128, out_features=32)
        # torch.nn.init.xavier_uniform_(self.fcii.weight)

        # Although the position encoder is only used for Transformer, the TorchScript compiler requires all referenced components
        # in the forward method to be defined
        self.pos_encoder = PositionalEncoding(d_model=74+in_channels_s, max_len=5000)

        # Transformer layers:
        encoder_layers = TransformerEncoderLayer(d_model=74+in_channels_s, nhead=4)
        self.transformer = TransformerEncoder(encoder_layers, num_layers=2)

        # Fully connected layer
        self.relu4 = nn.LeakyReLU(n_slope)
        self.dropout2 = torch.nn.Dropout(0.1)

        self.fc1 = nn.Linear(in_features=110+in_channels_s, out_features=(64+in_channels_s))
        torch.nn.init.xavier_uniform_(self.fc1.weight)

        self.fc1 = nn.Linear(in_features=154+in_channels_s, out_features=(64+in_channels_s))
        torch.nn.init.xavier_uniform_(self.fc1.weight)

        self.relu5 = nn.LeakyReLU(n_slope)
        self.dropout3 = torch.nn.Dropout(0.3)

        self.fc2 = nn.Linear(in_features=(64+in_channels_s), out_features=(64+in_channels_s))
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        self.relu6 = nn.LeakyReLU(n_slope)
        self.dropout4 = torch.nn.Dropout(0.1)


        self.fc3 = nn.Linear(in_features=64+in_channels_s, out_features=num_classes)
        torch.nn.init.xavier_uniform_(self.fc3.weight)


    def forward(self, x, x_freq, x_scl):
        # Convolutional layers, time domain
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x1 = self.relu1(x1)
        x1 = self.pool1(x1)

        x2 = self.conv2(x1)
        x2 = self.bn2(x2)
        x2 = self.relu2(x2)
        x2 = self.pool2(x2)

        # Residual connection
        x = self.pool0(x)
        x2 = torch.cat([x, x2], dim=1)
        # x2 = self.dropout1(x2)

        x3 = self.conv3(x2)
        x3 = self.bn3(x3)
        x3 = self.relu3(x3)
        x3 = self.pool3(x3)
        # x3 = self.dropout2(x3)

        # Reshape for recurrent layers
        # x3 = x3.permute(0, 2, 1)  # swap dimensions for RNN input
        x3 = self.flatten(x3)

        # Convolutional layers, freq domain
        x1f = self.conv1f(x_freq)
        x1f = self.bn1f(x1f)
        x1f = self.relu1f(x1f)
        x1f = self.pool1f(x1f)

        x2f = self.conv2f(x1f)
        x2f = self.bn2f(x2f)
        x2f = self.relu2f(x2f)
        x2f = self.pool2f(x2f)

        # Residual connection
        xf = self.pool0f(x_freq)
        x2f = torch.cat([xf, x2f], dim=1)
        # x2f = self.dropout1(x2f)

        x3f = self.conv3f(x2f)
        x3f = self.bn3f(x3f)
        x3f = self.relu3f(x3f)
        x3f = self.pool3f(x3f)
        # x3f = self.dropout2(x3f)

        # Reshape for recurrent layers
        # x3f = x3f.permute(0, 2, 1)  # swap dimensions for RNN input
        x3f = self.flatten(x3f)

        # Concat freq, scalar, and time domain
        x3 = torch.cat([x3[:, None, :], x3f[:, None, :]], dim=2)
        x3 = self.dropout1(x3)
        x3 = torch.cat([x3, x_scl[:, None, :]], dim=2)      # x3 has shape: (seq_len, batch=1, num_channels=80)

        # x3 = self.fci(x3)
        # x3 = self.relui(x3)
        # x3 = self.fcii(x3)
        # x3 = self.reluii(x3)

        x4 = self.pos_encoder(x3)                       # add positional encoding to x3
        x4_transformer = self.transformer(x4)                   # transformer output has shape: (seq_len, batch=1, num_channels)
        x4 = x4_transformer[:, -1, :]                   # removes batch dimension, x4 now has shape (seq_len, num_channels) for transformer

        # Fully connected layer
        x4 = torch.cat([x4, torch.squeeze(x3)], dim=1)      # torch.squeeze() removes dimension in x3 with length 1, so
                                                            # x4 now has shape (seq_len, 2 * num_channels) for transformer
                                                            # Note: seq_len is actually the "batch size" for our training objective  
        # x4 = self.dropout2(x4)
        

        x4 = self.fc1(x4)
        x4 = self.relu5(x4)
        # x4 = self.dropout3(x4)

        x4 = self.fc2(x4)
        x4 = self.dropout3(x4)
        x5 = self.fc3(x4)

        return x5
