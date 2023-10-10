import torch.nn as nn
import torch


# TODO: Add more residual for recurrent layers
# TODO: Plot confusion matrix

class CRNN(nn.Module):
    def __init__(self, num_classes, in_channels, in_channels_f, in_channels_s, model='rnn'):
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

        self.dropout1 = torch.nn.Dropout(0.4)


        # self.relui = nn.LeakyReLU(n_slope)
        # self.fci = nn.Linear(in_features=192, out_features=128)
        # torch.nn.init.xavier_uniform_(self.fci.weight)
        #
        # self.reluii = nn.LeakyReLU(n_slope)
        # self.fcii = nn.Linear(in_features=128, out_features=32)
        # torch.nn.init.xavier_uniform_(self.fcii.weight)

        # Recurrent layers:
        self.rnn_type = model.lower()
        if self.rnn_type == 'lstm':
            self.rnn = nn.LSTM(input_size=74, hidden_size=32, num_layers=2, dropout=0.5, bidirectional=True)
        elif self.rnn_type == 'gru':
            self.rnn = nn.GRU(input_size=74, hidden_size=32, num_layers=2, dropout=0.5, bidirectional=True)
        elif self.rnn_type == 'rnn':
            self.rnn = nn.RNN(input_size=74, hidden_size=32, num_layers=2, dropout=0.5, bidirectional=True)
        else:
            raise Exception("model is not one of 'lstm', 'gru', or 'rnn'.")

        # Fully connected layer
        self.fc1 = nn.Linear(in_features=106, out_features=64)
        self.relu5 = nn.LeakyReLU(n_slope)
        self.dropout3 = nn.Dropout(0.5)

        self.fc2 = nn.Linear(in_features=64, out_features=64)
        self.relu6 = nn.LeakyReLU(n_slope)
        self.dropout4 = nn.Dropout(0.5)

        self.fc3 = nn.Linear(in_features=64, out_features=num_classes)

        # Attention layer
        self.attention = nn.Linear(64, 1)
        self.softmax = nn.Softmax(dim=1)

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

        # Concat freq and time domain
        x3 = torch.cat([x3[:, None, :], x3f[:, None, :]], dim=2)
        x3 = self.dropout1(x3)

        # x3 = self.fci(x3)
        # x3 = self.relui(x3)
        # x3 = self.fcii(x3)
        # x3 = self.reluii(x3)

        # Recurrent layers
        x4, _ = self.rnn(x3)
        x4 = self.relu4(x4[:, -1, :])

        # Attention mechanism
        attention_weights = self.softmax(self.attention(x4))
        x4 = torch.sum(x4 * attention_weights, dim=1)

        # Fully connected layer
        x4 = torch.cat([x4, torch.squeeze(x3)], dim=1)

        x4 = self.fc1(x4)
        x4 = self.relu5(x4)
        x4 = self.dropout3(x4)

        x5 = self.fc2(x4)
        x5 = self.relu6(x5)
        x5 = self.dropout4(x5)

        x5 = self.fc3(x5)

        return x5
