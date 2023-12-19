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

        self.conv4 = nn.Conv1d(in_channels=in_channels + in_channels, out_channels=in_channels + in_channels,
                               stride=1, kernel_size=7)
        torch.nn.init.xavier_uniform_(self.conv4.weight)
        self.bn4 = nn.BatchNorm1d(num_features=in_channels + in_channels)
        self.relu4 = nn.LeakyReLU(n_slope)
        self.pool3 = nn.MaxPool1d(kernel_size=16)

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
        if model == 'lstm':
            self.rnn = nn.LSTM(input_size=24, hidden_size=18, num_layers=2,
                               bidirectional=True)
        elif model == 'gru':
            self.rnn = nn.GRU(input_size=24, hidden_size=18, num_layers=2,
                              bidirectional=True)
        elif model == 'rnn':
            self.rnn = nn.RNN(input_size=24, hidden_size=18, num_layers=2,
                              bidirectional=True)
        else:
            raise Exception("model is not one of 'lstm', 'gru', or 'rnn'.")

        # Fully connected layer
        self.relu4 = nn.LeakyReLU(n_slope)
        self.dropout2 = torch.nn.Dropout(0.1)

        self.fc1 = nn.Linear(in_features=60, out_features=(32))
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        self.relu5 = nn.LeakyReLU(n_slope)
        self.dropout3 = torch.nn.Dropout(0.2)

        self.fc2 = nn.Linear(in_features=(32), out_features=(32))
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        self.relu6 = nn.LeakyReLU(n_slope)

        self.fc3 = nn.Linear(in_features=(32), out_features=(16))
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        self.relu7 = nn.LeakyReLU(n_slope)


        self.fc4 = nn.Linear(in_features=16, out_features=num_classes)
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

        x3 = self.conv4(x3)
        x3 = self.bn4(x3)
        x3 = self.relu4(x3)

        x3 = self.pool3(x3)
        # x3 = self.dropout2(x3)

        # Reshape for recurrent layers
        # x3 = x3.permute(0, 2, 1)  # swap dimensions for RNN input
        x3 = self.flatten(x3)



        # Residual connection
        # x2f = self.dropout1(x2f)



        # Reshape for recurrent layers
        # x3f = x3f.permute(0, 2, 1)  # swap dimensions for RNN input

        # Concat freq, scalar, and time domain
        x3 = self.dropout1(x3)


        # Recurrent layers
        x4, _ = self.rnn(x3)
        # x4 = self.relu4(x4[:, -1])

        # Fully connected layer
        # Res from the CNN

        x4 = torch.cat([x4, torch.squeeze(x3)], dim=1)
        x4 = self.dropout2(x4)

        x4 = self.fc1(x4)
        x4 = self.relu5(x4)
        # x4 = self.dropout3(x4)

        x4 = self.fc2(x4)
        x4 =self.relu6(x4)
        x4 = self.fc3(x4)

        x4 = self.dropout3(x4)
        x4 = self.relu7(x4)
        x5 = self.fc4(x4)

        return x5
