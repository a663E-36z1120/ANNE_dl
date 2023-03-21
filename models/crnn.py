import torch.nn as nn


class CRNN(nn.Module):
    def __init__(self, num_classes, in_channels, model='rnn'):
        super(CRNN, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(num_features=64)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2)

        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(num_features=128)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2)

        self.conv3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm1d(num_features=256)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool1d(kernel_size=2)

        # Recurrent layers:
        if model == 'lstm':
            self.rnn = nn.LSTM(input_size=256, hidden_size=128, num_layers=2, bidirectional=True, batch_first=True)
        elif model == 'gru':
            self.rnn = nn.GRU(input_size=256, hidden_size=128, num_layers=2, bidirectional=True, batch_first=True)
        elif model == 'rnn':
            self.rnn = nn.RNN(input_size=256, hidden_size=128, num_layers=2, bidirectional=True, batch_first=True)
        else:
            raise Exception("model is not one of 'lstm', 'gru', or 'rnn'.")

        # Fully connected layer
        self.fc = nn.Linear(in_features=256, out_features=num_classes)

    def forward(self, x, lengths):
        # Convolutional layers
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.pool3(x)

        # Reshape for recurrent layers
        x = x.permute(0, 2, 1)  # swap dimensions for RNN input

        # Note: we are not using pack_padded_sequence for now as all sequences are 750 in length (and there is an error
        # when pad_packed_sequence() it is called...)

        # x = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)

        # Recurrent layers
        x, _ = self.rnn(x)

        # x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)

        # Fully connected layer
        x = self.fc(x[:, -1, :])

        return x
