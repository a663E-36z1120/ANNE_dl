import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self, num_classes):
        super(LSTM, self).__init__()

        # Recurrent layers
        self.lstm = nn.LSTM(input_size=8, hidden_size=128, num_layers=2, bidirectional=True, batch_first=True)

        # Fully connected layer
        self.fc = nn.Linear(in_features=256, out_features=num_classes)

    def forward(self, x, lengths):
        # Reshape for recurrent layers
        x = x.permute(0, 2, 1)  # swap dimensions for LSTM input
        # x = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)

        # Recurrent layers
        x, _ = self.lstm(x)
        # x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)

        # Fully connected layer
        x = self.fc(x[:, -1, :])

        return x
