import torch
from torch.utils.data import Dataset


class ANNEDataset(Dataset):

    def __init__(self, data, data_freq, data_scl, labels, device):
        self.data = torch.tensor(data).to(device)
        self.data_freq = torch.tensor(data_freq).to(device)
        self.data_scl = torch.tensor(data_scl).to(device)
        self.labels = torch.tensor(labels).to(device)
        self.n_samples = self.data.shape[0]
        self.lens = [d.shape[1] for d in data]      # used for padding sequences (i.e. pack_padded_sequence) when using RNN

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        return self.data[index, :, :], self.data_freq[index, :, :], self.data_scl[index, :], self.labels[0, index], self.lens[index]
