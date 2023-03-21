# @author: Andrew H. Zhang

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from preprocessing.main import integrate_data


class ANNEDataset(Dataset):

    def __init__(self, data, labels, device):
        self.data = torch.tensor(data).to(device)
        self.labels = torch.tensor(labels).to(device)
        self.n_samples = self.data.shape[0]

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        return self.data[index, :, :], self.labels[0, index]


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(  # input: 8 x 750 (num channels x sequence length)
            nn.Conv1d(8, 64, kernel_size=3, padding='same'),  # 64 x 750
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.MaxPool1d(5))  # 64 x 150
        self.layer2 = nn.Flatten()  # 9600
        self.layer3 = nn.Sequential(
            nn.Linear(9600, 100),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            nn.Linear(100, 3))

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        return out


def train_model(model, optimizer, train_loader, test_loader, epochs=100, print_every=10):

    # optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    train_accs = []
    test_accs = []
    train_losses = []
    test_losses = []

    # Using GPUs in PyTorch is pretty straightforward
    if torch.cuda.is_available():
        print("Using cuda")
        use_cuda = True
        device = torch.device("cuda")
    else:
        device = "cpu"

    # Move the model to GPU, if available
    model.to(device)
    model.train()

    for epoch in range(epochs):
        xentropy_loss_avg = 0.
        correct = 0.
        total = 0.

        for i, (inputs, labels) in enumerate(train_loader):
            model.zero_grad()
            inputs = inputs.to(device)
            labels = labels.to(device)
            # inputs = inputs.view(inputs.size(0), -1)  # Flatten input from [batch_size, 1, 28, 28] to [batch_size, 784]
            pred = model(inputs)
            xentropy_loss = criterion(pred, labels)
            xentropy_loss.backward()

            optimizer.step()

            xentropy_loss_avg += xentropy_loss.item()

            # Calculate running average of accuracy
            pred = torch.max(pred.data, 1)[1]
            total += labels.size(0)
            correct += (pred == labels.data).sum().item()
        accuracy = correct / total

        test_acc, test_loss = evaluate(model, test_loader, criterion, device)
        if epoch % print_every == 0:
            print("Epoch {}, Train acc: {:.2f}%, Test acc: {:.2f}%".format(epoch, accuracy * 100, test_acc * 100))


        train_accs.append(accuracy)
        test_accs.append(test_acc)
        train_losses.append(xentropy_loss_avg / i)
        test_losses.append(test_loss)

    return train_accs, test_accs, train_losses, test_losses


def evaluate(model, loader, criterion, device):
    model.eval()  # Change model to 'eval' mode (BN uses moving mean/var).
    correct = 0.
    total = 0.
    val_loss = 0.
    for i, (inputs, labels) in enumerate(loader):
        with torch.no_grad():
            inputs = inputs.to(device)
            labels = labels.to(device)
            # inputs = inputs.view(inputs.size(0), -1)
            pred = model(inputs)
            xentropy_loss = criterion(pred, labels)
            val_loss += xentropy_loss.item()

        pred = torch.max(pred.data, 1)[1]
        total += labels.size(0)
        correct += (pred == labels).sum().item()

    val_acc = correct / total
    val_loss = val_loss / i
    model.train()
    return val_acc, val_loss


def plot_grad_histograms(grad_list, epoch, init=False):
    fig, ax = plt.subplots(nrows=1, ncols=len(grad_list), figsize=(5 * len(grad_list), 5))
    for i, grad in enumerate(grad_list):
        plt.subplot(1, len(grad_list), i + 1)
        plt.hist(grad)
        if init:
            plt.title("Grads for Weights (Layer {}-{}) Init".format(i, i + 1))
        else:
            plt.title("Grads for Weights (Layer {}-{}) Epoch {}".format(i, i + 1, epoch))
    plt.show()


def plot_act_histograms(act_list, epoch, init=False):
    fig, ax = plt.subplots(nrows=1, ncols=len(act_list), figsize=(5 * len(act_list), 5))
    for i, act in enumerate(act_list):
        plt.subplot(1, len(act_list), i + 1)
        plt.hist(act)
        if init:
            plt.title("Activations for Layer {} Init".format(i + 1))
        else:
            plt.title("Activations for Layer {} Epoch {}".format(i + 1, epoch))
    plt.show()



if __name__ == "__main__":
    # Check gpu availability
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Load data:
    X1, t1 = integrate_data(132, -7.64)
    X2, t2 = integrate_data(135, -6.84)
    # Build model
    cnn_model = SimpleCNN()

    # Initialize dataloaders
    train_dataset = ANNEDataset(X1, t1, device)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
    val_dataset = ANNEDataset(X2, t2, device)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=64, shuffle=True)

    # Train model:
    # Define optimizer: Choose SGD for now
    learning_rate = 0.01
    optimizer = torch.optim.SGD(cnn_model.parameters(), lr=learning_rate, momentum=0.9)

    # Run the training loop
    train_accs, test_accs, train_losses, test_losses = train_model(cnn_model, optimizer, train_dataloader, val_dataloader, epochs=100,
                                                                   print_every=2)

    pass
