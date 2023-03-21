import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib
import matplotlib.pyplot as plt
from preprocessing.main import integrate_data


class ANNEDataset(Dataset):

    def __init__(self, data, labels, transform=None):
        self.data = data
        self.transform = transform
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        signal = self.data[idx].transpose().astype(np.float32)
        label = self.labels[idx].item()
        # Convert label to be one-hot encoded:
        # '?' = [1, 0, 0], 'W' = [1, 0, 0], 'N1'/'N2'/'N3' = [0, 1, 0], 'R' = [0, 0, 1]
        if label == '?' or label == 'W':
            label = np.array([1, 0, 0]).astype(np.float32)
        elif label == 'N1' or label == 'N2' or label == 'N3':
            label = np.array([0, 1, 0]).astype(np.float32)
        elif label == 'R':
            label = np.array([0, 0, 1]).astype(np.float32)
        else:
            print(f'Exception: Encounter label={label}.')
        if self.transform:
            signal = self.transform(signal)
        return signal, label


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(    # input: 8 x 750 (num channels x sequence length)
            nn.Conv1d(8, 64, kernel_size=3, padding='same'),    # 64 x 750
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.MaxPool1d(5))           # 64 x 150
        self.layer2 = nn.Flatten()      # 9600
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


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def fit_one_cycle(epochs, max_lr, model, train_loader, val_loader, criterion, opt_func):
    torch.cuda.empty_cache()
    history = []
    optimizer = opt_func(model.parameters())
    # set up one cycle lr scheduler
    sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=epochs, steps_per_epoch=len(train_loader))

    for epoch in range(epochs):
        # Training phase：
        model.train()
        train_losses = []
        lrs = []
        correct_sum = 0
        total = 0
        for batch in train_loader:
            signals, targets = batch
            # Run the forward pass
            outputs = model(signals)
            loss = criterion(outputs, targets)

            # Reset the gradients
            optimizer.zero_grad()
            # Calculates gradients
            loss.backward()
            # Perform gradient descent and modifies the weights
            optimizer.step()
            # Record and update lr
            lrs.append(get_lr(optimizer))
            # Modifies the lr value
            sched.step()

            # Track loss, correct_sum
            train_losses.append(loss)
            _, predicted = torch.max(outputs, dim=1)
            _, actual = torch.max(targets, dim=1)
            correct_sum += torch.eq(predicted, actual).sum().item()
            total += len(targets)

        # Evaluation phase at the end of an epoch：
        model.eval()
        with torch.no_grad():
            val_result = evaluate(model, val_loader, criterion)
            result = {'train_acc': correct_sum / total,
                        'train_loss': torch.stack(train_losses).mean().item(),
                        'val_loss': val_result['loss'],
                        'val_acc': val_result['acc'],
                        'lrs': lrs}
        history.append(result)

        print(f"Epoch [{epoch}] : "
              f"train_loss: {result['train_loss']:.4f}, "
              f"train_acc: {result['train_acc']:.4f}, ")

    return history


def try_batch(model, loader):   # useful for debug
    signals, _ = next(iter(loader))
    outputs = model(signals)
    return outputs


def evaluate(model, loader, criterion):
    model.eval()
    with torch.no_grad():
        correct_sum = 0
        total = 0
        losses = []
        for signals, targets in loader:
            outputs = model(signals)
            loss = criterion(outputs, targets)
            _, predicted = torch.max(outputs, dim=1)
            _, actual = torch.max(targets, dim=1)
            correct_sum += torch.eq(predicted, actual).sum().item()
            total += len(targets)
            losses.append(loss)
        result = {'acc': correct_sum / total,
                  'loss': torch.stack(losses).mean().item()}
        return result


if __name__ == "__main__":
    # Load data:
    X1, t1 = integrate_data(132, -7.64)
    X2, t2 = integrate_data(135, -7.64)
    train_dataset = ANNEDataset(X1, t1.transpose())
    val_dataset = ANNEDataset(X2, t2.transpose())

    batch_size = 64
    train_dl = DataLoader(train_dataset, batch_size, shuffle=True)
    val_dl = DataLoader(val_dataset, batch_size, shuffle=True)

    # Initialize model:
    cnn_model = SimpleCNN()

    # Train model:
    history = fit_one_cycle(epochs=10,
                            max_lr=0.01,
                            model=cnn_model,  # may change to other models
                            train_loader=train_dl,
                            val_loader=val_dl,
                            criterion=nn.CrossEntropyLoss(),
                            opt_func=torch.optim.Adam)

    print("Over")   # can place a breakpoint here to check the history
