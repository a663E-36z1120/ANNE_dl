import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append('/h/335/paulslss300/ANNE_dl')

from preprocessing.main import main, get_edf_files, read_strings_from_json
from crnn_tfs import CRNN
# from crnn_tfs_transformer import CRNN
from dataloader import ANNEDataset

import json
import math
import time
import random
import os

random.seed(42)
torch.manual_seed(42)
timestr = time.strftime("%Y%m%d-%H%M%S")

N_CLASSES = 3
DATA_DIR = "/media/a663e-36z/Common/Data/ANNE-data/"


class CosineWithWarmupLR(LambdaLR):
    def __init__(self, optimizer, warmup_epochs, max_epochs, max_lr=0.001, min_lr=0.0001):
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        # self.base_lr = base_lr
        warmup_scheduler = lambda epoch: epoch / warmup_epochs if epoch < warmup_epochs else \
            (0.5 * (1.0 + math.cos(math.pi * (epoch - warmup_epochs) / (max_epochs - warmup_epochs))) if epoch < max_epochs else min_lr/max_lr)

        super(CosineWithWarmupLR, self).__init__(optimizer, lr_lambda=warmup_scheduler)


def save_strings_to_json(strings_list, filename):
    data = {"strings": strings_list}

    with open(filename, "w") as json_file:
        json.dump(data, json_file, indent=4)

def train_model(model, optimizer, train_loaders, test_loaders, lr_scheduler, epochs=100, print_every=10):

    # Using GPUs in PyTorch is pretty straightforward
    if torch.cuda.is_available():
        print("Using cuda")
        use_cuda = True
        device = torch.device("cuda")
    else:
        device = "cpu"

    if N_CLASSES == 3:
        xentropy_weight = torch.tensor([1 / 33.1 ** 1.25, 1 / 58.7 ** 1.25, 1 / 8.2 ** 1.25]).to(device)
    else:
        xentropy_weight = torch.tensor([1 / 31.3 ** 1.85, 1 / 68.7 ** 1.85]).to(device)

    criterion = nn.CrossEntropyLoss(weight=xentropy_weight)
    train_accs = []
    test_accs = []
    train_losses = []
    test_losses = []
    learning_rates = []
    # Early return parameters
    patience = 1000       # we early return if there is no improvement after patience number of epochs
    counter = 0
    # min_delta = 0.01    # at least 1% accuracy increase is needed to count it as an improvement
    min_delta = 0
    best_test_loss = float("inf")

    # Move the model to GPU, if available
    model.to(device)
    model.train()

    for epoch in range(epochs):
        xentropy_loss_total = 0.
        correct = 0.
        total = 0.
        total_trainer_len = 0
        for train_loader in train_loaders:
            for i, (inputs, inputs_freq, inputs_scl, labels, lengths) in enumerate(train_loader):
                model.zero_grad()
                inputs = inputs.to(device)
                inputs_freq = inputs_freq.to(device)
                inputs_scl = inputs_scl.to(device)
                labels = labels.to(device)
                # inputs = inputs.view(inputs.size(0), -1)  # Flatten input from [batch_size, 1, 28, 28] to [batch_size, 784]
                pred = model(inputs, inputs_freq, inputs_scl)
                xentropy_loss = criterion(pred, labels)
                xentropy_loss.backward()

                xentropy_loss_total += xentropy_loss.item()

                # Calculate running average of accuracy
                pred = torch.max(pred.data, 1)[1]
                total += labels.size(0)
                correct += (pred == labels.data).sum().item()

            optimizer.step()
            total_trainer_len += len(train_loader)

        # lr_scheduler.step(epoch + i / len(train_loader))    # Important: use this for CosineAnnealingWarmRestarts
        lr_scheduler.step()  # use this for CyclicLR
        current_lr = lr_scheduler.get_last_lr()[0]
        learning_rates.append(current_lr)
        print(f"Learning Rate {current_lr}")

        accuracy = correct / total
        avg_xentropy_loss = xentropy_loss_total / total_trainer_len

        test_acc_sum = 0
        test_loss_sum = 0

        for test_loader in test_loaders:
            test_acc, test_loss = evaluate(model, test_loader, criterion, device)
            test_acc_sum += test_acc
            test_loss_sum += test_loss

        test_acc = test_acc_sum / len(test_loaders)
        test_loss = test_loss_sum / len(test_loaders)

        if epoch % print_every == 0:
            print("Epoch {}, Train acc: {:.2f}%, Test acc: {:.2f}%".format(epoch, accuracy * 100, test_acc * 100))
            print("Epoch {}, Train loss: {:.2f}, Test loss: {:.2f}".format(epoch, avg_xentropy_loss, test_loss))

        train_accs.append(accuracy)
        test_accs.append(test_acc)
        train_losses.append(avg_xentropy_loss)
        test_losses.append(test_loss)

        # Check for early stopping
        if test_loss < best_test_loss - min_delta:
            best_test_loss = test_loss
            model_scripted = torch.jit.script(model)
            model_scripted.save(f"checkpoints/es_{timestr}.pt")
            print("saved new model")
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping triggered.")
                break

    return train_accs, test_accs, train_losses, test_losses, learning_rates


def evaluate(model, loader, criterion, device):
    model.eval()  # Change model to 'eval' mode (BN uses moving mean/var).
    correct = 0.
    total = 0.
    val_loss = 0.
    for i, (inputs, inputs_freq, inputs_scl, labels, lengths) in enumerate(loader):
        with torch.no_grad():
            inputs = inputs.to(device)
            inputs_freq = inputs_freq.to(device)
            labels = labels.to(device)
            # inputs = inputs.view(inputs.size(0), -1)
            pred = model(inputs, inputs_freq, inputs_scl)
            xentropy_loss = criterion(pred, labels)
            val_loss += xentropy_loss.item()

        pred = torch.max(pred.data, 1)[1]
        total += labels.size(0)
        correct += (pred == labels).sum().item()

    val_acc = correct / total
    val_loss = val_loss / len(loader)
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


    # validation_list = [DATA_DIR + file for file in read_strings_from_json("./validation.json")]
    # print(validation_list)

    torch.cuda.empty_cache()
    # Load data:
    train_list = get_edf_files("/media/a663e-36z/Common/Data/ANNE-data-expanded/")

    validation_list = random.sample(train_list, 20)
    # print(validation_list)
    save_strings_to_json(validation_list, "./validation_new.json")


    # train_list = train_list_[:2]
    # print(train_list)
    # validation_list = [train_list_[1]]

    train_dataloaders = []
    valid_dataloaders = []
    for path in train_list:
        # try:
            basename = os.path.basename(path)
            save_filename = f"{os.path.splitext(basename)[0]}_preprocessed.pkl"
            save_path = os.path.join(PREPROCESSED_DIR, save_filename)
            X, X_freq, X_scl, t = main(path, save_path=save_path)
            print(f"Loaded: {save_filename}.")
            # for binary classification
            if N_CLASSES == 2:
                t = np.where(t == 2, 1, t)
            # print(t)
            dataset = ANNEDataset(X, X_freq, X_scl, t, device)
            size = len(X)
            if path not in validation_list:
                train_dataloaders.append(DataLoader(dataset=dataset, batch_size=size))

            else:
                valid_dataloaders.append(DataLoader(dataset=dataset, batch_size=size))
        # except:
        #     print(f"Something went wrong for file {path}")
    # random.shuffle(train_list)

    # Build model
    model = CRNN(num_classes=N_CLASSES, in_channels=X.shape[1], in_channels_f=X_freq.shape[1], in_channels_s=X_scl.shape[1], model='lstm')
    #
    # MODEL_PATH = ""
    # model = torch.load(MODEL_PATH)
    # Initialize dataloaders
    # train_dataset = ANNEDataset(X1, X1f, X1s, t1, device)
    # train_dataloader = DataLoader(dataset=train_dataset, batch_size=4096)
    # val_dataset = ANNEDataset(X2, X2f, X2s, t2, device)
    # val_dataloader = DataLoader(dataset=val_dataset)

    # Visualize model
    # dummy_input = torch.randn(1024, 6, 25 * 30)
    # torch.onnx.export(model, dummy_input, "./model.onnx")

    # Train model:
    # learning_rate = 0.00075
    learning_rate = 0.002
    epochs = 400
    # dummy_input = torch.randn(4096, X.shape[1], 25*30)
    # dummy_input_freq = torch.randn(4096, X_freq.shape[1], X_freq.shape[2])
    # dummy_input_scl = torch.randn(4096, X_scl.shape[1], X_scl.shape[2])
    # torch.onnx.export(model, dummy_input, "./model.onnx")

    # Train model:
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    # Create the learning rate scheduler
    # scheduler = CosineWithWarmupLR(optimizer, warmup_epochs=15, max_epochs=150, max_lr=learning_rate, min_lr=0.000001)
    scheduler = CosineWithWarmupLR(optimizer, warmup_epochs=15, max_epochs=105, max_lr=learning_rate, min_lr=0.000002)
    # scheduler = CyclicLR(optimizer, max_lr = 0.01, base_lr =0.0000001, step_size_up=15, step_size_down=20,
    # gamma=0.85, cycle_momentum=False, mode="triangular2") Run the training loop
    train_accs, test_accs, train_losses, test_losses, learning_rates = train_model(model, optimizer, train_dataloaders,
                                                                                   valid_dataloaders,
                                                                                   scheduler,
                                                                                   epochs=epochs,
                                                                                   print_every=1)
    model_scripted = torch.jit.script(model)
    model_scripted.save(f"checkpoints/model_{timestr}.pt")
    print("Model Saved")

    plt.plot(train_losses)
    plt.plot(test_losses)
    plt.title("loss")
    plt.show()

    plt.plot(train_accs)
    plt.plot(test_accs)
    plt.title("accuracy")
    plt.show()

    plt.plot(learning_rates)
    plt.plot(learning_rates)
    plt.title("learning_rates")
    plt.show()
