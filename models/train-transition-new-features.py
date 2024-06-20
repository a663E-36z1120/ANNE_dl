import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import matplotlib.pyplot as plt
import numpy as np 

import sys
sys.path.append(".")
sys.path.append("/scratch/alexhemo/ANNE_dl/")
from preprocessing.main_new_features2 import main, get_edf_files, read_strings_from_json
from crnn_tfs_nf import CRNN
from dataloader import ANNEDataset
from dataloader import ANNEDataset_transition

import json
import math
import time
import random

random.seed(42)
torch.manual_seed(42)
timestr = time.strftime("%Y%m%d-%H%M%S")

#N_CLASSES = 4 for binary
N_CLASSES = 5
IS_REM = False
IS_NREM = False
DATA_DIR = "/scratch/alexhemo/real_new_data/ANNE-PSG140421"




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
        xentropy_weight = torch.tensor([1 / 45 ** 1.05, 1 / 46 ** 1.05, 1 / 8.1 ** 1.05]).to(device)
    else:
        xentropy_weight = torch.tensor([1 / 33.1 ** 1.9, 1 / 66.9 ** 1.9]).to(device)
    if IS_NREM:
        xentropy_weight = torch.tensor([1 / 41.3 ** 1.85, 1 / 58.7 ** 1.85]).to(device)
    if IS_REM:
        xentropy_weight = torch.tensor([1 / 91.8 ** 0.9, 1 / 8.2 ** 0.9]).to(device)
    
    if N_CLASSES == 4:
        transition_weight = torch.tensor([1/95 ** 1.05, 1/5 ** 1.05]).to(device)
        xentropy_weight = torch.tensor([1 / 80 ** 1.05, 1 / 23 ** 1.05, 1 / 4.2 ** 1.05]).to(device)
    if N_CLASSES == 5:
        # lll
        transition_weight = torch.tensor([1/95 ** 1.05, 1/5 ** 1.05]).to(device)
        #xentropy_weight = torch.tensor([1 / 43 ** 1.05, 1 / 43.3 ** 1.05, 1 / 7.6 ** 1.05]).to(device)
        #xentropy_weight = torch.tensor([1 / 80 ** 1.05, 1 / 23 ** 1.05, 1 / 4.2 ** 1.05]).to(device) 
        xentropy_weight = torch.tensor([1 / 43.3 ** 1.05, 1 / 40.3 ** 1.05, 1 / 8.0 ** 1.05]).to(device)
    if N_CLASSES == 6:  
        xentropy_weight = torch.tensor([1 / 43 ** 1.05, 1 / 43.3 ** 1.05, 1 / 7.6 ** 1.05]).to(device)
        transition_weight = torch.tensor([1/95 ** 1.05, 1/2.5**1.05, 1/2.5**1.05]).to(device)

    criterion = nn.CrossEntropyLoss(weight=xentropy_weight)
    #criterion2 = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([30.]).float().to(device))
    criterion2 = nn.CrossEntropyLoss(weight=transition_weight)
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
        full_preds = None
        full_labels = None
        full_pred_init = False
        xentropy_loss_total = 0.
        transition_loss_total = 0.
        correct = 0.
        total = 0.
        wake_total = 0.
        nrem_total = 0.
        rem_total = 0.
        transition_correct = 0.
        correct_rem = 0.
        total_trainer_len = 0
        total_predicted_transitions = 1.
        total_labelled_transitions = 1.
        for train_loader in train_loaders:
            for i, (inputs, inputs_freq, inputs_scl, labels, transition_labels, lengths) in enumerate(train_loader):
                model.zero_grad()

                #inputs = inputs.reshape(inputs.shape[0], inputs.shape[1] * inputs.shape[2])
                #inputs_freq = inputs_freq.reshape(inputs_freq.shape[0], inputs_freq.shape[1] * inputs_freq.shape[2])
                
                #inputs_full = torch.cat((inputs, inputs_freq, inputs_scl), 1)

                inputs = inputs.to(device)
                inputs_freq = inputs_freq.to(device)
                inputs_scl = inputs_scl.to(device)
                #inputs_full = torch.flatten(inputs_full)
                #inputs_full = inputs_full.to(device)
                labels = labels.to(device)
                # ANNEDataset_transition provides transition_labels
                transition_labels = transition_labels.to(device)
                
                pred = model(inputs, inputs_freq, inputs_scl)
                
                # pred2 are the transition state predictions
                pred1 = pred[:, :3]
                pred2 = pred[:, 3:]

                xentropy_loss = criterion(pred1, labels) 
                transition_loss = criterion2(pred2, transition_labels)

                total_loss = xentropy_loss + transition_loss / 3
                total_loss.backward()

                xentropy_loss_total += xentropy_loss.item()
                transition_loss_total += transition_loss.item()

                # Calculate running average of accuracy
                pred1 = torch.max(pred1.data, 1)[1]
                pred2 = torch.max(pred2.data, 1)[1] 
                total += labels.size(0)

                
                transition_correct += (pred2 == transition_labels.data).sum().item()

                
                correct += (pred1 == labels.data).sum().item()

            optimizer.step()
            total_trainer_len += len(train_loader)

        # lr_scheduler.step(epoch + i / len(train_loader))    # Important: use this for CosineAnnealingWarmRestarts
        lr_scheduler.step()  # use this for CyclicLR
        current_lr = lr_scheduler.get_last_lr()[0]
        learning_rates.append(current_lr)
        print(f"Learning Rate {current_lr}")
        print("Correct Transitions: " + str(transition_correct / total))
        print("Total Labelled: " + str(total_labelled_transitions))
        print("Total Predicted: " + str(total_predicted_transitions))

        accuracy = correct / total
        avg_xentropy_loss = xentropy_loss_total / total_trainer_len

        test_acc_sum = 0
        test_loss_sum = 0
        
        # Edited for confusion matrix after each epoch
        for test_loader in test_loaders:
            test_acc, test_loss, test_preds, test_labels = evaluate(model, test_loader, criterion, device)
            test_acc_sum += test_acc 
            test_loss_sum += test_loss
            if not full_pred_init:
                full_pred_init = True
                full_preds = test_preds.cpu()
                full_labels = test_labels.cpu()
            else:
                full_preds = np.concatenate((full_preds, test_preds.cpu()), axis=0)
                full_labels = np.concatenate((full_labels, test_labels.cpu()), axis=0)
        
        conf_mat = confusion_matrix(full_labels, full_preds, normalize="true")
        
        plt.plot(full_labels, marker="x", linestyle="", alpha=0.75, markersize=15)
        plt.plot(full_preds, marker=".", linestyle="", alpha=0.75, markersize=15)
        #plt.show()
        
        print(conf_mat)
        #disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat)
        #disp.plot()
        #plt.show()
                

        test_acc = test_acc_sum / len(test_loaders)
        test_loss = test_loss_sum / len(test_loaders)

        if epoch % print_every == 0:
            print("Epoch {}, Train acc: {:.2f}%, Test acc: {:.2f}%".format(epoch, accuracy * 100, test_acc * 100))
            print("Epoch {}, Train loss: {:.2f}, Test loss: {:.2f}".format(epoch, avg_xentropy_loss, test_loss))
            print("BCE loss: {:.2f}".format(transition_loss_total))
            # print("REM acc: {:.2f}".format(rem_correct_percent))

        train_accs.append(accuracy)
        test_accs.append(test_acc)
        train_losses.append(avg_xentropy_loss)
        test_losses.append(test_loss)

        # Check for early stopping
        if test_loss < best_test_loss - min_delta:
            best_test_loss = test_loss
            model_scripted = torch.jit.script(model)
            model_scripted.save(f"/scratch/alexhemo/ANNE_dl/models/checkpoints/tfsnf_rew_{timestr}.pt")
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
    # criterion2 = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([30.]).float().to(device))
    # transition_weight = torch.tensor([1/95 ** 1.05, 1/2.5**1.05, 1/2.5**1.05]).to(device)
    transition_weight = g_transition_weight
    criterion2 = nn.CrossEntropyLoss(weight=transition_weight)
    full_pred_init = False
    full_preds = None
    full_labels = None
    for i, (inputs, inputs_freq, inputs_scl, labels, transition_labels, lengths) in enumerate(loader):
        with torch.no_grad():
            #inputs = inputs.reshape(inputs.shape[0], inputs.shape[1] * inputs.shape[2])
            #inputs_freq = inputs_freq.reshape(inputs_freq.shape[0], inputs_freq.shape[1] * inputs_freq.shape[2])

            #inputs_full = torch.cat((inputs, inputs_freq, inputs_scl), 1)

            inputs = inputs.to(device)
            inputs_freq = inputs_freq.to(device)
            inputs_scl = inputs_scl.to(device)
            #inputs_full = torch.flatten(inputs_full)
            #inputs_full = inputs_full.to(device)
            labels = labels.to(device)
            transition_labels = transition_labels.to(device)
            #transition_labels = transition_labels.view(transition_labels.size(0), 1)
            # inputs = inputs.view(inputs.size(0), -1)
            pred = model(inputs, inputs_freq, inputs_scl)
            pred1 = pred[:, :3]
            pred2 = pred[:, 3:]
            xentropy_loss = criterion(pred1, labels)
            # transition_labels = np.where(transition_labels == 1, [0, 1], [1, 0])
            transition_loss = criterion2(pred2, transition_labels)
            # xentropy_loss = criterion(pred, labels)
            val_loss += xentropy_loss.item() + transition_loss.item()

        pred1 = torch.max(pred1.data, 1)[1]
        total += labels.size(0)
        correct += (pred1 == labels).sum().item()
        
        if not full_pred_init:
            full_pred_init = True
            full_preds = pred1.cpu()
            full_labels = labels.cpu()
        else:
            full_preds = np.concatenate((full_preds, pred1.cpu()), axis=0)
            full_labels = np.concatenate((full_labels, labels.cpu()), axis=0)

    val_acc = correct / total
    val_loss = val_loss / len(loader)
    model.train()
    return val_acc, val_loss, full_preds, full_labels


def plot_grad_histograms(grad_list, epoch, init=False):
    fig, ax = plt.subplots(nrows=1, ncols=len(grad_list), figsize=(5 * len(grad_list), 5))
    for i, grad in enumerate(grad_list):
        plt.subplot(1, len(grad_list), i + 1)
        plt.hist(grad)
        if init:
            plt.title("Grads for Weights (Layer {}-{}) Init".format(i, i + 1))
        else:
            plt.title("Grads for Weights (Layer {}-{}) Epoch {}".format(i, i + 1, epoch))
    #plt.show()


def plot_act_histograms(act_list, epoch, init=False):
    fig, ax = plt.subplots(nrows=1, ncols=len(act_list), figsize=(5 * len(act_list), 5))
    for i, act in enumerate(act_list):
        plt.subplot(1, len(act_list), i + 1)
        plt.hist(act)
        if init:
            plt.title("Activations for Layer {} Init".format(i + 1))
        else:
            plt.title("Activations for Layer {} Epoch {}".format(i + 1, epoch))
    #plt.show()

def get_transitions(t):
    t0 = np.copy(t)
    t1 = np.copy(t)
    t2 = np.copy(t)

    t1 = np.concatenate((np.array([0]), t1))
    t2 = np.concatenate((t2, np.array([0])))

    t3 = t2 - t1
    t4 = t3[1:]

    t0 = np.where(t4 != 0, 3, t0) 
    t0 = np.where(t0 != 3, 0, t0)
    t0 = np.where(t0 == 3, 1, t0)
    
    return t0

def count_elements(arrays_list):
    counts = {
        0: 0,
        1: 0,
        2: 0,
        3: 0,
        4: 0
    }

    for arr in arrays_list:
        unique, counts_per_arr = np.unique(arr, return_counts=True)
        for element, count in zip(unique, counts_per_arr):
            counts[element] += count

    return counts

if __name__ == "__main__":
    print("0")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("1")
    torch.cuda.empty_cache()
    print("2")
    # Load data:
    train_list = get_edf_files("/scratch/alexhemo/real_new_data/ANNE-PSG140421")
    #print(len(train_list))
    validation_list = random.sample(train_list, 20)

    print("train-transition-new-features-no-SHAP.py")
    
    #print(validation_list)
    save_strings_to_json(validation_list, "./validation_list.json")
    test_list = read_strings_from_json("./test_list.json")
    #test_list = random.sample(train_list, 20)
    #save_strings_to_json(test_list, "./test_list.json")
    #
    # train_list = train_list_[:2]
    # print(train_list)
    # validation_list = [train_list_[1]]
    
    g_transition_weight = torch.tensor([1/95 ** 1.05, 1/5 ** 1.05]).to(device)

    #train_list = random.sample(train_list, 20)

    train_dataloaders = []
    valid_dataloaders = []
    for path in train_list:
        # try:
            if path == "/scratch/alexhemo/real_new_data/ANNE-PSG140421/21-12-21-22_46_35.C3882.L3562.344/21-12-21-22_46_35.C3882.L3562.344-score.edf":
                continue
            X, X_freq, X_scl, t = main(path)
            if isinstance(X, str):
                continue
            # for binary classification
            if N_CLASSES == 2:
                t = np.where(t == 2, 1, t)
            t = t[0]
            #print(get_transitions(t))
            X = np.nan_to_num(X, nan=0.5)
            X_freq = np.nan_to_num(X_freq, nan=0.5)
            X_scl = np.nan_to_num(X_scl, nan=0.5)
            dataset = ANNEDataset_transition(X, X_freq, X_scl, np.array([t]), np.array([get_transitions(t)]), device)
            size = len(X)
            if path not in validation_list and path not in test_list:
                train_dataloaders.append(DataLoader(dataset=dataset, batch_size=size))
                print("appended")
            elif path not in test_list:
                valid_dataloaders.append(DataLoader(dataset=dataset, batch_size=size))

    targets = []
    for train_loader in train_dataloaders:
        for i, (inputs, inputs_freq, inputs_scl, labels, transition_labels, lengths) in enumerate(train_loader):
            #print(labels.cpu())
            print(labels.cpu())
            print(transition_labels.cpu())
            targets.append(labels.cpu())
                
    
    result = count_elements(targets)
    wake_count, nrem_count, rem_count = result[0], result[1], result[2]
    transition_count = result[3]
    print(wake_count)

    total = wake_count + rem_count + nrem_count + transition_count

    print(f"wake: {wake_count / total}")
    print(f"nrem: {nrem_count / total}")
    print(f"rem: {rem_count / total}")
    print(f"transition: {transition_count / total}")

    valid_dataloaders = train_dataloaders[-20:]
    train_dataloaders = train_dataloaders[:-20]

    import crnn_t_single
    import crnn_tfs_nf2
    import importlib
    importlib.reload(crnn_tfs_nf2)
    torch.cuda.empty_cache()
    CRNN = crnn_tfs_nf2.CRNN
    
    torch.backends.cudnn.enabled = False 
    
    # Build model
    #model = CRNN(num_classes=N_CLASSES, in_channels=X.shape[1], in_channels_f=X_freq.shape[1], in_channels_s=X_scl.shape[1], model='gru')
    model = CRNN(num_classes=N_CLASSES, in_channels=X.shape[1], in_channels_f=X_freq.shape[1], in_channels_s=X_scl.shape[1], model='gru')
    
    #
    # MODEL_PATH = ""
    # model = torch.load("checkpoints/nrem_model.pt")
    
    
    # Visualize model
    # dummy_input = torch.randn(1024, 6, 25 * 30)
    # torch.onnx.export(model, dummy_input, "./model.onnx")
    
    # Train model:
    learning_rate = 0.0002
    # learning_rate = 0.00125
    epochs = 40
    # dummy_input = torch.randn(4096, X.shape[1], 25*30)
    # dummy_input_freq = torch.randn(4096, X_freq.shape[1], X_freq.shape[2])
    # dummy_input_scl = torch.randn(4096, X_scl.shape[1], X_scl.shape[2])
    # torch.onnx.export(model, dummy_input, "./model.onnx")
    
    # Train model:
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    # Create the learning rate scheduler
    # scheduler = CosineWithWarmupLR(optimizer, warmup_epochs=15, max_epochs=150, max_lr=learning_rate, min_lr=0.000001)
    scheduler = CosineWithWarmupLR(optimizer, warmup_epochs=15, max_epochs=200, max_lr=learning_rate, min_lr=0.000002)
    # scheduler = CyclicLR(optimizer, max_lr = 0.01, base_lr =0.0000001, step_size_up=15, step_size_down=20,
    # gamma=0.85, cycle_momentum=False, mode="triangular2") Run the training loop
    train_accs, test_accs, train_losses, test_losses, learning_rates = train_model(model, optimizer, train_dataloaders, valid_dataloaders, scheduler, epochs=epochs, print_every=1)
    model_scripted = torch.jit.script(model)
    model_scripted.save(f"/scratch/alexhemo/ANNE_dl/models/checkpoints/model_{timestr}.pt")
    print("Model Saved")
    
    plt.plot(train_losses)
    plt.plot(test_losses)
    plt.title("loss")
    #plt.show()
    
    plt.plot(train_accs)
    plt.plot(test_accs)
    plt.title("accuracy")
    #plt.show()
    
    plt.plot(learning_rates)
    plt.plot(learning_rates)
    plt.title("learning_rates")
    #plt.show()

    import shap

    predict_loaders = []
    #train_list = get_edf_files("/home/alexhemo/scratch/data/")
    #validation_list = random.sample(train_list, 20)
    #for file in read_strings_from_json("/scratch/alexhemo/ANNE_dl/models/validation.json"):
    prex = 0
    prexf = 0
    prexs = 0
    validation_list = validation_list[1:5]
    for file in validation_list:
        X, X_freq, X_scl, t = main(file)
        print(X.shape)
        print(X_freq.shape)
        print(X_scl.shape)
        t = t[0]
        # t = np.where(t == 1, 0, t)
        # t = np.where(t == 2, 1, t)
        dataset = ANNEDataset_transition(X, X_freq, X_scl, np.array([t]), np.array([get_transitions(t)]), device)
        size = len(X)
    
        predict_loaders.append(DataLoader(dataset=dataset, batch_size=size))
    
        prex = X
        prexf = X_freq
        prexs = X_scl
    
    time_s = []
    time_f = []
    time = None
    for predict_loader in predict_loaders:
        for i, (data, data_freq, data_scl, labels, transitions, lens) in enumerate(predict_loader):
            data1 = data.cpu()
            data_freq1 = data_freq.cpu()
            data_scl1 = data_scl.cpu()
    
            data1 = data1.reshape(data1.shape[0], data1.shape[1] * data1.shape[2])
            data_freq1 = data_freq1.reshape(data_freq1.shape[0], data_freq1.shape[1] * data_freq1.shape[2])
    
            data_full = torch.cat((data1, data_freq1, data_scl1), 1)
            if time is None:
                time = data_full
            time = np.concatenate((time, data_full))
            #time_f.append(data_freq1)
            #time_s.append(data_scl1)
            #print(data1.shape)
            #print(data_freq1.shape)
            #print(data_scl1.shape)
    
    #model = CRNN1(num_classes=6, in_channels=prex.shape[1], in_channels_f=prexf.shape[1], in_channels_s=prexs.shape[1], model='gru')
    
    
    
    
    #time_s = np.array(time_s)
    #time_f = np.array(time_f)
    #time = np.array(time)
    #print(time[0].shape)
    print("Time Shape")
    print(time.shape)
    time = torch.tensor(time)
    length = time.shape[0]
    samps1 = torch.clone(time)
    samps = samps1[0:10]
    samps = torch.flatten(samps)
    # samps = torch.cat((samps, torch.tensor([10])), 0)
    # for i in range(1, 10):
    #     cur_samps = samps1[i]
    #     cur_samps = torch.flatten(cur_samps)
    #     cur_samps = torch.cat((cur_samps, torch.tensor([1])), 0)
    #     samps = torch.cat((samps, cur_samps), 0)
    #
    # samps = samps.reshape(10, cur_samps.shape[0])
    
    backg = torch.clone(time)
    print("Bacg")
    print(backg.shape)
    backg = backg[10:11]
    backg = torch.flatten(backg)
    # backg = torch.cat((backg, torch.tensor([2])), 0)
    print(backg.shape)
    #backg = backg.reshape(1, backg.shape[0])
    
    #def predicts(input_values):
    #    input_values = torch.tensor(input_values, dtype=torch.float32)
    #    return np.array(model(input_values))
    samps = samps.to(device)
    e = shap.DeepExplainer(model, samps)
    backg = backg.to(device)
    shap_values = e.shap_values(backg, check_additivity=False)
    
    np.set_printoptions(threshold=sys.maxsize)
    for i in shap_values:
        print(i)
