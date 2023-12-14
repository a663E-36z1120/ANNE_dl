import torch
from preprocessing.main import main, get_edf_files
import numpy as np
from models.dataloader import ANNEDataset
from torch.utils.data import DataLoader
import csv
import matplotlib.pyplot as plt
from scipy.special import softmax

import os
import argparse

MODEL_PATH_3_CLASS = "models/checkpoints/3-class.pt"
MODEL_PATH_2_CLASS_WAKE = "models/checkpoints/2-class.pt"
MODEL_PATH_2_CLASS_NREM = "models/checkpoints/2-class-NREM.pt"
MODEL_PATH_2_CLASS_REM = "models/checkpoints/2-class-REM.pt"


def infer(model, loader, device, return_softmax=False):
    # Change model to 'eval' mode (BN uses moving mean/var).
    model.eval()

    all_preds = []
    all_softmaxes = []

    for i, (inputs, inputs_freq, inputs_scl, labels, lengths) in enumerate(loader):
        with torch.no_grad():
            inputs = inputs.to(device)
            inputs_freq = inputs_freq.to(device)
            inputs_scl = inputs_scl.to(device)
            # inputs = inputs.view(inputs.size(0), -1)
            pred = model(inputs, inputs_freq, inputs_scl)
        if return_softmax:
            pred = pred.cpu().detach().numpy()
            pred_softmax = softmax(pred, axis=1)
            all_softmaxes.append(pred_softmax)
        else:
            pred = torch.max(pred.data, 1)[1]
            pred_np = pred.cpu().detach().numpy()
            all_preds.append(pred_np)

    if return_softmax:
        return np.concatenate(all_softmaxes)
    else:
        return np.concatenate(all_preds)


def ensemble(loader, device, base_model=None, wake_model=None, nrem_model=None, rem_model=None, alpha=1):
    assert (base_model is not None or
            (wake_model is not None and nrem_model is not None and rem_model is not None))

    softmaxes = []

    if base_model:
        base_softmaxes = infer(base_model, loader, device, return_softmax=True)
        softmaxes.append(
            base_softmaxes / 33.1 ** alpha + base_softmaxes / 58.7 ** alpha + base_softmaxes / 8.2 ** alpha)

    if wake_model:
        wake_softmaxes = infer(wake_model, loader, device, return_softmax=True)
        wake_softmaxes_3_class = np.column_stack((wake_softmaxes, wake_softmaxes[:, 1]))
        wake_softmaxes_3_class = wake_softmaxes_3_class / np.sum(wake_softmaxes_3_class, axis=1, keepdims=True)
        softmaxes.append(wake_softmaxes_3_class / 33.1 ** alpha)

    if nrem_model:
        nrem_softmaxes = infer(nrem_model, loader, device, return_softmax=True)
        nrem_softmaxes_3_class = np.column_stack((nrem_softmaxes, nrem_softmaxes[:, 0]))
        nrem_softmaxes_3_class = nrem_softmaxes_3_class / np.sum(nrem_softmaxes_3_class, axis=1, keepdims=True)
        softmaxes.append(nrem_softmaxes_3_class / 58.7 ** alpha)

    if rem_model:
        rem_softmaxes = infer(rem_model, loader, device, return_softmax=True)
        rem_softmaxes_3_class = np.column_stack((rem_softmaxes[:, 0], rem_softmaxes))
        rem_softmaxes_3_class = rem_softmaxes_3_class / np.sum(rem_softmaxes_3_class, axis=1, keepdims=True)
        softmaxes.append(rem_softmaxes_3_class / 8.2 ** alpha)

    ensembled_softmaxes = np.sum(softmaxes, axis=0)
    ensembled_softmaxes = ensembled_softmaxes / np.sum(ensembled_softmaxes, axis=1, keepdims=True)

    return np.argmax(ensembled_softmaxes, axis=1)

def parse_path(filepath):
    # Split the filepath into the directory and filename
    directory, filename = os.path.split(filepath)

    # Split the filename into the name and extension
    name, extension = os.path.splitext(filename)

    return name, directory


def write_csv(vector, file_path):
    try:
        with open(file_path, mode='w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            for value in vector:
                csv_writer.writerow([value])
        print(f"Inference results saved as {file_path}")
    except Exception as e:
        print(f"Error: {e}")


def write_png(vector, file_path):
    try:
        # Create a simple plot of the integers
        plt.plot(vector)
        plt.xlabel('Index')
        plt.ylabel('Value')
        plt.title('Plot of Integers')

        # Save the plot as a PNG file at the specified filepath
        plt.savefig(file_path, format='png')

        # Close the plot to release resources
        plt.close()

        print(f"Plot saved as {file_path}")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    cuda_available = torch.cuda.is_available()
    device = torch.device("cuda:0" if cuda_available else "cpu")

    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--list', help='comma delimited list of edf files', type=str, required=False)
    parser.add_argument('-d', '--dir', help='directory containing edf files', type=str, required=False)
    parser.add_argument('-o', '--output', help='output directory', type=str, required=False, default=".")
    parser.add_argument('-v', '--visualize', help='output directory', required=False, default=False, action="store_true")
    parser.add_argument('-c', '--classes', help='number of classes', type=int, required=False, default=3)
    parser.add_argument('-e', '--ensemble', help='ensemble for 3 class model', required=False, default=False,
                        action="store_true")

    args = vars(parser.parse_args())

    if args["classes"] == 3:
        model_path = MODEL_PATH_3_CLASS
    else:
        model_path = MODEL_PATH_2_CLASS_WAKE

    if not cuda_available:
        model_base = torch.jit.load(model_path, map_location=device)
        model_wake = torch.jit.load(MODEL_PATH_2_CLASS_WAKE, map_location=device)
        model_nrem = torch.jit.load(MODEL_PATH_2_CLASS_NREM, map_location=device)
        model_rem = torch.jit.load(MODEL_PATH_2_CLASS_REM, map_location=device)
    else:
        model_base = torch.jit.load(model_path)
        model_wake = torch.jit.load(MODEL_PATH_2_CLASS_WAKE)
        model_nrem = torch.jit.load(MODEL_PATH_2_CLASS_NREM)
        model_rem = torch.jit.load(MODEL_PATH_2_CLASS_REM)

    edf_list = []

    lst = args["list"]
    dirc = args["dir"]
    out = args["output"]
    esmb = args["ensemble"]

    if out[-1] == "/":
        out = out[:-1]
    visualize = args["visualize"]

    assert os.path.isdir(out)

    if lst:
        edf_list.extend([item for item in lst.split(',')])
    if dirc:
        edf_list.extend(get_edf_files(dirc))

    for path in edf_list:
        X, X_freq, X_scl, t = main(path, inference=True)
        dataset = ANNEDataset(X, X_freq, X_scl, t, device)
        size = len(X)

        data_loader = DataLoader(dataset=dataset, batch_size=size)
        print(f"Performing inference with {MODEL_PATH_3_CLASS} ...")

        if esmb and args["classes"] == 3:
            preds = ensemble(data_loader, device, base_model=model_base,
                             wake_model=model_wake, nrem_model=model_nrem, rem_model=model_rem,)
        else:
            preds = infer(model_base, data_loader, device)

        name, directory = parse_path(path)

        write_csv(preds, f"{out}/{name}_predictions.csv")
        if visualize:
            write_png(preds, f"{out}/{name}_predictions.png")
