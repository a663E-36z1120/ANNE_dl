import torch
from preprocessing.main import main, get_edf_files
import numpy as np
from models.dataloader import ANNEDataset
from torch.utils.data import DataLoader
import csv
import matplotlib.pyplot as plt

import os
import argparse

MODEL_PATH_3_CLASS = "models/checkpoints/3-class.pt"
MODEL_PATH_2_CLASS = "models/checkpoints/2-class.pt"


def infer(model, loader, device):
    # Change model to 'eval' mode (BN uses moving mean/var).
    model.eval()

    all_preds = []

    for i, (inputs, inputs_freq, inputs_scl, labels, lengths) in enumerate(loader):
        with torch.no_grad():
            inputs = inputs.to(device)
            inputs_freq = inputs_freq.to(device)
            inputs_scl = inputs_scl.to(device)
            # inputs = inputs.view(inputs.size(0), -1)
            pred = model(inputs, inputs_freq, inputs_scl)

        pred = torch.max(pred.data, 1)[1]
        pred_np = pred.cpu().detach().numpy()
        all_preds.append(pred_np)

    return np.concatenate(all_preds)


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

    args = vars(parser.parse_args())

    if args["classes"] == 3:
        model_path = MODEL_PATH_3_CLASS
    else:
        model_path = MODEL_PATH_2_CLASS

    if not cuda_available:
        model = torch.jit.load(model_path, map_location=device)
    else:
        model = torch.jit.load(model_path)

    edf_list = []

    lst = args["list"]
    dirc = args["dir"]
    out = args["output"]
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
        preds = infer(model, data_loader, device)

        name, directory = parse_path(path)

        write_csv(preds, f"{out}/{name}_predictions.csv")
        if visualize:
            write_png(preds, f"{out}/{name}_predictions.png")
