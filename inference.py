import sys
import os

sys.path.append(os.getcwd())



import torch
from preprocessing.main import main
import numpy as np
from models.dataloader import ANNEDataset
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

import os
import argparse

MODEL_PATH = "checkpoints/3-class.pt"

# TODO: Make pipeline compatible with CPU workflows
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

def write_csv():
    pass

def write_png():
    pass

if __name__ == "__main__":
    model = torch.load(MODEL_PATH)
    model_scripted = torch.jit.script(model)
    model_scripted.save('checkpoints/3-class-torch-script.pt')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = "cpu"

    # parser = argparse.ArgumentParser()
    # parser.add_argument('-l', '--list', help='delimited list input', type=str)
    # # TODO: parse directory
    # args = parser.parse_args()
    # edf_list = [item for item in args.list.split(',')]




    for path in ["/mnt/Common/data/19-12-17-20_37_24.C823.L775.1-annotated.edf"]:

        X, X_freq, X_scl, t = main(path, inference=True)
        dataset = ANNEDataset(X, X_freq, X_scl, t, device)
        size = len(X)

        data_loader = DataLoader(dataset=dataset, batch_size=size)
        preds = infer(model, data_loader, device)
        print(preds)

        # TODO: write to csv's and png's