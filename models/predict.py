import torch
import json
from preprocessing.main import integrate_data
import numpy as np
from dataloader import ANNEDataset
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


MODEL_PATH = "./model.pt"


def predict(model, loader, device):
    # Change model to 'eval' mode (BN uses moving mean/var).
    model.eval()

    all_preds = []
    all_labels = []

    for i, (inputs, inputs_freq, inputs_scl, labels, lengths) in enumerate(loader):
        with torch.no_grad():
            inputs = inputs.to(device)
            labels = labels.to(device)
            inputs_freq = inputs_freq.to(device)
            inputs_scl = inputs_scl.to(device)
            # inputs = inputs.view(inputs.size(0), -1)
            pred = model(inputs, inputs_freq, inputs_scl)

        pred = torch.max(pred.data, 1)[1]
        pred_np = pred.cpu().detach().numpy()
        labels_np = labels.cpu().detach().numpy()
        all_preds.append(pred_np)
        all_labels.append(labels_np)

    return np.concatenate(all_labels), np.concatenate(all_preds)


if __name__ == "__main__":
    model = torch.load(MODEL_PATH)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



    test_id_list = [135, 157, 248]

    with open('../preprocessing/white_list.json') as json_file:
        white_list = json.load(json_file)

    X_test = None
    X_freq_test = None
    X_scl_test = None
    t_test = None

    for id in test_id_list:
        lag = white_list[str(id)]
        X, X_freq, X_scl, t = integrate_data(id, lag)
        if X_test is None:
            X_test = X
            X_freq_test = X_freq
            X_scl_test = X_scl
            t_test = t
        else:
            X_test = np.concatenate((X_test, X), axis=0)
            X_freq_test = np.concatenate((X_freq_test, X_freq), axis=0)
            X_scl_test = np.concatenate((X_scl_test, X_scl), axis=0)
            t_test = np.concatenate((t_test, t), axis=1)

    test_dataset = ANNEDataset(X_test, X_freq_test, X_scl_test, t_test, device)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=test_dataset.__len__())

    labels, preds = predict(model, test_dataloader, device)
    conf_mat = confusion_matrix(labels, preds, normalize="true")

    disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat)
    disp.plot()
    plt.show()

    # Visualize model
    dummy_input = torch.randn(4096, X.shape[1], 25 * 30).to(device)
    dummy_input_freq = torch.randn(4096, X_freq.shape[1], X_freq.shape[2]).to(device)
    dummy_input_scl = torch.randn(4096, X_scl.shape[1]).to(device)
    torch.onnx.export(model, args=(dummy_input, dummy_input_freq, dummy_input_scl), f="./model.onnx")



    pass

