import torch
from preprocessing.main import main, read_strings_from_json
import numpy as np
from dataloader import ANNEDataset
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


MODEL_PATH = "checkpoints/es_20231005-152829.pt"


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
    model = torch.jit.load(MODEL_PATH)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = "cpu"

    predict_loaders = []
    for path in read_strings_from_json("./validation.json"):
        X, X_freq, X_scl, t = main(path)
        # t = np.where(t == 2, 1, t)
        dataset = ANNEDataset(X, X_freq, X_scl, t, device)
        size = len(X)

        predict_loaders.append(DataLoader(dataset=dataset, batch_size=size))

    # test_dataset = ANNEDataset(X_test, X_freq_test, X_scl_test, t_test, device)
    # test_dataloader = DataLoader(dataset=test_dataset, batch_size=test_dataset.__len__())
    labels1 = None
    preds1 = None

    for test_dataloader in predict_loaders:
        labels, preds = predict(model, test_dataloader, device)
        if labels1 is None:
            labels1, preds1 = labels, preds
        else:
            labels1 = np.concatenate((labels1, labels), axis=0)
            preds1 = np.concatenate((preds1, preds), axis=0)

    print(labels1)
    print(preds1)
    conf_mat = confusion_matrix(labels1, preds1, normalize="true")

    print(np, sum(labels1) * 30 / 60 / 60)

    plt.plot(labels1, marker="x", linestyle="", alpha=0.75, markersize=15)
    plt.plot(preds1, marker=".", linestyle="", alpha=0.75, markersize=15)
    plt.savefig("plot1.png")  # Specify the path and filename where you want to save the plot
    plt.close()  # Close the plot to free up memory

    disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat)
    disp.plot()
    plt.savefig("confusion_matrix.png")  # Specify the path and filename where you want to save the plot
    plt.close()  # Close the plot to free up memory

    # Visualize model
    dummy_input = torch.randn(4096, X.shape[1], 25 * 30).to(device)
    dummy_input_freq = torch.randn(4096, X_freq.shape[1], X_freq.shape[2]).to(device)
    dummy_input_scl = torch.randn(4096, X_scl.shape[1]).to(device)
    torch.onnx.export(model, args=(dummy_input, dummy_input_freq, dummy_input_scl), f="./model.onnx")

    pass
