import torch
import torch.nn as nn
from matplotlib.backends.backend_pdf import PdfPages
from torch.utils.data import DataLoader
import numpy as np
import umap
import matplotlib.pyplot as plt
import shap
import os
import matplotlib
matplotlib.use('Agg')


def get_embeddings(path: str) -> tuple[list[str], list[np.ndarray], list[np.ndarray]]:
    ids, embeddings, targets = [], [], []
    for root, dirs, files in os.walk(path):
        for filename in files:
            if not filename.startswith('embedding'):
                continue
            embedding = torch.load(os.path.join(root, filename), map_location='cpu').numpy()

            ids.append(filename[filename.rfind('_') + 1:filename.find('.pt')])
            embeddings.append(embedding[:, :-2])
            targets.append(embedding[:, -2])
    return ids, embeddings, targets


def plot_umap_individual(path: str) -> None:
    ids, embeddings, targets = get_embeddings(path)
    with PdfPages(f'umaps.pdf') as pdf:
        for id, embedding, target in zip(ids, embeddings, targets):
            reducer = umap.UMAP()
            embedding = reducer.fit_transform(embedding)

            fig = plt.figure()
            plt.scatter(embedding[target == 0, 0], embedding[target == 0, 1], c='black', s=1)
            plt.scatter(embedding[target == 1, 0], embedding[target == 1, 1], c='blue', s=1)
            plt.scatter(embedding[target == 2, 0], embedding[target == 2, 1], c='red', s=1)
            plt.title(f'U-Map Embedding, {id}')
            plt.xlabel('x')
            plt.ylabel('y')
            plt.grid()
            plt.legend(['Awake', 'Non-REM Sleep', 'REM Sleep'])
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()


def plot_umap(path: str) -> None:
    _, embeddings, targets = get_embeddings(path)
    embeddings = np.concatenate(embeddings)
    targets = np.concatenate(targets)

    reducer = umap.UMAP()  # https://umap-learn.readthedocs.io/en/latest/basic_usage.html
    embeddings = reducer.fit_transform(embeddings)

    plt.figure()
    plt.scatter(embeddings[targets == 0, 0], embeddings[targets == 0, 1], c='black', s=1)
    plt.scatter(embeddings[targets == 1, 0], embeddings[targets == 1, 1], c='blue', s=1)
    plt.scatter(embeddings[targets == 2, 0], embeddings[targets == 2, 1], c='red', s=1)
    plt.title(f'U-Map Embeddings')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid()
    plt.legend(['Awake', 'Non-REM Sleep', 'REM Sleep'])
    plt.savefig(f'umap.png')
    plt.close()


# https://github.com/google-research/google-research/tree/master/sufficient_input_subsets#references
def shapley_analysis(model: nn.Module, dataloader: DataLoader) -> None:
    data = []
    for batch in dataloader:
        data.append(batch[1][0])
    explainer = shap.DeepExplainer(model, torch.stack(data)[0])
    shap_values = explainer.shap_values(data, check_additivity=False)

