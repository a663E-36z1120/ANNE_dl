from plots import *
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from dataset import AnneDataset, AnneDataLoader
import os
import math
import random
from typing import Optional, Union
from sklearn.metrics import balanced_accuracy_score

random.seed(42)
torch.manual_seed(42)

# torch.autograd.set_detect_anomaly(True)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
RUN_LOCALLY = os.name == 'nt'
TIME_LIMIT = 5  # in hours (set to 0 for no limit)

if RUN_LOCALLY:
    NUM_EPOCHS = 1
    DATA_PATH = "../ANNE-PSG240421 toy features"
    OUTPUT_PATH = '../ANNE-PSG240421 toy output'
else:
    import matplotlib
    matplotlib.use('Agg')

    NUM_EPOCHS = 300
    DATA_PATH = '/scratch/a/alim/yrichard/anne-psg-dl-features'
    OUTPUT_PATH = '/scratch/a/alim/yrichard/anne-psg-dl-output'


class CosineWithWarmupLR(LambdaLR):
    def warmup_scheduler(self, epoch: int) -> float:
        if epoch < self.warmup_epochs:
            return epoch / self.warmup_epochs
        elif epoch < self.max_epochs:
            lr = math.pi * (epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)
            return (1 + math.cos(lr)) / 2
        else:
            return self.min_learning_rate / self.max_learning_rate

    def __init__(self, optimizer: torch.optim.Optimizer, warmup_epochs: int, max_epochs: int,
                 max_learning_rate: float = 0.001, min_learning_rate: float = 0.0001) -> None:
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.max_learning_rate = max_learning_rate
        self.min_learning_rate = min_learning_rate

        super(CosineWithWarmupLR, self).__init__(optimizer, lr_lambda=self.warmup_scheduler)


# def early_stopping(stats: dict, stop_stat: float, epoch: int) -> bool:
#     if stop_stat < stats['best'] - stats['early_return_min_delta']:
#         stats['early_return_counter'] = 0
#         stats['best'] = stop_stat
#         stats['best_epoch'] = epoch
#         return False
#
#     stats['early_return_counter'] += 1
#     return stats['early_return_counter'] >= stats['early_return_patience']

def get_accuracy_weights(y: list[Union[np.ndarray, torch.Tensor]],
                         t: list[Union[np.ndarray, torch.Tensor]]) -> Optional[torch.Tensor]:
    assert len(y) == len(t)
    assert len(y[0].shape) == len(t[0].shape) == 1, f'{y[0].shape}, {t[0].shape}'

    for i in range(len(y)):
        if isinstance(y[i], torch.Tensor):
            y[i] = tensor_to_array(y[i])
        if isinstance(t[i], torch.Tensor):
            t[i] = tensor_to_array(t[i])
    y = np.concatenate(y)
    t = np.concatenate(t)
    labels = np.unique(t)
    expected_labels = 3
    if len(labels) != expected_labels:
        return None
    accuracies = torch.tensor([np.sum((y == t) & (t == i)) / np.sum(t == i) for i in np.unique(t)])
    weights = 1 - accuracies / torch.sum(accuracies)
    weights *= expected_labels
    return weights


def get_accuracies(y: list[Union[np.ndarray, torch.Tensor]],
                   t: list[Union[np.ndarray, torch.Tensor]]) -> tuple[float, float, float]:
    assert len(y) == len(t)
    assert len(y[0].shape) == len(t[0].shape) == 1, f'{y[0].shape}, {t[0].shape}'

    for i in range(len(y)):
        if isinstance(y[i], torch.Tensor):
            y[i] = tensor_to_array(y[i])
        if isinstance(t[i], torch.Tensor):
            t[i] = tensor_to_array(t[i])

    accuracy = [balanced_accuracy_score(t[i], y[i]) for i in range(len(y))]
    accuracy_equal = np.mean(accuracy)
    accuracy_stdev = np.std(accuracy)
    accuracy_weighted = balanced_accuracy_score(np.concatenate(t), np.concatenate(y))
    return float(accuracy_equal), float(accuracy_stdev), accuracy_weighted


def train_model(model: nn.Module,
                train_dataloader: DataLoader,
                valid_dataloader: DataLoader,
                epochs: int = 100,
                print_every: int = 10) -> dict:
    optimizer = torch.optim.AdamW(model.parameters())
    # scheduler = CosineWithWarmupLR(optimizer, warmup_epochs=15, max_epochs=200,
    #                                max_learning_rate=0.0002, min_learning_rate=0.000002)
    stats = {
        'train_losses': [],
        'train_accs_equal': [],
        'train_accs_stdev': [],
        'train_accs_weighted': [],
        'valid_losses': [],
        'valid_accs_equal': [],
        'valid_accs_stdev': [],
        'valid_accs_weighted': [],
        'learning_rates': [],
        'best_stat': None,
        'best_epoch': 0
    }

    model.train()
    weights = get_weights(train_dataloader, i=0, verbose=True)
    weights_transition = get_weights(train_dataloader, i=1, verbose=True)

    time = datetime.now()
    print(f"Beginning training on {sum(p.numel() for p in model.parameters())} parameters...")

    # accuracy_weights = None
    for epoch in range(epochs):
        torch.cuda.empty_cache()

        train_loss = 0.
        y_all = []
        t_all = []

        for i, batch in enumerate(train_dataloader):
            model.zero_grad()
            ids, y, t, loss = model.training_step(batch, weights, weights_transition)
            assert not torch.isnan(loss), f'NaN loss training, epoch {epoch}, batch {i}, id {ids[0]}'
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            y_all.append(torch.max(y[:, :3], dim=1).indices)
            t_all.append(t)

            if RUN_LOCALLY:
                print(f'Epoch {epoch}, batch {i}, time {datetime.now()}, loss = {loss.item()}')

            if TIME_LIMIT > 0 and (datetime.now() - time).seconds > (TIME_LIMIT - 0.5) * 60 * 60:
                print(f"Approaching time limit, stopping training after epoch {epoch}, batch {i}.")
                print(f"Training complete at {datetime.now()} with duration {datetime.now() - time}")
                save_model(model)
                return stats

        # scheduler.step()

        train_acc_equal, train_acc_stdev, train_acc_weighted = get_accuracies(y_all, t_all)
        # accuracy_weights = get_accuracy_weights(y_all, t_all)
        train_loss /= len(train_dataloader)

        # if RUN_LOCALLY:
        #     valid_acc_equal, valid_acc_stdev, valid_acc_weighted, valid_loss = 0., 0., 0., 0.
        # else:
        valid_acc_equal, valid_acc_stdev, valid_acc_weighted, valid_loss = evaluate(model, valid_dataloader, weights, weights_transition)

        # current_lr = scheduler.get_last_lr()[0]
        current_lr = optimizer.param_groups[-1]['lr']
        stats['learning_rates'].append(current_lr)
        stats['train_accs_equal'].append(train_acc_equal)
        stats['train_accs_stdev'].append(train_acc_stdev)
        stats['train_accs_weighted'].append(train_acc_weighted)
        stats['train_losses'].append(train_loss)
        stats['valid_accs_equal'].append(valid_acc_equal)
        stats['valid_accs_stdev'].append(valid_acc_stdev)
        stats['valid_accs_weighted'].append(valid_acc_weighted)
        stats['valid_losses'].append(valid_loss)

        if epoch % print_every == 0:
            print(f"Epoch {epoch}:")
            print(f"\tLearning Rate:                   {current_lr}")
            print(f"\tTrain Accuracy (Equal):          ({round(train_acc_equal * 100, 2)} ± {round(train_acc_stdev * 100, 2)})%")
            print(f"\tTrain Accuracy (Weighted):       {round(train_acc_weighted * 100, 2)}%")
            print(f"\tTrain Loss:                      {train_loss}")
            print(f"\tValidation Accuracy (Equal):     ({round(valid_acc_equal * 100, 2)} ± {round(valid_acc_stdev * 100, 2)})%")
            print(f"\tValidation Accuracy (Weighted):  {round(valid_acc_weighted * 100, 2)}%")
            print(f"\tValidation Loss:                 {valid_loss}")

        if (stats['best_stat'] is None or valid_acc_weighted > stats['best_stat']) and epoch > 10 and not RUN_LOCALLY:
            stats['best_stat'] = valid_acc_weighted
            stats['best_epoch'] = epoch
            save_model(model, epoch)

    print(f"Training complete at {datetime.now()} with duration {datetime.now() - time}")
    save_model(model)
    return stats


def evaluate(model: nn.Module, dataloader: DataLoader, weights: torch.Tensor,
             weights_transition: torch.Tensor) -> tuple[float, float, float, float]:
    val_loss = 0.
    y_all = []
    t_all = []

    for batch in dataloader:
        with torch.no_grad():
            _, y, t, loss = model.training_step(batch, weights, weights_transition, train=False)

        val_loss += loss.item()
        y_all.append(torch.max(y[:, :3], dim=1).indices)
        t_all.append(t)

    val_acc_equal, val_acc_stdev, val_acc_weighted = get_accuracies(y_all, t_all)
    val_loss /= len(dataloader)
    return val_acc_equal, val_acc_stdev, val_acc_weighted, val_loss


def save_predictions(model: nn.Module, dataloader: DataLoader, save_embeddings: bool = True,
                     save_results: bool = True, prefix: str = '') -> None:
    assert save_embeddings or save_results
    prefix = f'{prefix}_' if len(prefix) > 0 else ''

    for batch in dataloader:
        with torch.no_grad():
            if save_embeddings:
                id, _, _, _, t = batch
                t = t[0]
                x = model.get_embeddings(batch)
                path = f'{OUTPUT_PATH}/embedding_{prefix}{id[0]}.pt'
                torch.save(torch.concatenate([x, t], dim=1), path)

            if save_results:
                id, y, t, _ = model.training_step(batch, train=False)
                path = f'{OUTPUT_PATH}/result_{prefix}{id[0]}.pt'
                torch.save(torch.concatenate([y[:, :3], t.unsqueeze(1)], dim=1), path)


####################################################################################################
# Helper functions
####################################################################################################
def tensor_to_array(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


def save_model(model: nn.Module, epoch: int = 0) -> None:
    if epoch > 0:
        torch.save(model.state_dict(), f"{OUTPUT_PATH}/model{epoch}.pt")
    else:
        torch.save(model.state_dict(), f"{OUTPUT_PATH}/model.pt")


def load_model(model: nn.Module, epoch: int = 0) -> None:
    if epoch > 0:
        model.load_state_dict(torch.load(f"{OUTPUT_PATH}/model{epoch}.pt"))
    else:
        model.load_state_dict(torch.load(f"{OUTPUT_PATH}/model.pt"))


def plot_diagnostics(model: nn.Module, dataloader: DataLoader, prefix: str = '') -> None:
    logits_all = []
    t_all = []
    ids = []

    for batch in dataloader:
        with torch.no_grad():
            id, logits, t, _ = model.training_step(batch, train=False)
            assert len(id) == 1, "Set your batch size to 1"

            logits_all.append(logits[:, :3].cpu().numpy())
            t_all.append(t.cpu().numpy())
            ids.append(id[0])

    logits = np.concatenate(logits_all, axis=0)
    t = np.concatenate(t_all, axis=0)

    class_names = ['Awake', 'Non-REM Sleep', 'REM Sleep']
    plot_confusion_matrix(OUTPUT_PATH, logits, t, class_names, prefix)
    plot_recordings(OUTPUT_PATH, logits_all, t_all, ids, class_names, prefix)


def get_weights(dataloader: DataLoader, i: int, verbose: bool = False) -> torch.Tensor:
    targets = [labels[0, :, i].detach().cpu().int() for _, _, _, _, labels in dataloader]
    targets = np.concatenate(targets, axis=0, dtype=int)
    targets, target_counts = np.unique(targets, return_counts=True)

    if verbose:
        print('Target Distributions:')
        for target, target_count in zip(targets, target_counts):
            print(f'{target}: {target_count} = {100 * round(target_count / np.sum(target_counts), 2)}%')

    distribution = target_counts / np.sum(target_counts)
    weights = 1 / distribution ** 1.05
    weights /= np.mean(weights)
    weights = np.clip(weights, 1 / 100, 100)

    return torch.from_numpy(weights).to(DEVICE)


def main() -> nn.Module:
    dataset = AnneDataset(DATA_PATH, DEVICE)

    train_dataloader = AnneDataLoader(dataset, mode='training', batch_size=1)
    valid_dataloader = AnneDataLoader(dataset, mode='validation', batch_size=1)

    # from crnn_t_nf2 import CRNN
    # model = CRNN(num_classes=3, window_size=dataset.window_size, in_channels=dataset.num_dims,
    #              model='gru', device=DEVICE)

    # from crnn_tfs_nf2 import CRNN
    # from crnn_mlp import CRNN
    from crnn_more import CRNN
    model = CRNN(num_classes=3,  window_size=dataset.window_size, in_channels=dataset.num_dims,
                 in_channels_freq=dataset.num_dims_frequency, in_channels_scalar=dataset.num_dims_scalar,
                 model='mamba', device=DEVICE)
    # load_model(model, 239)

    train_stats = train_model(model, train_dataloader, valid_dataloader, epochs=NUM_EPOCHS, print_every=1)

    print(f"Best epoch: {train_stats['best_epoch']} ({train_stats['best_stat']})")
    load_model(model, train_stats['best_epoch'])

    train_dataloader.split_into_chunks = False
    plot_training_curves(OUTPUT_PATH, train_stats)
    plot_diagnostics(model, train_dataloader, 'train')
    plot_diagnostics(model, valid_dataloader, 'valid')
    save_predictions(model, valid_dataloader, save_embeddings=True, save_results=True, prefix='valid')
    return model


if __name__ == "__main__":
    t1 = datetime.now()
    print(f"Started job at {t1}")

    model = main()

    t2 = datetime.now()
    print(f"Finished job at {t2} with job duration {t2 - t1}")
