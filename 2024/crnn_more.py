"""
Model that uses data, data_frequency, and data_scalar.

>>> model = CRNN(num_classes=3, window_size=750, in_channels=10, in_channels_freq=9, in_channels_scalar=8, model='mamba')
>>> x = torch.rand(5, 10, 750)
>>> x_freq = torch.rand(5, 9, 375)
>>> x_scalar = torch.rand(5, 8)
>>> model(x, x_freq, x_scalar).shape
torch.Size([5, 3])

x's shape is (num_windows, window_size, num_dims)
The num_windows dimension is sequential in nature! It is not a batch, even if it looks like one.
Recordings are processed one-by-one, not in batches.

self = CRNN(num_classes=3, window_size=750, in_channels=10, in_channels_freq=9, in_channels_scalar=8, model='mamba')
x = torch.rand(5, 10, 750)
x_freq = torch.rand(5, 9, 375)
x_scalar = torch.rand(5, 8)
"""

import torch.nn as nn
import torch
from typing import Optional
from mamba import MambaConfig, Mamba


# def ghm_c_loss(y: torch.Tensor, t: torch.Tensor, bins: int = 10) -> torch.Tensor:
#     # (N, 2)
#     # (N)
#     p = torch.softmax(y, dim=-1)
#     gradient_norm = torch.abs(p[torch.arange(t.shape[0], dtype=int), t] - 1)
#
#     min_val = torch.min(gradient_norm)
#     max_val = torch.max(gradient_norm)
#     bin_width = (max_val - min_val) / bins
#     gradient_bins = torch.floor((gradient_norm - min_val) / bin_width).long()
#
#     bin_counts = torch.bincount(gradient_bins)
#     gradient_density = bin_counts / torch.sum(bin_counts)
#
#     return 1 / (gradient_density[gradient_bins] + 1e-6)


class MambaNet(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 100, device: str = 'cpu') -> None:
        super().__init__()

        config = MambaConfig(d_model=input_dim, n_layers=3, bias=True, use_cuda=device == 'cuda')
        self.mamba_forward = Mamba(config)
        self.mamba_backward = Mamba(config)

        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.mlp = nn.Sequential(
            nn.BatchNorm1d(input_dim * 2),
            nn.Linear(input_dim * 2, hidden_dim),
            nn.LeakyReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, output_dim)
        )
        self.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.concatenate([self.mamba_forward(x),
                               torch.flip(self.mamba_backward(torch.flip(x, dims=(1,))), dims=(1,))], dim=2)
        B, L, D = x.shape
        return self.mlp(x.reshape(B * L, D)).reshape(B, L, self.output_dim)


# class ResidualBlock(nn.Module):
#     def __init__(self, input_dims: int, output_dims: int, device: str = 'cpu') -> None:
#         super().__init__()
#         self.net1 = nn.Sequential(
#             nn.BatchNorm1d(input_dims),
#             nn.Conv1d(input_dims, input_dims, kernel_size=7, padding=3),
#             nn.LeakyReLU()
#         )
#         self.net2 = nn.Sequential(
#             nn.BatchNorm1d(input_dims),
#             nn.Conv1d(input_dims, output_dims, kernel_size=7, padding=3, stride=2),
#             nn.LeakyReLU()
#         )
#         self.mlp = nn.Sequential(
#             nn.Linear(input_dims + output_dims, output_dims),
#             nn.LeakyReLU(),
#             nn.Linear(output_dims, output_dims),
#             nn.LeakyReLU(),
#             nn.Linear(output_dims, output_dims),
#             nn.LeakyReLU()
#         )
#         self.pool = nn.MaxPool1d(2)
#         self.to(device)
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         # x         is (B, input_dims, N)
#         # output    is (B, output_dims, N // 2)
#         x = x + self.net1(x)
#         _x = self.net2(x)
#         if x.shape[-1] // 2 != _x.shape[-1]:
#             _x = _x[:, :, :-1]
#         __x = torch.concatenate([_x, self.pool(x)], dim=1)
#         return _x + torch.swapdims(self.mlp(torch.swapdims(__x, 1, 2)), 1, 2)


class ResidualBlock(nn.Module):
    """
    >>> self = ResidualBlock(5, 6)
    >>> x = torch.rand(3, 5, 750)
    >>> self(x).shape
    torch.Size([3, 6, 375])
    >>> x = torch.rand(3, 5, 375)
    >>> self(x).shape
    torch.Size([3, 6, 187])
    """
    def __init__(self, input_dims: int, output_dims: int, device: str = 'cpu') -> None:
        super().__init__()
        self.net1 = nn.Sequential(
            nn.BatchNorm1d(input_dims),
            nn.Conv1d(input_dims, input_dims, kernel_size=7, padding=3),
            nn.LeakyReLU()
        )
        self.net2 = nn.Sequential(
            nn.BatchNorm1d(input_dims),
            nn.Conv1d(input_dims, input_dims, kernel_size=15, padding=7),
            nn.LeakyReLU()
        )
        self.net3 = nn.Sequential(
            nn.BatchNorm1d(input_dims),
            nn.Conv1d(input_dims, output_dims, kernel_size=7, padding=3, stride=2),
            nn.LeakyReLU()
        )
        self.mlp = nn.Sequential(
            nn.Linear(input_dims + output_dims, output_dims),
            nn.LeakyReLU(),
            nn.Linear(output_dims, output_dims),
            nn.LeakyReLU(),
            nn.Linear(output_dims, output_dims),
            nn.LeakyReLU()
        )
        self.pool = nn.MaxPool1d(2)
        self.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x         is (B, input_dims, N)
        # output    is (B, output_dims, N // 2)
        x = self.net1(x)
        _x = self.pool(x)
        x = self.net2(x)
        __x = self.net3(x)
        if _x.shape[-1] != __x.shape[-1]:
            __x = __x[:, :, :-1]
        x = torch.concatenate([_x, __x], dim=1)
        return __x + torch.swapdims(self.mlp(torch.swapdims(x, 1, 2)), 1, 2)


class CRNN(nn.Module):
    def __init__(self, num_classes: int, window_size: int, in_channels: int,
                 in_channels_freq: int, in_channels_scalar: int, model: str = 'rnn',
                 device: str = 'cpu') -> None:
        super(CRNN, self).__init__()

        self.window_size = window_size
        self.in_channels = in_channels
        self.in_channels_freq = in_channels_freq
        self.in_channels_scalar = in_channels_scalar
        self.model = model
        self.device = device

        # Regular
        n = 9
        dim_expansion = 3
        block_dims = lambda x: round(in_channels * (1 - x / (n - 1))) + round(in_channels * dim_expansion * x / (n - 1))
        self.conv_blocks = nn.ModuleList([ResidualBlock(block_dims(i - 1), block_dims(i), device) for i in range(1, n)])

        rnn_x_dims = in_channels * dim_expansion * (window_size >> (n - 1))
        assert rnn_x_dims > 0, f"Window size of {window_size} is too small"

        # Frequency
        n = 8
        block_dims = lambda x: round(in_channels_freq * (1 - x / (n - 1))) + round(in_channels_freq * dim_expansion * x / (n - 1))
        self.conv_blocks_freq = nn.ModuleList([ResidualBlock(block_dims(i - 1), block_dims(i), device) for i in range(1, n)])
        self.concatenate_dropout = torch.nn.Dropout(0.4)

        rnn_x_freq_dims = in_channels_freq * dim_expansion * ((window_size // 2) >> (n - 1))
        assert rnn_x_freq_dims > 0, f"Window size of {window_size} is too small"

        rnn_dims = rnn_x_dims + rnn_x_freq_dims + in_channels_scalar
        # self.mlp_preprocess = nn.Sequential(
        #     nn.Linear(rnn_dims, rnn_dims),
        #     nn.LeakyReLU(),
        #     nn.Linear(rnn_dims, rnn_dims),
        #     nn.LeakyReLU(),
        #     nn.Linear(rnn_dims, rnn_dims),
        #     nn.LeakyReLU()
        # )
        # rnn_dims = in_channels + in_channels_scalar + in_channels_freq

        # Recurrent layers:
        assert model in {'lstm', 'gru', 'rnn', 'mamba'}
        self.hidden_size = 50
        if model == 'lstm':
            self.rnn = nn.LSTM(input_size=rnn_dims, hidden_size=self.hidden_size, num_layers=2, bidirectional=True)
        elif model == 'gru':
            self.rnn = nn.GRU(input_size=rnn_dims, hidden_size=self.hidden_size, num_layers=2, bidirectional=True)
        elif model == 'rnn':
            self.rnn = nn.RNN(input_size=rnn_dims, hidden_size=self.hidden_size, num_layers=2, bidirectional=True)
        else:
            self.rnn = MambaNet(rnn_dims, self.hidden_size * 2)
        self.rnn_activation = nn.LeakyReLU()

        mlp_dims = rnn_dims + 2 * self.hidden_size
        self.mlp = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(in_features=mlp_dims, out_features=64 + in_channels_scalar),
            nn.LeakyReLU(),
            nn.Linear(in_features=64 + in_channels_scalar, out_features=64 + in_channels_scalar),
            nn.LeakyReLU(),
            nn.Linear(in_features=64 + in_channels_scalar, out_features=16 + in_channels_scalar),
            nn.Dropout(0.2),
            nn.LeakyReLU(),
            nn.Linear(in_features=16 + in_channels_scalar, out_features=num_classes)
        )
        self.apply(self.initialize_weights)
        self.to(device)

    def initialize_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
            nn.init.kaiming_uniform_(module.weight)

    def forward(self, x: torch.Tensor, x_freq: torch.Tensor, x_scalar: torch.Tensor) -> torch.Tensor:
        assert len(x.shape) == len(x_freq.shape) == 3,              f"Expected x and x_freq to be 3-dimensional, got {x.shape} and {x_freq.shape}"
        assert len(x_scalar.shape) == 2,                            f"Expected x_scalar to be 2-dimensional, got {x_scalar.shape}"
        assert x.shape[0] == x_freq.shape[0] == x_scalar.shape[0],  f"Inconsistent number of windows across x, x_freq, x_scalar: {x.shape[0], x_freq.shape[0], x_scalar.shape[0]}"
        assert x.shape[1] == self.in_channels,                      f"Expected x to have {self.in_channels} channels, got {x.shape[1]}"
        assert x.shape[2] == self.window_size,                      f"Expected {self.window_size} points per window, got {x.shape[2]}"
        assert x_freq.shape[1] == self.in_channels_freq,            f"Expected x_freq to have {self.in_channels_freq} channels, got {x_freq.shape[1]}"
        assert x_freq.shape[2] == x.shape[2] // 2,                  f"Expected x_freq to have {x.shape[2] // 2} bins, got {x_freq.shape[2]}"
        assert x_scalar.shape[1] == self.in_channels_scalar,        f"Expected x_scalar to have {self.in_channels_scalar} channels, got {x_scalar.shape[1]}"

        # x = torch.mean(x, dim=-1, keepdim=True)
        # x_freq = torch.mean(x_freq, dim=-1, keepdim=True)

        # Regular data
        for block in self.conv_blocks:
            x = block(x)

        # Frequency
        for block in self.conv_blocks_freq:
            x_freq = block(x_freq)

        # x = torch.mean(x, dim=-1, keepdim=True)
        # x_freq = torch.mean(x_freq, dim=-1, keepdim=True)

        # Concatenate
        x = torch.cat([x.flatten(1, -1), x_freq.flatten(1, -1)], dim=1)
        x = self.concatenate_dropout(x)
        x = torch.cat([x, x_scalar], dim=1)
        # x = self.mlp_preprocess(x)

        # RNN + MLP
        if self.model == 'mamba':
            _x = self.rnn(x.unsqueeze(0)).squeeze(0)
        else:
            _x, _ = self.rnn(x)
        _x = self.rnn_activation(_x)
        x = torch.cat([_x, x], dim=1)
        x = self.mlp(x)
        return x

    def get_embeddings(self, batch: tuple[tuple[str], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
        _, x, x_freq, x_scalar, _ = batch
        x = x[0]
        x_freq = x_freq[0]
        x_scalar = x_scalar[0]

        # Regular data
        for block in self.conv_blocks:
            x = block(x)

        # Frequency
        for block in self.conv_blocks_freq:
            x_freq = block(x_freq)

        # x = torch.mean(x, dim=-1, keepdim=True)
        # x_freq = torch.mean(x_freq, dim=-1, keepdim=True)

        # Concatenate
        x = torch.cat([x.flatten(1, -1), x_freq.flatten(1, -1)], dim=1)
        x = self.concatenate_dropout(x)
        x = torch.cat([x, x_scalar], dim=1)
        return x

    def get_weight_vector(self, t: torch.Tensor, class_weights: Optional[torch.Tensor],
                          accuracy_weights: Optional[torch.Tensor]) -> torch.Tensor:
        result = torch.ones_like(t).double().to(t.device)
        if class_weights is not None:
            w = torch.empty_like(t).double().to(t.device)
            for i in range(len(class_weights)):
                w[t == i] = class_weights[i]
            result *= w
        if accuracy_weights is not None:
            w = torch.empty_like(t).double().to(t.device)
            for i in range(len(accuracy_weights)):
                w[t == i] = accuracy_weights[i]
            result *= w
        return result

    def loss_function(self, y: torch.Tensor, t: torch.Tensor, weights: torch.Tensor, binary: bool) -> torch.Tensor:
        if binary:
            sigmoid_y = torch.sigmoid(y)
            p_correct = sigmoid_y * (t == 1) + (1 - sigmoid_y) * (t == 0)
            focal_weight = (1 - p_correct) ** 2
            loss = nn.functional.binary_cross_entropy_with_logits(y.double(), t.double(), reduction='none')
            return torch.mean(loss * focal_weight)

        assert len(y.shape) == 2 and len(t.shape) == 1, f'{y.shape}, {t.shape}'
        t = t.long()
        sigmoid_y = torch.log_softmax(y, dim=-1)
        loss = nn.functional.nll_loss(sigmoid_y, t, reduction='none')
        p_correct = torch.exp(sigmoid_y[range(len(sigmoid_y)), t])
        focal_weight = (1 - p_correct) ** 2

        # return torch.mean(focal_weight * loss)  # Focal
        # return torch.mean(weights * (loss + (1 - p_correct)))  # Poly1
        return torch.mean(focal_weight * loss + weights * (1 - p_correct) ** 3)  # Poly1Focal

    def training_step(self,
                      batch: tuple[tuple[str], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
                      weights: Optional[torch.Tensor] = None,
                      weights_transition: Optional[torch.Tensor] = None,
                      accuracy_weights: Optional[torch.Tensor] = None,
                      train: bool = True) -> tuple[tuple[str], torch.Tensor, torch.Tensor, torch.Tensor]:
        ids, X, X_freq, X_scalar, t = batch
        X = X[0].to(self.device)
        X_freq = X_freq[0].to(self.device)
        X_scalar = X_scalar[0].to(self.device)
        t = t[0].to(self.device)

        # Mask metadata values (age/sex) to handle unknown cases
        if train:
            self.train()
            p = 0.3
            if torch.rand(1) < p:
                X_scalar[:, 0] = -1
            if torch.rand(1) < p:
                X_scalar[:, 1] = -1
            # if torch.rand(1) < p:
            #     scale = 4 * torch.std(X, dim=(0, 1))
            #     X += torch.randn_like(X) * scale[None, None] / 10
            # if torch.rand(1) < p:
            #     scale = 4 * torch.std(X_freq, dim=(0, 1))
            #     X_freq += torch.randn_like(X_freq) * scale[None, None] / 10

        else:
            self.eval()

        # sqi = (X_scalar[:, 2] + X_scalar[:, 3]) / 2

        t, t_transition = t[:, 0], t[:, 1]
        y = self.forward(X, X_freq, X_scalar)
        loss = self.loss_function(y[:, :3], t, self.get_weight_vector(t, weights, accuracy_weights), binary=False)
        # loss_transition = self.loss_function(y[:, 3], t_transition, self.get_weight_vector(t_transition, weights_transition), binary=True)
        assert not torch.isnan(loss), f'batch {ids}, y_nan = {torch.any(torch.isnan(y[:, :3]))}, y_nan = {torch.any(torch.isnan(t))}'
        # assert not torch.isnan(loss_transition), f'batch {ids}, y_nan = {torch.any(torch.isnan(y[:, 3]))}, y_nan = {torch.any(torch.isnan(t_transition))}'
        return ids, y, t, loss
