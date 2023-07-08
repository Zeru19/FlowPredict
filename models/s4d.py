import math

import numpy as np
import torch
from torch import nn
from einops import repeat, rearrange

from models.base import BaseModel


class TimeEncoder(nn.Module):
    """Time and mask encoder"""

    def __init__(self, d_model):
        super().__init__()

        self.mask_embedding = nn.Embedding(2, d_model)

    def forward(self, x, mark, mask):
        mask_encode = self.mask_embedding(mask)

        return x + mask_encode

class DropoutNd(nn.Module):
    def __init__(self, p: float = 0.5, tie=True, transposed=True):
        """
        tie: tie dropout mask across sequence lengths (Dropout1d/2d/3d)
        """
        super().__init__()
        if p < 0 or p >= 1:
            raise ValueError("dropout probability has to be in [0, 1), " "but got {}".format(p))
        self.p = p
        self.tie = tie
        self.transposed = transposed
        self.binomial = torch.distributions.binomial.Binomial(probs=1 - self.p)

    def forward(self, X):
        """X: (batch, dim, lengths...)."""
        if self.training:
            if not self.transposed:
                X = rearrange(X, 'b ... d -> b d ...')
            # binomial = torch.distributions.binomial.Binomial(probs=1-self.p) # This is incredibly slow because of CPU -> GPU copying
            mask_shape = X.shape[:2] + (1,) * (X.ndim - 2) if self.tie else X.shape
            # mask = self.binomial.sample(mask_shape)
            mask = torch.rand(*mask_shape, device=X.device) < 1. - self.p
            X = X * mask * (1.0 / (1 - self.p))
            if not self.transposed:
                X = rearrange(X, 'b d ... -> b ... d')
            return X
        return X


class S4DKernel(nn.Module):
    """Generate convolution kernel from diagonal SSM parameters."""

    def __init__(self, d_model, N=64, dt_min=0.001, dt_max=0.1, lr=None):
        super().__init__()
        # Generate dt
        H = d_model
        log_dt = torch.rand(H) * (
                math.log(dt_max) - math.log(dt_min)
        ) + math.log(dt_min)

        # H: number of channels
        # N: number of state dimensions
        # C: project matrix from x to y
        C = torch.randn(H, N // 2, dtype=torch.cfloat)
        self.C = nn.Parameter(torch.view_as_real(C))
        self.register("log_dt", log_dt, lr)

        log_A_real = torch.log(0.5 * torch.ones(H, N // 2))
        A_imag = math.pi * repeat(torch.arange(N // 2), 'n -> h n', h=H)
        self.register("log_A_real", log_A_real, lr)
        self.register("A_imag", A_imag, lr)

    def forward(self, L):
        """
        returns: (..., c, L) where c is number of channels (default 1)
        """

        # Materialize parameters
        dt = torch.exp(self.log_dt)  # (H)
        C = torch.view_as_complex(self.C)  # (H N)
        A = -torch.exp(self.log_A_real) + 1j * self.A_imag  # (H N)

        # Vandermonde multiplication
        dtA = A * dt.unsqueeze(-1)  # (H N)
        K = dtA.unsqueeze(-1) * torch.arange(L, device=A.device)  # (H N L)
        C = C * (torch.exp(dtA) - 1.) / A
        K = 2 * torch.einsum('hn, hnl -> hl', C, torch.exp(K)).real

        return K

    def register(self, name, tensor, lr=None):
        """Register a tensor with a configurable learning rate and 0 weight decay"""

        if lr == 0.0:
            self.register_buffer(name, tensor)
        else:
            self.register_parameter(name, nn.Parameter(tensor))

            optim = {"weight_decay": 0.0}
            if lr is not None:
                optim["lr"] = lr
            setattr(getattr(self, name), "_optim", optim)


class S4DModule(nn.Module):
    """
    Standard implementation of Diagonal State Space Model (S4D).
    Stolen from https://github.com/HazyResearch/state-spaces
    Gu A, Goel K, Gupta A, et al. On the parameterization and initialization of diagonal state space models[J].
    Advances in Neural Information Processing Systems, 2022, 35: 35971-35983.
    """

    def __init__(self, d_model, d_state=64, dropout=0.0, transposed=True, **kernel_args):
        super().__init__()
        self.h = d_model
        self.n = d_state
        self.dropout = dropout
        self.transposed = transposed

        self.D = nn.Parameter(torch.randn(self.h))

        # SSM Kernel
        self.kernel = S4DKernel(self.h, N=self.n, **kernel_args)

        # Pointwise
        self.activation = nn.GELU()
        # dropout_fn = nn.Dropout2d # NOTE: bugged in PyTorch 1.11
        dropout_fn = DropoutNd
        self.dropout = dropout_fn(dropout) if dropout > 0.0 else nn.Identity()

        # position-wise output transform to mix features
        self.output_linear = nn.Sequential(
            nn.Conv1d(self.h, 2 * self.h, kernel_size=1),
            nn.GLU(dim=-2),
        )

    def forward(self, u, src_key_padding_mask=None, **kwargs):  # absorbs return_output and transformer src mask
        """ Input and output shape (B, H, L) """
        if not self.transposed:
            u = u.transpose(-1, -2)
        L = u.size(-1)

        # padding mask
        if src_key_padding_mask is not None:  # (B, L)
            src_key_padding_mask = repeat(src_key_padding_mask, 'b l -> b h l', h=self.h)
            u = u * (~src_key_padding_mask)

        # Compute SSM Kernel
        k = self.kernel(L=L)  # (H L)

        # Convolution
        k_f = torch.fft.rfft(k, n=2 * L)  # (H L)
        u_f = torch.fft.rfft(u, n=2 * L)  # (B H L)
        y = torch.fft.irfft(u_f * k_f, n=2 * L)[..., :L]  # (B H L)

        # Compute D term in state space equation - essentially a skip connection
        y = y + u * self.D.unsqueeze(-1)

        y = self.dropout(self.activation(y))
        y = self.output_linear(y)
        if not self.transposed:
            y = y.transpose(-1, -2)
        return y


class S4DLayer(nn.Module):
    """Absorbs Transformer Encoder Layer"""

    def __init__(self, d_model, d_state=64, hidden_size=64, dropout=0.0, transposed=True, norm_first=True, **kernel_args):
        super(S4DLayer, self).__init__()
        self.norm_first = norm_first

        self.norm1 = nn.LayerNorm(d_model)
        self.s4d_module = S4DModule(d_model, d_state, dropout, transposed, **kernel_args)
        self.dropout1 = DropoutNd(dropout) if dropout > 0.0 else nn.Identity()
        # feed forward model
        self.linear1 = nn.Linear(d_model, hidden_size)
        self.activation = nn.GELU()
        self.dropout2 = DropoutNd(dropout) if dropout > 0.0 else nn.Identity()
        self.linear2 = nn.Linear(hidden_size, d_model)
        self.dropout3 = DropoutNd(dropout) if dropout > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, u, src_key_padding_mask=None, **kwargs):
        if self.norm_first:
            u = u + self._s4d_block(self.norm1(u), src_key_padding_mask, **kwargs)
            y = u + self._ff_block(self.norm2(u))
        else:
            u = self.norm1(u + self._s4d_block(u, src_key_padding_mask, **kwargs))
            y = self.norm2(u + self._ff_block(u))
        return y

    def _s4d_block(self, u, src_key_padding_mask, **kwargs):
        y = self.s4d_module(u, src_key_padding_mask, **kwargs)
        y = self.dropout1(y)
        return y

    def _ff_block(self, y):
        y = self.linear1(y)
        y = self.activation(y)
        y = self.dropout2(y)
        y = self.linear2(y)
        y = self.dropout3(y)
        return y


class S4DModel(BaseModel):
    def __init__(self, lag, horizon, d_model, input_size, output_size, normalizer, denormalizer, 
                 d_state=64, num_layers=2, hidden_size=128, dropout=0.1):
        super(S4DModel, self).__init__(normalizer, denormalizer,
                                       'S4DModel-' + 
                                       f'lag{lag}-hor{horizon}-' +
                                       f'h{hidden_size}-l{num_layers}-drop{dropout}')

        self.horizon = horizon
        # TODO Convert rep vector from input_size to d_model
        self.d_model = d_model
        self.d_model = self.input_size = input_size
        self.output_size = output_size

        self.time_embed = TimeEncoder(d_model)
        self.encoders = nn.ModuleList(
            [S4DLayer(self.d_model, d_state, hidden_size, dropout=dropout, transposed=False) for _ in range(num_layers)]
        )
        self.out_linear = nn.Linear(self.d_model, output_size)

    def forward(self, x, mark, mask):
        """
        :param x: (b, lag + horizon, h)

        """
        x = self.normalizer(x)
        x = x * repeat(1 - mask, 'b l -> b l h', h=self.d_model)

        x = self.time_embed(x, mark, mask)
        for layer in self.encoders:
            x = layer(x)
        x = self.out_linear(x)
        x = self.denormalizer(x)

        return x[:, -self.horizon:, :]
