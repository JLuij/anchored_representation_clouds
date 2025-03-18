# By Emilien Dupont https://github.com/EmilienDupont/coin/blob/main/siren.py
# Based on https://github.com/lucidrains/siren-pytorch
import torch
from torch import nn
from math import sqrt


class Sine(nn.Module):
    """Sine activation with scaling.

    Args:
        w0 (float): Omega_0 parameter from SIREN paper.
    """
    def __init__(self, w0=1.):
        super().__init__()
        self.w0 = w0

    def forward(self, x):
        return torch.sin(self.w0 * x)


class SirenLayer(nn.Module):
    """Implements a single SIREN layer.

    Args:
        dim_in (int): Dimension of input.
        dim_out (int): Dimension of output.
        w0 (float):
        c (float): c value from SIREN paper used for weight initialization.
        is_first (bool): Whether this is first layer of model.
        use_bias (bool):
        activation (torch.nn.Module): Activation function. If None, defaults to
            Sine activation.
    """
    def __init__(self, dim_in, dim_out, w0=30., c=6., is_first=False,
                 use_bias=True, activation=None):
        super().__init__()
        self.dim_in = dim_in
        self.is_first = is_first

        self.linear = nn.Linear(dim_in, dim_out, bias=use_bias)

        # Initialize layers following SIREN paper
        w_std = (1 / dim_in) if self.is_first else (sqrt(c / dim_in) / w0)
        nn.init.uniform_(self.linear.weight, -w_std, w_std)
        if use_bias:
            nn.init.uniform_(self.linear.bias, -w_std, w_std)

        self.activation = Sine(w0) if activation is None else activation

    def forward(self, x):
        out = self.linear(x)
        out = self.activation(out)
        return out


class Siren(nn.Module):
    
    def __init__(self, layers,
                 w0=30., w0_initial=30., use_bias=True, final_activation=None):
        super().__init__()
        
        siren_layers = []
        for idx, layer_dim in enumerate(layers):
            if idx == (len(layers) - 2):
                break
            
            next_layer_dim = layers[idx+1]
            is_first = idx == 0
            layer_w0 = w0_initial if is_first else w0
            
            siren_layers.append(SirenLayer(
                dim_in = layer_dim,
                dim_out = next_layer_dim,
                w0 = layer_w0,
                use_bias=use_bias,
                is_first=is_first,
            ))
        
        self.net = nn.Sequential(*siren_layers)

        final_activation = nn.Identity() if final_activation is None else final_activation
        self.last_layer = SirenLayer(dim_in=layers[-2], dim_out=layers[-1], w0=w0,
                                use_bias=use_bias, activation=final_activation)
    
    def forward(self, x):
        x = self.net(x)
        return self.last_layer(x)
