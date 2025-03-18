import torch.nn as nn
import torchvision

class MLP(nn.Module):
    def __init__(self, layers: list):
        """
        Return a simple MLP

        Args:
            layers (list): Provide layers and dimensions as e.g. [in_dim, hidden_dim1, hidden_dim2, out_dim]
        """
        
        super(MLP, self).__init__()
        
        self.mlp = torchvision.ops.MLP(
            layers[0],
            layers[1:],
            activation_layer=nn.ReLU,
        )
    
    def forward(self, x):
        return self.mlp(x)
