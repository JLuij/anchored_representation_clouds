import torch


def psnr(gt, recon):
    """
    Compute PSNR [dB]

    Args:
        gt: Ground truth signal
        recon: Reconstructed signal

    Returns:
        psnr: PSNR [dB]
    """
    
    psnr = 10 * torch.log10(torch.max(gt) / torch.mean(pow(gt - recon, 2)))

    return psnr

def generate_grid_uneqsidelen(shape, range_, device='cpu'):
    # Generate 1D grid values along each dimension
    # Returns a list of `dim` grids, each being (points_per_dim x points_per_dim)
    grid = torch.meshgrid([torch.linspace(-range_, range_, dim, device=device) for dim in shape], indexing='ij')
    grid = torch.stack(grid)
    grid = grid.permute(*range(1, len(shape)+1), 0)
    return grid


def generate_grid_integer_coords(shape, device='cpu'):
    if isinstance(shape, int):
        shape = (shape, shape)
        num_dims = 2
    else:
        num_dims = len(shape)
    
    # Returns a list of `dim` grids, each being (points_per_dim x points_per_dim)
    grid = torch.meshgrid([torch.arange(dimension, device=device) for dimension in shape], indexing='ij')
    grid = torch.stack(grid)
    grid = grid.permute(*range(1, num_dims+1), 0)
    
    return grid

## By francois-rozet at https://github.com/pytorch/pytorch/issues/35674#issuecomment-1741608630
def unravel_index(indices: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    """
    This is a `torch` implementation of `numpy.unravel_index`.
    Converts a tensor of flat indices into a tensor of coordinate vectors.
    E.g. [1,2,3] with shape (2,2) will convert to [[0,1], [1,0], [1,1]]

    Args:
        indices (torch.Tensor): A tensor of flat indices (L)
        shape (torch.Size): Target shape (*) where len(shape) = D

    Returns:
        torch.Tensor: The unraveled coordinates, (L, D).
    """
    
    shape = indices.new_tensor((*shape, 1))
    coefs = shape[1:].flipud().cumprod(dim=0).flipud()

    return torch.div(indices[..., None], coefs, rounding_mode='trunc') % shape[:-1]
