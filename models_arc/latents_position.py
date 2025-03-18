import torch
import kornia

import logging

import data.utils


def sample_positions_from_gradient(signal, num_points):
    # signal [C, *spatial_dims] e.g. [3, H, W]
    spatial_dims = torch.tensor(signal.shape[1:])
    dimensionality = len(spatial_dims)
    
    if dimensionality == 0:
        raise ValueError(f"Expected signal to be at least shape [C, H], not {signal.shape}")
    if dimensionality == 1:
        # Signal shape is [C, H]
        # Transform to [C, 1, H]
        signal = signal.unsqueeze(1)
    
    
    with torch.no_grad():
        ## Compute gradient
        # [B,C,H,W] -> [B,C,2,H,W]
        signal_gradient = kornia.filters.spatial_gradient(signal.unsqueeze(0).float())
        # [H,W], gradient magnitude of all channels
        signal_gradient = (signal_gradient ** 2).sum(dim=[1,2]).sqrt().squeeze(0)
        
        ## Determine how many points to sample
        if num_points < 1:
            # Sample fraction of image's pixels
            num_points_discrete = int(num_points * torch.prod(spatial_dims))
            
            # There's not enough gradient content to sample num_points_discrete points
            if int(torch.count_nonzero(signal_gradient)) < num_points_discrete:
                logging.warning(f'Image gradient is smaller than {num_points} of pixels. So taking {int(torch.count_nonzero(signal_gradient))} \
                        instead of {num_points_discrete} samples.')
                num_points_discrete = int(torch.count_nonzero(signal_gradient))
        else:
            num_points_discrete = num_points
        
        logging.debug(f"Gradient nonzero size {int(torch.count_nonzero(signal_gradient))}, {num_points=}, {spatial_dims=}, {num_points_discrete=}")
        
        ## Sample gradient
        signal_gradient_flat = signal_gradient.flatten()
        samples = torch.multinomial(signal_gradient_flat, num_points_discrete, replacement=False)
        # Convert the flat indices back to 2D indices
        coords_tensor = data.utils.unravel_index(samples, signal_gradient.shape)
        
        return coords_tensor, signal_gradient
