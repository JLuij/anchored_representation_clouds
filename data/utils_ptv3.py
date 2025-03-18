import torch
import numpy as np

from pathlib import Path


## Utility functions
def batch_to_indiv(batched_attribute, batch_offset):
    original_offset = torch.diff(batch_offset, prepend=torch.tensor([0]).to(batch_offset.device))
    original_offset = original_offset.tolist()

    # Gives n tensors of different sizes
    splits = torch.split(batched_attribute, original_offset)
    return splits


## Transforms
class NormalisePerLatentDim:
    def __init__(self, statistics_path):
        statistics_path = Path(statistics_path)
        assert statistics_path.exists(), f"Statistics path does not exist {statistics_path}"

        stats = torch.load(statistics_path)
        self.mean_per_latent_dim = stats['mean_per_latent_dim']
        self.std_per_latent_dim = stats['std_per_latent_dim']

    def __call__(self, latent_field):
        latent_field["feat"] = (latent_field["feat"] - self.mean_per_latent_dim) / self.std_per_latent_dim
        return latent_field

class NormaliseOverWholeDataset:
    def __init__(self, statistics_path):
        statistics_path = Path(statistics_path)
        assert statistics_path.exists(), f"Statistics path does not exist {statistics_path}"

        stats = torch.load(statistics_path)
        self.mean = stats['mean']
        self.std = stats['std']

    def __call__(self, latent_field):
        latent_field["feat"] = (latent_field["feat"] - self.mean) / self.std
        return latent_field

class PointDropout:
    def __init__(self, drop_factor):
        assert drop_factor > 0 and drop_factor < 1
        self.num_points = (1 - drop_factor)

    def __call__(self, latent_field):
        num_latents = latent_field["feat"].shape[0]
        random_idx = torch.randperm(num_latents)[:int(self.num_points * num_latents)]
        
        latent_field["coord"] = latent_field["coord"][random_idx]
        latent_field["feat"] = latent_field["feat"][random_idx]
        
        return latent_field

class AddGaussianNoise:
    def __call__(self, latent_field):
        latent_field["feat"] = latent_field["feat"] + torch.randn_like(latent_field["feat"]) * 0.001
        return latent_field

class JumblePush:
    def __call__(self, latent_field):
        seed = abs(hash(latent_field["path"])) % (2**32)
        rng = np.random.default_rng(seed)
        
        # Generate random pushes deterministically
        push = rng.integers(-50, 51, size=latent_field["coord"][:, :2].shape)
        push = torch.from_numpy(push).to(latent_field["coord"].device)
        
        # Add the random push to the first two dimensions of "coord"
        latent_field["coord"][:, :2] += push
        
        return latent_field

class Jumble:
    def __call__(self, latent_field):
        seed = abs(hash(latent_field["path"])) % (2**32)
        rng = torch.Generator()
        rng.manual_seed(seed)
        
        indices = torch.randperm(latent_field["feat"].shape[0], generator=rng)
        latent_field["feat"] = latent_field["feat"][indices]
        return latent_field

class RandomPush:
    def __init__(self, magnitude=(-2, 2)):
        self.magnitude = magnitude

    def __call__(self, latent_field):
        # Generate random pushes for the first two dimensions only
        push = torch.empty_like(latent_field["coord"][:, :2]).uniform_(*self.magnitude)
        
        # Keep the last dimension unchanged
        latent_field["coord"][:, :2] += push
        return latent_field

class RandomRotation:
    def __init__(self, angle_range=(-180, 180)):
        # Generate a random angle within the given range in degrees
        self.angle_range = angle_range

    def __call__(self, latent_field):
        # Random angle in radians
        angle = torch.empty(1).uniform_(*self.angle_range) * (torch.pi / 180)

        # 2D rotation matrix, affecting only the first two dimensions
        rotation_matrix = torch.tensor([
            [torch.cos(angle), -torch.sin(angle)],
            [torch.sin(angle), torch.cos(angle)]
        ])

        # Apply rotation to the first two dimensions only
        latent_field["coord"][:, :2] = latent_field["coord"][:, :2] @ rotation_matrix
        return latent_field

class RandomFlip:
    def __init__(self, flip_x=True, flip_y=True):
        self.flip_x = flip_x
        self.flip_y = flip_y

    def __call__(self, latent_field):
        # Generate random flips based on the chosen axes
        if self.flip_x and torch.rand(1).item() > 0.5:
            latent_field["coord"][:, 0] *= -1  # Flip x-axis
        if self.flip_y and torch.rand(1).item() > 0.5:
            latent_field["coord"][:, 1] *= -1  # Flip y-axis
        return latent_field
