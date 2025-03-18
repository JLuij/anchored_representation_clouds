import torch
import torch.nn as nn

from sklearn.neighbors import NearestNeighbors

from functools import partial


class ArcRelativePosEncoder(nn.Module):
    def __init__(
        self,
        # Net parameters
        num_latents,
        latent_dim,
        num_neighbours,
        
        latent_init_distribution=partial(nn.init.uniform_, a=-1e-4, b=1e-4),
        
        # Interpolation options
        dist_function=lambda x: x,
        normalise_relative_pos=True,
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.num_latents = num_latents
        self.num_neighs = num_neighbours
        
        self.dist_function = dist_function
        self.normalise_relative_pos = normalise_relative_pos

        self.latents_pos = torch.nn.Embedding(self.num_latents, 2).requires_grad_(False)
        self.latents = torch.nn.Embedding(self.num_latents, latent_dim)

        # Initialise latent vectors
        with torch.no_grad():
            self.latent_init_distribution = latent_init_distribution
            latent_init_distribution(self.latents.weight.data)

    def forward(self, idx_flat):
        # idx_flat: [num_pixels]
        
        latent_vecs = self.latents(self.idxcache[idx_flat])     # [num_pixels, num_neighs, latent_dim]
        relative_pos = self.relpos_cache[idx_flat].cuda()       # [num_pixels, num_neighs, 2]

        # [num_pixels, latent_dim*num_neighs + 2*num_neighs]
        out = torch.cat([latent_vecs.flatten(start_dim=1), relative_pos.flatten(start_dim=1)], dim=1) 

        return out

    def get_NN_information(self, integer_grid_np):
        
        with torch.no_grad():
            latents_pos = self.latents_pos.weight.cpu()
            
            nearest_neighbours = NearestNeighbors(n_neighbors=self.num_neighs)
            nearest_neighbours.fit(latents_pos)
            # Returns for each neighbour the distance to it and its index into x_np
            _, indices = nearest_neighbours.kneighbors(integer_grid_np)

            relpos_cache = (
                latents_pos[indices].squeeze() - integer_grid_np[:, None, :]
            ).float()
            
            relpos_cache = self.dist_function(relpos_cache)
            
            # Normalise relpos_cache to zero mean, 1 std
            if self.normalise_relative_pos:
                mean = relpos_cache.view(-1).mean()
                std = relpos_cache.view(-1).std()
                relpos_cache = (relpos_cache - mean) / std
            
            # Transfer to CUDA
            self.relpos_cache = relpos_cache.cuda()
            self.idxcache = torch.from_numpy(indices).to("cuda", torch.long)
