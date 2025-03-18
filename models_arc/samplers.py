import torch
import torch.nn as nn

def full_forward_pass(model, img_shape):
    with torch.no_grad():
        spatial_dims = torch.tensor(img_shape[1:])
        num_pixels = torch.prod(spatial_dims)
        
        # Flat tensor of size [number_of_pixels, C] where C=3 is RGB, C=1 is grayscale
        full_pred_flat = torch.zeros(img_shape).view(-1, img_shape[0])
        
        max_num_coords_per_pass = int(1024**2)    # The number of pixels in a 1024x1024 image
        num_passes = (num_pixels // max_num_coords_per_pass) + 1
        
        for i in range(num_passes):
            end_idx = min((i+1) * max_num_coords_per_pass, num_pixels)
            idx_flat = torch.arange(start = i * max_num_coords_per_pass, end = end_idx, device='cuda')
            
            colour = model['decoder'](model['encoder'](idx_flat))
            full_pred_flat[idx_flat] = colour.cpu()
        full_pred_flat = full_pred_flat.permute(1,0)
        
        return full_pred_flat.view(img_shape)


class Sampler(object):
    def __init__(self):
        raise NotImplementedError()
    
    def __call__(self):
        raise NotImplementedError()

class SampleRandomSubset(nn.Module, Sampler):
    def __init__(self, image_object, num_samples, pooled=True):
        super().__init__()
        self.image_object = image_object
        
        image = self.image_object['image']
        
        # self.register_buffer('image', image)
        self.register_buffer('image_flat', image.flatten(start_dim=1))
        self.num_samples = num_samples
        self.num_pixels = self.image_flat.shape[1]
        
        if self.num_samples < 1:
            self.num_samples_discrete = int(num_samples * self.num_pixels)
        else:
            self.num_samples_discrete = num_samples
        
        self.pooled = pooled
        if self.pooled:
            self.step = 0
            self.precomputed_idx = torch.split(torch.randperm(self.num_pixels, device='cuda'), self.num_samples_discrete)
            self.num_splits = len(self.precomputed_idx)
    
    def __call__(self):
        if self.pooled:
            idx = self.precomputed_idx[self.step % self.num_splits]
            self.step += 1
        else:
            idx = torch.randperm(self.num_pixels, device=self.image_flat.device)[:self.num_samples_discrete]
        
        image_subset = self.image_flat[:, idx]
        
        return image_subset, idx


class SampleAll(nn.Module, Sampler):
    def __init__(self, image_object):
        super().__init__()
        self.image_object = image_object
        
        image = self.image_object['image']
        
        # self.register_buffer('image', image)
        self.register_buffer('image_flat', image.flatten(start_dim=1))
        
        self.num_pixels = self.image_flat.shape[1]
        self.idx = torch.arange(self.num_pixels)
    
    def __call__(self):
        self.idx = self.idx.to(self.image_flat.device)      # Put idx on cuda for sped up subsequent calls
        return self.image_flat, self.idx
