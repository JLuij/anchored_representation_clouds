import torch
import torchvision.transforms.functional as TF
import numpy as np

from PIL import Image

import random

class MNISTinCanvas(object):
    def __init__(self, canvas_size, centered=False):
        """
        Place MNIST either centered or in a random position in a canvas of zeros

        Args:
            canvas_size (array_like): Size of the resulting padded MNIST
            centered (bool, optional): If True, center digit. Else place it randomly. Defaults to False.
        """        
        self.canvas_size = torch.tensor(canvas_size)
        self.centered = centered
    
    def __call__(self, image):
        
        # Canvas of size (channels, width, height)
        canvas = torch.zeros((1, self.canvas_size[0], self.canvas_size[1]))
        
        mnist_height, mnist_width = 28, 28
        if self.centered:
            center = (self.canvas_size - 28) // 2
            x, y = center   
        else:
            x, y = np.random.randint(0, high=(self.canvas_size[0] - mnist_width,
                                              self.canvas_size[1] - mnist_height))
        
        canvas[:, x:x + mnist_width, y:y + mnist_height] = image
        
        return canvas

class PlaceMnistOnBackground:
    def __init__(self, centered = False):
        self.centered = centered

    def __call__(self, mnist_image, background):
        # Create a mask where the MNIST digit is not background
        background_threshold = 0.3
        mask = (mnist_image > background_threshold)

        # Determine the placement position
        height, width = mnist_image.shape[1:]
        bg_height, bg_width = background.shape[1:]

        if self.centered:
            x = (bg_width - width) // 2
            y = (bg_height - height) // 2
        else:
            y, x = np.random.randint(
                0, high=(bg_height - height, bg_width - width)
            )

        # Expand MNIST image to RGB
        mnist_image = torch.cat([mnist_image] * 3, dim=0)
        mask = torch.cat([mask] * 3, dim=0)

        # Overlay MNIST digit on background
        background[:, y:y + height, x:x + width][mask] = mnist_image[mask]

        return background
        

class PlaceMnistOnBackgroundTightly:
    def __init__(self, background_image_paths:list, centered:bool=False, scale_mnist_factor=-1):
        self.background_image_paths = background_image_paths
        self.centered = centered
        self.scale_mnist_factor = scale_mnist_factor
    
    def __call__(self, mnist_image):
        assert mnist_image.max() < 1.01 and mnist_image.min() > -0.01      # Make sure image is normalised to [0, 1]
        # Get random background
        background_path = random.choice(self.background_image_paths)
        background = TF.pil_to_tensor(Image.open(background_path)) / 255.
        
        # Create a mask where MNIST image is non-zero
        background_threshold = 0.3
        
        if self.scale_mnist_factor != -1:
            mnist_image = TF.resize(mnist_image, torch.tensor(mnist_image.shape[1:]) * self.scale_mnist_factor, interpolation=Image.NEAREST)
        mask = (mnist_image > background_threshold)
        
        # Crop MNIST to obtain a tight bounding box around the digit
        digit_indices = torch.nonzero(mask, as_tuple=False)
        min_corner = torch.min(digit_indices, dim=0).values
        max_corner = torch.max(digit_indices, dim=0).values
        mnist_image = mnist_image[:, min_corner[1]:max_corner[1], min_corner[2]:max_corner[2]]
        
        # Recompute the mask as the image size has now changed
        mask = (mnist_image > background_threshold)
        
        # Compute the random position
        padding = 1
        height = max_corner[1] - min_corner[1]
        width = max_corner[2] - min_corner[2]
        bg_shape = background.shape[1:]
        if self.centered:
            canvas_size = torch.tensor([bg_shape[0], bg_shape[1]])
            center = (canvas_size - 28) // 2
            x, y = center
        else:
            y, x = np.random.randint(padding, high=(bg_shape[0] - height -  padding,
                                                    bg_shape[1] - width - padding))
        
        # Make RGB image from mnist digit
        mnist_image = torch.concat([mnist_image] * 3, dim=0)
        mask = torch.concat([mask] * 3, dim=0)
        
        background[:, y:y+height, x:x+width][mask] = mnist_image[mask]
        
        return background

    # Use as follows:
    # # background_paths = list((platform_utilities.dataset_root / 'cifar10').glob('**/*.png'))
    # background_paths = list((platform_utilities.dataset_root / 'DIV2K_train_HR').glob('**/*.png'))
    # tr = PlaceMnistOnBackgroundTightly(background_paths, False)
    # img = TF.pil_to_tensor(Image.open(platform_utilities.dataset_root / 'MNIST_pngs/train/2/train_2_0.png')) / 255.
    # result = tr(img)
