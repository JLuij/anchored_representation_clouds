import torch
import matplotlib.pyplot as plt

import utils_platform

def legend_to_side():
    plt.legend(bbox_to_anchor=(1.0, 1.0), loc='upper left')

def show_image(img_tensor, title=None, print_debug_info=False, figsize=(8, 6)):
    import torch
    import numpy as np
    import matplotlib.pyplot as plt
    
    if isinstance(img_tensor, np.ndarray):
        img_tensor = torch.from_numpy(img_tensor)
    
    if len(img_tensor.shape) == 4:
        if img_tensor.shape[0] == 1:
            # Remove batch dimension
            img_tensor.squeeze_(0)
        else:
            ValueError(f"Detected > 1 batch ({img_tensor.shape}), can't show image")
    
    if len(img_tensor.shape) == 2:
        # Add single dimension to this grayscale image to act as channel
        img_tensor.unsqueeze_(0)
    
    if print_debug_info:
        print(img_tensor.shape, img_tensor.dtype, img_tensor.device,
              'min:', img_tensor.min(), 'max', img_tensor.max())
    
    plt.figure(figsize=figsize)
    if title is not None: plt.title(title)
    plt.imshow(img_tensor.permute(1,2,0), interpolation='none')
    plt.show()


def visualise_latents_pos(latents_pos, img_size):
    vis = torch.zeros(img_size)
    
    vis[latents_pos[:, 0], latents_pos[:, 1]] = 1
    
    plt.title("Latents positions visualisation")
    plt.imshow(vis, interpolation='none')
    plt.show()

def visualise_latents_pretty(latents_pos, image=None, image_size=None, figsize=(5,5), hide_axes=True):
    if image is None:
        assert image_size is not None, "If no image is provided, give an image_size"
    else:
        assert image_size is None, "If image is provided, no image_size has to be provided"
    
    # Set up figure and remove most padding
    plt.figure(figsize=figsize)
    if hide_axes:
        plt.gca().tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)
    
    ## First, plot image
    if image is not None:
        plt.imshow(image.permute(1,2,0), cmap='gray', interpolation='none')
    else:
        # If no image is shown, we need to swap axes else scatter will be shown
        #   upside down. Also, limits must be set, else padding will be added to
        #   the plot.
        plt.gca().set_xlim(0, image_size[1])
        plt.gca().set_ylim(image_size[0], 0)
        
    
    ## Second, plot latents position
    linewidth = 0.5
    s = 1 * figsize[0]
    
    plt.scatter(
        latents_pos[:, 1], latents_pos[:, 0],
        marker='D',
        c='#FCD709',
        edgecolor='black',
        s=s,
        linewidth=linewidth,
    )
    plt.show()


def format_size(num, suffix="B"):
    for unit in ("", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"):
        if abs(num) < 1024.0:
            return f"{num:3.1f}{unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f}Yi{suffix}"


if not utils_platform.on_cluster:
    def display_animation(images, fps=10):
        import matplotlib.pyplot as plt
        from IPython.display import HTML
        from matplotlib.animation import FuncAnimation
        
        """
        Display an animation of a list of equal-sized images.

        Parameters:
            images (list of plotable arrays (pytorch, numpy)): List of images to display.
            fps (int): Frames per second for the animation.

        Returns:
            animation: Matplotlib animation object.
        """
        fig, ax = plt.subplots()
        ax.set_axis_off()
        ax.set_aspect('equal')

        im = ax.imshow(images[0], cmap='gray')

        def update(frame):
            im.set_array(images[frame])
            return im,

        animation = FuncAnimation(fig, update, frames=len(images), interval=1000 / fps)

        plt.close(fig)  # Prevent duplicate display in Jupyter notebook
        return HTML(animation.to_jshtml())
