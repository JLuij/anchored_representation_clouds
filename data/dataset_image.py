from pprint import pprint
import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader, Subset

import random
from PIL import Image
from pathlib import Path

import utils_platform


## Define image transformations
from abc import ABC, abstractmethod
class ImageTransform(ABC):
    
    @abstractmethod
    def reverse(self, img_object):
        pass

class NormaliseFrom0to1(ImageTransform):
    def __call__(self, img_object):
        img_object["image"] = img_object["image"] / 255.
        return img_object

    def reverse(self, img_object):
        img_object["image"] = img_object["image"] * 255.
        return img_object

class Minus1ToMax1(ImageTransform):
    def __call__(self, image):
        assert image.min() <= 1.0 and image.max() >= 0.0, f'{image.aminmax()}, not in [0.0, 1.0]'
        return image * 2.0 - 1.0

    def reverse(self, image):
        return (image + 1.0) / 2.0

class Normalise(ImageTransform):
    def __call__(self, img_object):
        img = img_object['image']
        
        mean = img.mean(dim=[1,2], keepdim=True)
        std = img.std(dim=[1,2], keepdim=True)
        
        img_object['mean'] = mean
        img_object['std'] = std
        
        img_object['image'] = (img - mean) / std
        return img_object
    
    def reverse(self, img_object):
        img = img_object['image']
        img = (img * img_object['std']) + img_object['mean']
        img_object['image'] = img
        return img_object

class Lambda(ImageTransform):
    def __init__(self, lambd):
        if not callable(lambd):
            raise TypeError(f"Argument lambd should be callable, got {repr(type(lambd).__name__)}")
        self.lambd = lambd
    
    def __call__(self, img_object):
        img_object["image"] = self.lambd(img_object["image"])
        return img_object
    
    def reverse(self, img_object):
        NotImplementedError()

class NormaliseManual(ImageTransform):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image):
        return (image - self.mean) / self.std

    def reverse(self, image):
        return image * self.std + self.mean

## Define dataset class
class ImageDataset(Dataset):
    def __init__(
        self,
        root_dir,
        glob_string,
        label_extractor,
        image_transforms=[],
        include_if_path_lambda=None,
        load_img_as_rgb=True,
        verbose=True,
    ):
        """
        General dataset for images for the purpose of fitting INRs

        Args:
            root_dir (Path or string): _description_
            glob_string (string): A string like `train/**.jpg` or `*.png`. Note: NO leading slash.
            label_extractor (function(path)): Extract class label from path. Must return int.
            image_transforms (list, optional): Image transforms which WILL BE CACHED.
                Defaults to empty list i.e. no transforms.
            exclude_function (function, optional): Function that takes in path name
                and returns True to exclude, False to exclude. Defaults to None.
        """
        
        self.root_dir = Path(root_dir)
        self.label_extractor = label_extractor
        self.image_transforms = image_transforms
        self.load_img_as_rgb = load_img_as_rgb
        
        # Accept both 'platform_utilities.dataset_root / datasetdir' and 'datasetdir'
        try:
            self.root_dir.relative_to(utils_platform.dataset_root)
        except ValueError:
            self.root_dir = utils_platform.dataset_root / self.root_dir
        
        assert (self.root_dir.exists()), f"!! Error. Root dir {self.root_dir} does not exist"
        assert isinstance(self.image_transforms, list), "!! Error. Image transforms must be a list"
        
        self.file_paths = list(self.root_dir.glob(glob_string))
        assert len(self.file_paths), f"!! Error. Found no files in {str(root_dir)} glob string {glob_string}"
        
        if include_if_path_lambda is not None:
            print("Found include_function")
            prev_len = len(self.file_paths)
            self.file_paths = [x for x in self.file_paths if include_if_path_lambda(x)]
            print(f"include_function removed {prev_len - len(self.file_paths)} files, now {len(self.file_paths)}")
        
        if verbose:
            sample_img = self.__getitem__(0)["image"]
            print(f'Image sample: size {tuple(sample_img.shape)}, min {torch.min(sample_img)}, max {torch.max(sample_img)}')

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        image_path = self.file_paths[idx]
        
        image = Image.open(image_path)
        if self.load_img_as_rgb:
            image = image.convert("RGB")
        image = torchvision.transforms.functional.pil_to_tensor(image)
        
        label = self.label_extractor(image_path)
        assert isinstance(label, int), (
            f"!! Error. `label_extractor` must return type int but returned "
            f"{label} of type {type(label)} instead"
        )
        
        img_object = dict(
            path=self.file_paths[idx].as_posix(),
            image=image,
            label=label,
        )
        
        for transform in self.image_transforms:
            img_object = transform(img_object)
        
        return img_object



def GetDataloader(image_dataset, seed=0, fraction_or_num_elements=-1, dataset_fraction="1/1"):
    
    # We'll use dataset indices throughout
    indices = list(range(len(image_dataset)))
    
    # Shuffle the dataset by shuffling its indices
    if seed != -1:
        random.Random(seed).shuffle(indices)
    else:
        random.Random().shuffle(indices)
    
    # Optionally get a subset of the data first
    if fraction_or_num_elements != -1:
        if fraction_or_num_elements < 1.0:
            subset_length = int(fraction_or_num_elements * len(image_dataset))
        else:
            subset_length = int(fraction_or_num_elements)
        
        indices = indices[:subset_length]
        dataset_length = subset_length
        print(f"Getting subset {subset_length} of dataset")
    else:
        dataset_length = len(image_dataset)
    
    # Select the required subset of the dataset
    if dataset_fraction != "1/1":
        part, whole = dataset_fraction.split('/')
        part = int(part)
        whole = int(whole)
        assert part != 0, f'!! Error. `part` must be in [1, {whole}] but is {part}'
        assert part <= whole, f'!! Error. `part` must be in [1, {whole}] but is {part}'
        
        fraction_length = (1 / whole) * dataset_length
        start_idx = int((part-1) * fraction_length)
        end_idx = None if part == whole else int(part * fraction_length)
        
        indices = indices[start_idx : end_idx]
        print(f'Getting {dataset_fraction}={part}/{whole} of {dataset_length} so ' \
              f'{start_idx}:{end_idx} = {len(indices)} ({(len(indices) / dataset_length) * 100.0:.2f}%)')
    
    
    subset = Subset(image_dataset, indices)
    
    # Create the DataLoader
    dataloader = DataLoader(
        subset,
        num_workers=0,
        batch_size=1,
        collate_fn=lambda x: x[0]   # Remove list that the single img_object is contained in
    )
    
    return dataloader


def mnist_labelextractor(path):
    return int(path.parent.stem)

def fashionmnist_labelextractor(path):
    # 0	T-shirt/top
    # 1	Trouser
    # 2	Pullover
    # 3	Dress
    # 4	Coat
    # 5	Sandal
    # 6	Shirt
    # 7	Sneaker
    # 8	Bag
    # 9	Ankle boot
    return int(path.parent.stem)

def trafficsigns_labelextractor(path):
    # 0 empty
    # 1 50_sign
    # 2 70_sign
    # 3 80_sign
    label = int(path.stem.split('_')[-1])
    return label

def imagenette_labelextractor(path):
    labels_readable = dict(
        n01440764=0, # 'Tench',
        n02102040=1, # 'English springer',
        n02979186=2, # 'Cassette player',
        n03000684=3, # 'Chain saw',
        n03028079=4, # 'Church',
        n03394916=5, # 'French horn',
        n03417042=6, # 'Garbage truck',
        n03425413=7, # 'Gas pump',
        n03445777=8, # 'Golf ball',
        n03888257=9, # 'Parachute'
    )
    
    label = int(labels_readable[path.parent.stem])
    return label


def cifar_labelextractor(path):
    labels_dict = {
        'airplane' : 0,
        'automobile' : 1,
        'bird' : 2,
        'cat' : 3,
        'deer' : 4,
        'dog' : 5,
        'frog' : 6,
        'horse' : 7,
        'ship' : 8,
        'truck' : 9,
    }
    
    label = int(labels_dict[path.parent.stem])
    return label

mnist_settings = dict(
    glob_string = '**/*.png',
    label_extractor = mnist_labelextractor,
    load_img_as_rgb = False,
)

fashionmnist_settings = dict(
    glob_string = '**/*.png',
    label_extractor = fashionmnist_labelextractor,
    load_img_as_rgb = False,
)

div2k_settings = dict(
    glob_string = '**/*.png',
    label_extractor = lambda x: 0,
    load_img_as_rgb = True,
)
custom_settings = dict(
    glob_string = '**/*.png',
    label_extractor = lambda x: 0,
    load_img_as_rgb = True,
)

kodak_settings = dict(
    root_dir = utils_platform.dataset_root / 'kodak',
    glob_string = '**/*.png',
    label_extractor = lambda x: 0,
    load_img_as_rgb = True,
)

imagenette_settings = dict(
    root_dir = utils_platform.dataset_root / 'imagenette2-320/train',
    glob_string = '**/*.JPEG',
    label_extractor = imagenette_labelextractor,
)

cifar_settings = dict(
    glob_string = '**/*.png',
    label_extractor = cifar_labelextractor,
)

trafficsigns_settings = dict(
    glob_string = '*.jpg',
    label_extractor = trafficsigns_labelextractor,
)


def save_as_pil(img, save_path, unnormalise = True):
    img = img.squeeze().numpy()
    if unnormalise:
        img = img.clip(0, 1) * 255
    img = img.clip(0, 255).astype(np.uint8)

    img = Image.fromarray(img)
    img.save(save_path)
