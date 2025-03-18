## Imports for config
from dataclasses import dataclass
from enum import Enum, auto
import inspect
from typing import Any, Dict, Tuple, List, Union
from functools import partial

## Imports for instantiation
import torch

import utils_platform
import data.dataset_image as ImageData
import data.utils_mnist

import models_arc.samplers as samplers
from models_baselines.siren_correct import Siren
from models_baselines.finer import Finer
from models_baselines.siren_dwsnets import SirenDws

## Base config class
class ConfigField:
    def parse(self):
        raise NotImplementedError()
    
    
    def __repr__(self):
        # Overwrite otherwise this returns `<value: value_int>`
        return str(self)

# Enum for INR type with instantiation logic
class SirenTypeEnum(ConfigField, Enum):
    STANDARD = auto()
    FINER = auto()
    DWSNETS = auto()

    def parse(self, args):
        if self == self.STANDARD:
            return partial(Siren, **args)
        if self == self.FINER:
            return partial(Finer, **args)
        elif self == self.DWSNETS:
            return partial(SirenDws, **args)
        else:
            raise ValueError()


class DatasetEnum(ConfigField, Enum):
    MNIST = auto()
    IMAGENETTE_FULLRES = auto()
    IMAGENETTE_CONSTANT = auto()
    CIFAR = auto()
    FASHIONMNIST = auto()
    DIV2K = auto()
    KODAK = auto()
    NONE = auto()
    
    def parse(self, args):
        if self == self.MNIST:
            return partial(ImageData.ImageDataset, **ImageData.mnist_settings, **args)
        elif self == self.IMAGENETTE_FULLRES:
            return partial(ImageData.ImageDataset, **ImageData.imagenette_settings, **args)
        elif self == self.IMAGENETTE_CONSTANT:
            return partial(ImageData.ImageDataset, **ImageData.imagenette_settings, **args)
        elif self == self.CIFAR:
            return partial(ImageData.ImageDataset, **ImageData.cifar_settings, **args)
        elif self == self.FASHIONMNIST:
            return partial(ImageData.ImageDataset, **ImageData.fashionmnist_settings, **args)
        elif self == self.DIV2K:
            return partial(ImageData.ImageDataset, **ImageData.div2k_settings, **args)
        elif self == self.KODAK:
            return partial(ImageData.ImageDataset, **ImageData.kodak_settings, **args)
        elif self == self.NONE:
            return None
        else:
            raise ValueError()


class TransformEnum(ConfigField, Enum):
    MNISTINCANVAS = auto()
    MIN1TO1 = auto()
    NORMALISE = auto()
    NORMALISE_MANUALLY = auto()
    ONBACKGROUND = auto()
    
    def parse(self, args):
        if self == self.MNISTINCANVAS:
            return data.utils_mnist.MNISTinCanvas(**args)
        elif self == self.MIN1TO1:
            return ImageData.Minus1ToMax1(**args)
        elif self == self.NORMALISE_MANUALLY:
            return ImageData.NormaliseManual(**args)
        elif self == self.NORMALISE:
            return ImageData.NormaliseOverSelf(**args)
        elif self == self.ONBACKGROUND:
            raise NotImplementedError()
        else:
            raise ValueError()


class SampleEnum(ConfigField, Enum):
    SUB = auto()
    ALL = auto()
    
    def parse(self, args):
        if self == self.SUB:
            return partial(samplers.SampleRandomSubset, **args)
        elif self == self.ALL:
            return partial(samplers.SampleAll, **args)
        else:
            raise ValueError()

class OptimEnum(ConfigField, Enum):
    ADAM = auto()
    
    def parse(self, args):
        if self == self.ADAM:
            return partial(torch.optim.Adam, **args)
        else:
            raise ValueError()

@dataclass
class SirenConfig:
    alias: str
    
    # e.g. SirenTypeEnum.WEIGHTSCALE with {alpha=1.0}
    siren: Tuple[SirenTypeEnum, dict]
    
    # If last dim is 'out', out_dim is determined by dataset used.
    # Else if it's int, the int will be used.
    # If 'in' is used, it will be replaced by 2
    layers: list # e.g. ['in', 32, 32, 'out'] or ['in', 32, 1]
    share_init: bool
    weight_scaling_alpha: float
    
    img_dataset: Tuple[DatasetEnum, dict]
    img_signal_transforms: List[Tuple[TransformEnum, dict]]
    coord_subsampler: Tuple[SampleEnum, dict]
    
    num_steps: int
    save_intermittently_at: List[int]
    optimiser: Tuple[OptimEnum, dict]
    early_stop_psnr: int
    
    # def __post_init__(self):
    #     print('Validating config')
        
        # if self.siren_type == SirenTypeEnum.WEIGHTSCALE:
        #     assert self.weight_scaling_alpha is not None
        # if self.weight_scaling_alpha is not None:
        #     assert self.siren_type == SirenTypeEnum.WEIGHTSCALE
    
    def parse(self, verbose=True):
        if verbose: print("Parsing config")
        
        settings = {}
        # For each parameter, call its parse method with any additional arguments
        for param_name, param_value in self.__dict__.items():
            if isinstance(param_value, ConfigField):
                settings[param_name] = param_value.parse()
            elif isinstance(param_value, tuple):
                param_value, args = param_value
                assert isinstance(param_value, ConfigField), f"{param_value=}"
                settings[param_name] = param_value.parse(args)
            elif isinstance(param_value, (bool, str, int, float, type(None))):
                settings[param_name] = param_value
            elif isinstance(param_value, list):
                if verbose: print(f"Param {param_name} may require extra parsing")
            else:
                raise ValueError(f"Got `{param_value}` (type {type(param_value)} for {param_name})")
        
        
        transforms = []
        for transform, args in self.img_signal_transforms:
            transforms.append(transform.parse(args))
        transforms.insert(0, ImageData.NormaliseFrom0to1())
        
        if self.img_dataset[0] is not DatasetEnum.NONE:
            settings['img_signal_transforms'] = transforms
            settings['img_dataset'] = settings['img_dataset'](
                image_transforms=settings['img_signal_transforms'], verbose=verbose
            )
        
        layers = []
        for idx, layer in enumerate(self.layers):
            if idx == 0:
                if layer == 'in':
                    layers.append(2)
                else:
                    layers.append(layer)
            
            if idx > 0 and idx < (len(self.layers) - 1):
                layers.append(layer)
            
            if idx > 0 and idx == len(self.layers) - 1:
                if layer == 'out':
                    if self.img_dataset[0] is DatasetEnum.NONE:
                        print("!! You must set out_dim yourself")
                    else:
                        img_channels = settings['img_dataset'].__getitem__(0)['image'].shape[0]
                        layers.append(img_channels)
                else:
                    layers.append(layer)
        settings['layers'] = layers
        
        settings['save_intermittently_at'] = self.save_intermittently_at
        
        return settings
    
    def get_save_path(self, save_dir=None):
        if save_dir is None:
            save_dir = utils_platform.code_root / "models_baseline/configs"
        save_path = save_dir / f"{self.alias}.py"
        return save_path
        
    def save(self, save_dir=None):
        body = """
                from models_baseline.configs.siren_config_core import *
                
                config = 
            """
        
        body = inspect.cleandoc(body)
        body += self.__repr__().replace(',', ',\n')
        
        save_path = self.get_save_path(save_dir)
        
        with open(save_path, 'w') as f:
            f.write(body + '\n')
        print(f"Saved config to {save_path}")

if __name__ == "__main__":
    
    aconfiginstance = SirenConfig(
        alias = 'a_fmnist1024centered_dws_sample25_norm01',
        siren = (SirenTypeEnum.DWSNETS, dict()),
        # siren = (SirenTypeEnum.WEIGHTSCALE, dict(alpha=2.0)),
        layers = ['in', 32, 32, 'out'],
        share_init = True,
        weight_scaling_alpha = None,
        
        img_dataset = DatasetEnum.FASHIONMNIST,
        img_signal_transforms = [
            (TransformEnum.MNISTINCANVAS, dict(canvas_size=(1024, 1024), centered=True)),
            # (TransformEnum.MIN1TO1, dict()),
        ],
        # coord_subsampler = (SampleEnum.ALL, dict()),
        coord_subsampler = (SampleEnum.SUB, dict(num_samples=0.25, pooled=False)),
        
        num_steps = 1_000,
        save_intermittently_at = [500],
        optimiser = (OptimEnum.ADAM, dict(lr=5e-4)),
        early_stop_psnr = 55,
    )
    
    aconfiginstance.save()
    