## Imports for config
from dataclasses import dataclass
from enum import Enum, auto
import inspect
import traceback
from typing import Tuple, List, Union
from functools import partial

## Imports for instantiation
import torch
import torch.nn as nn
import numpy as np

import utils_platform
import models_arc.arc as arc
import models_arc.distance_transforms as distance_transforms
import models_arc.samplers as samplers
import models_arc.latents_position as latents_position
import data.utils_mnist
import models_baselines.mlp
import data.dataset_image as ImageData


## Base config class
class ConfigField:
    def parse(self):
        raise NotImplementedError()
    
    
    def __repr__(self):
        # Overwrite otherwise this returns `<value: value_int>`
        return str(self)

# Enum for INR type with instantiation logic
class InrTypeEnum(ConfigField, Enum):
    RELPOS = auto()

    def parse(self):
        if self == self.RELPOS:
            return arc.ArcRelativePosEncoder
        else:
            raise ValueError()

class DistFuncEnum(ConfigField, Enum):
    IDENTITY = auto()
    
    def parse(self):
        if self == self.IDENTITY:
            return distance_transforms.dists_identity
        else:
            raise ValueError()

class LatentInitEnum(ConfigField, Enum):
    UNIFORM = auto()
    
    def parse(self, args):
        if self == self.UNIFORM:
            return partial(nn.init.uniform_, **args)
        else:
            raise ValueError()

class HarmonicsMethodEnum(ConfigField, Enum):
    EXP = auto()
    LINEAR = auto()
    LOG = auto()
    NONE = auto()
    
    def parse(self, args):
        if self == self.EXP:
            harms = torch.linspace(np.log2(args['start']), np.log2(args['end']), args['latent_dim'])
            return torch.exp2(harms)
        elif self == self.LINEAR:
            raise NotImplementedError()
            # return torch.linspace(args['start'], args['end'], args['latent_dim'])
        elif self == self.LOG:
            raise NotImplementedError()
            # return torch.linspace(args['start'], args['end'], args['latent_dim'])
        elif self == self.NONE:
            return torch.ones(args['latent_dim'])
            # raise NotImplementedError()
        else:
            raise ValueError()

class HarmonicsFuncEnum(ConfigField, Enum):
    SIN = auto()
    COS = auto()
    NONE = auto()
    
    def parse(self) -> callable:
        if self == self.SIN:
            return torch.sin
        elif self == self.COS:
            return torch.cos
        elif self == self.NONE:
            return lambda x: x
        else:
            raise ValueError()
        

class LatentPosInitEnum(ConfigField, Enum):
    GRADIENT = auto()
    
    def parse(self, args):
        if self == self.GRADIENT:
            return partial(latents_position.sample_positions_from_gradient, **args)
        else:
            raise ValueError()


class DecTypeEnum(ConfigField, Enum):
    RELU = auto()

    def parse(self):
        if self == self.RELU:
            return models_baselines.mlp.MLP
        else:
            raise ValueError()

class DatasetEnum(ConfigField, Enum):
    MNIST = auto()
    IMAGENETTE_FULLRES = auto()
    IMAGENETTE_CONSTANT = auto()
    CIFAR = auto()
    FASHIONMNIST = auto()
    DIV2K = auto()
    CUSTOM = auto()
    
    def parse(self, args):
        # args: contains root_dir
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
        elif self == self.CUSTOM:
            return partial(ImageData.ImageDataset, **ImageData.custom_settings, **args)
        else:
            raise ValueError()

class TransformEnum(ConfigField, Enum):
    MNISTINCANVAS = auto()
    MIN1TO1 = auto()
    NORMALISE = auto()
    NORMALISE_MANUALLY = auto()
    
    def parse(self, args):
        if self == self.MNISTINCANVAS:
            return data.utils_mnist.MNISTinCanvas(**args)
        elif self == self.MIN1TO1:
            return ImageData.Minus1ToMax1(**args)
        elif self == self.NORMALISE_MANUALLY:
            return ImageData.NormaliseManual(**args)
        elif self == self.NORMALISE:
            return ImageData.NormaliseOverSelf(**args)
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
class Config:
    alias: str
    
    inr_seed: int
    dec_seed: int
    
    inr_type: InrTypeEnum
    relpos_normalise: Union[bool, None]
    latent_dim: int
    num_latents: float
    num_neighbours: int
    dist_function: DistFuncEnum
    latent_init_distr: Tuple[LatentInitEnum, dict]
    harmonics_method: Tuple[HarmonicsMethodEnum, dict]
    harmonics_function: HarmonicsFuncEnum
    latent_position_init: Tuple[LatentPosInitEnum, dict]
    
    dec_share: bool
    dec_shared_at_step: int
    dec_type: DecTypeEnum
    dec_layers: list
    out_dim: int
    
    img_dataset: Tuple[DatasetEnum, dict]
    img_signal_transforms: List[Tuple[TransformEnum, dict]]
    coord_subsampler: Tuple[SampleEnum, dict]
    
    num_steps: int
    save_intermittently_at: List[int]
    dec_optimiser: Tuple[OptimEnum, dict]
    enc_optimiser: Tuple[OptimEnum, dict]
    
    num_repeats: int
    
    def parse(self, verbose=True):
        ## Check relative position compatibility
        if self.inr_type == InrTypeEnum.RELPOS:
            assert self.relpos_normalise is not None
        if self.relpos_normalise is not None:
            assert self.inr_type == InrTypeEnum.RELPOS or self.inr_type == InrTypeEnum._RELPOS_FUNC
        
        dataset_path = utils_platform.dataset_root / self.img_dataset[1]['root_dir']
        assert dataset_path.exists(), f"Dataset path {dataset_path} does not exist!"
        
        dataset_path = utils_platform.dataset_root / self.img_dataset[1]['root_dir']
        assert dataset_path.exists(), f"Dataset path {dataset_path} does not exist!"
        
        # Append extra required information
        self.harmonics_method[1]['latent_dim'] = self.latent_dim
        self.latent_position_init[1]['num_points'] = self.num_latents
        
        if verbose: print("Parsing config")
        
        settings = {}
        # For each parameter, call its parse method with any additional arguments
        for param_name, param_value in self.__dict__.items():
            try:
                if isinstance(param_value, ConfigField):
                    settings[param_name] = param_value.parse()
                elif isinstance(param_value, tuple):
                    param_value, args = param_value
                    assert isinstance(param_value, ConfigField)
                    settings[param_name] = param_value.parse(args)
                elif isinstance(param_value, (bool, str, int, float, type(None))):
                    settings[param_name] = param_value
                elif isinstance(param_value, list):
                    if verbose: print(f"Param {param_name} requires extra parsing")
                else:
                    raise ValueError(f"Got `{param_value}` (type {type(param_value)} for {param_name})")
            except Exception as e:
                print(f"!! Error on {param_name} ({param_value})", e)
                traceback.print_exc()
        
        ## Parse some lists
        layers = []
        for layer in self.dec_layers:
            if layer == 'in':
                if self.inr_type == InrTypeEnum.RELPOS:
                    layers.append(self.num_neighbours * (self.latent_dim + 2))
                else:
                    layers.append(self.latent_dim)
            
            elif layer == 'out':
                layers.append(self.out_dim)
            
            else:
                if self.inr_type == InrTypeEnum.RELPOS:
                    layers.append(int(layer * self.num_neighbours * (self.latent_dim + 2)))
                else:
                    layers.append(int(layer * self.latent_dim))
        
        # for transform, args in self.img_signal_transforms:
        #     transforms.append(transform.parse(args))
        transforms = []
        transforms.insert(0, ImageData.NormaliseFrom0to1())
        settings['img_signal_transforms'] = transforms
        
        ## Group some parameters
        settings['enc_arguments'] = dict(
            # num_latents = self.num_latents,   # Will be provided per INR
            latent_dim = self.latent_dim,
            num_neighbours = self.num_neighbours,
            
            latent_init_distribution = settings['latent_init_distr'],
            dist_function = settings['dist_function'],
            # harmonics = settings['harmonics_method'],
            # harm_function = settings['harmonics_function'],
        )
        
        settings['dec_arguments'] = dict(
            layers = layers
        )
        
        settings['img_dataset'] = settings['img_dataset'](
            image_transforms=settings['img_signal_transforms'], verbose=verbose
        )
        # settings['img_dataset'] = settings['img_dataset']()
        
        settings['save_intermittently_at'] = self.save_intermittently_at
        
        return settings
    
    def get_save_path(self, save_dir=None):
        if save_dir is None:
            save_dir = utils_platform.code_root / "configs/temp_configs"
        save_path = save_dir / f"{self.alias}.py"
        return save_path
        
    def save(self, save_dir=None):
        body = """
                from configs.placeholder_config_core import *
                
                config = 
            """
        
        body = inspect.cleandoc(body)
        body += self.__repr__().replace(',', ',\n')
        
        save_path = self.get_save_path(save_dir)
        
        with open(save_path, 'w') as f:
            f.write(body + '\n')
        print(f"Saved config to {save_path}")


def load_config(config_path) -> Config:
    import importlib.util
    import sys
    
    spec = importlib.util.spec_from_file_location("config_module", config_path)
    config_module = importlib.util.module_from_spec(spec)
    sys.modules["config_module"] = config_module
    spec.loader.exec_module(config_module)
    return config_module.config