import torch
import torch.nn.functional as F
import numpy as np

import logging
import copy
from pathlib import Path

from torch.utils.data import Dataset

def label_extractor_default(path, num_classes = 10):
    label_int = torch.tensor(int(path.stem.split('_')[1]))
    
    return F.one_hot(label_int, num_classes=num_classes).float()

def label_extractor_imagenette(path, num_classes = 10):
    label_int = torch.tensor(int(path.stem.split('_')[-2])) 
    
    return F.one_hot(label_int, num_classes=num_classes).float()

def label_extractor_cifar(path, num_classes = 10):
    label_int = torch.tensor(int(path.stem.split('_')[2])) 
    
    return F.one_hot(label_int, num_classes=num_classes).float()


def get_arc_files(start_dir):
    dataset_files = dict(train=[], val=[], test=[])
    
    for split in dataset_files.keys():
        subdir = start_dir / split
        if subdir.exists():
            dataset_files[split] = list(subdir.glob("**/*.pt"))
        else:
            # Otherwise, fall back to prefixed files in the root directory
            dataset_files[split] = list(start_dir.glob(f"{split}_*.pt"))

    return dataset_files

class ArcDataset(Dataset):
    def __init__(
        self,
        root_dir,
        shared_decoder_path=None,
        train=True,
        grid_size=0.01,
        transform=[],
        use_cache=True,
        label_extractor=label_extractor_default,
        include_if_path_lambda=lambda x: 'test' not in x.stem,
        file_paths=None,
    ):
        self.root_dir = Path(root_dir)
        self.shared_decoder_path = shared_decoder_path
        self.grid_size = grid_size
        self.transform = transform
        self.label_extractor = label_extractor
        if isinstance(self.label_extractor, tuple):
            self.label_extractor = self.label_extractor[0]
        self.include_if_path_lambda = include_if_path_lambda
        self.train = train
        
        assert self.root_dir.exists(), f"!! Error. Path {self.root_dir} does not exist!"
        assert isinstance(self.transform, list), "!! Error. Image transforms must be a list"
        
        if file_paths is not None:
            logging.info("Got file_paths")
            self.file_paths = file_paths
        else:
            file_paths = list(self.root_dir.glob(f"*.pt"))
            self.file_paths = file_paths
            assert len(self.file_paths) > 0, f"Found no files in {self.root_dir}"
        
        logging.debug(f"Prefiltering, found {len(file_paths)} files")
        if self.include_if_path_lambda is not None:
            logging.debug("Found include filter")
            prev_len = len(self.file_paths)
            self.file_paths = [x for x in self.file_paths if self.include_if_path_lambda(x)]
            logging.info(f"Filtered out {prev_len - len(self.file_paths)} files, now {len(self.file_paths)}")
        
        if self.train:
            logging.debug("In train mode")
            traintest_filter = lambda x: 'test' not in x.stem
        else:
            logging.debug("In test mode")
            traintest_filter = lambda x: 'test' in x.stem
        prev_len = len(self.file_paths)
        self.file_paths = [x for x in self.file_paths if traintest_filter(x)]
        logging.info(f"Train/test filtered out {prev_len - len(self.file_paths)} files, now {len(self.file_paths)}")
        
        # Possibly extract shared decoder
        for path in self.file_paths:
            if "dec" in path.stem:
                self.shared_decoder_path = path
                self.file_paths.remove(path)
                logging.debug(f"PTv3 dataset found shared dec at {self.shared_decoder_path}")
        
        self.file_paths = [path for path in self.file_paths if "dec" not in path.stem]
        self.file_paths = [path for path in self.file_paths if "statistics" not in path.stem]
        
        assert len(self.file_paths) > 0
        
        self.use_cache = use_cache
        if use_cache:
            self.cache = [None] * self.__len__()
        
        logging.debug(f"PTv3 dataset of {self.__len__()} latent fields")
    
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        arc = None
        if self.use_cache:
            arc = self.cache[idx]

        if arc is None:
            path = self.file_paths[idx]
            state_dict = torch.load(path, map_location="cpu")

            coords = state_dict["encoder"]['latents_pos.weight']
            feats = state_dict["encoder"]['latents.weight']
            label_onehot = self.label_extractor(path)
            
            # Add a third, zero dimension to coords to make it 3D
            num_points = coords.shape[0]
            coords = torch.cat(
                [coords, torch.zeros(num_points, 1)], dim=1
            )
            
            # Point transformer V3 compatible datapoint
            arc = {
                "coord": coords,
                "feat": feats,
                "class": label_onehot,
                "grid_size": self.grid_size,
                "offset": torch.tensor(num_points),
                "path": path,
            }
            
            if self.use_cache:
                self.cache[idx] = arc
        
        arc = copy.deepcopy(arc)
        with torch.no_grad():
            for tr in self.transform:
                arc = tr(arc)
        
        return arc



class ArcCollator:
    def __init__(self, num_classes=10, apply_cutmix=False, cutmix_alpha=1.0):
        self.num_classes = num_classes
        self.apply_cutmix = apply_cutmix
        self.cutmix_alpha = cutmix_alpha
    
    def cutmix_naive(self, batch):
        lam = np.random.beta(self.cutmix_alpha, self.cutmix_alpha)
        
        for i in range(len(batch)):
            # Get current and next item in shuffled batch to mix with
            j = (i + 1) % len(batch)
            coords1, feats1, labelonehot1 = batch[i]["coord"], batch[i]["feat"], batch[i]["class"]
            coords2, feats2, labelonehot2 = batch[j]["coord"], batch[j]["feat"], batch[j]["class"]
            
            # Remove <lam> points from 1
            # Remove <1-lam> points from 2
            num_points1 = int(lam * coords1.shape[0])
            num_points2 = int((1-lam) * coords2.shape[0])
            random_idx1 = torch.randperm(coords1.shape[0])[:num_points1]
            random_idx2 = torch.randperm(coords2.shape[0])[:num_points2]
            
            # Apply CutMix
            batch[i]["coord"] = torch.cat([coords1[random_idx1], coords2[random_idx2]])
            batch[i]["feat"] = torch.cat([feats1[random_idx1], feats2[random_idx2]])
            onehot_mixed = lam * labelonehot1 + (1 - lam) * labelonehot2
            batch[i]["class"] = onehot_mixed
        
        return batch
    
    def __call__(self, batch):
        # If CutMix is enabled, apply it to the batch
        if self.apply_cutmix:
            batch = self.cutmix_naive(batch)
        
        # Collate the batch as usual
        offsets = torch.tensor([point["coord"].shape[0] for point in batch])
        offsets = torch.cumsum(offsets, dim=0)

        # Assemble the batch dictionary
        batch_point = {
            "coord":    torch.cat([point["coord"] for point in batch]),
            "feat":     torch.cat([point["feat"] for point in batch]),
            "grid_size": batch[0]["grid_size"],
            "offset":   offsets,
            "batch_size": len(batch),
            "path":     [point["path"] for point in batch],
            "cutmix":   self.apply_cutmix,
            "class":    torch.stack([point["class"] for point in batch])
        }
        
        return batch_point
