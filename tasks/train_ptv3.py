from datetime import datetime
from pathlib import Path
import random
import torch
from torch.utils.data import DataLoader
import lightning as pl
import lightning.pytorch.callbacks as pl_callbacks

import sys
import logging
from argparse import ArgumentParser

import utils_platform
import data.dataset_ptv3 as Ptv3Data
from models_downstream.ptv3 import PointtransformerV3Lightning


if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True
    
    parser = ArgumentParser()
    parser.add_argument("--alias", type=str, default=None, required=True)
    parser.add_argument("--config-path", type=str, required=True, help="Name of config file")
    parser.add_argument("--dataset-dir", type=str, help="Name of dataset folder")
    parser.add_argument("--subset", type=int, default=-1)
    parser.add_argument("--no-wandb", action='store_true', default=False, help="If provided, don't use wandb")
    parser.add_argument("--no-imagelogging", action='store_true', default=False)
    parser.add_argument("--wandb-project", type=str, default="Placeholder_CIFAR")
    
    args = parser.parse_args()
    
    ## Parse input arguments
    subset = args.subset
    # Accept both 'utils_platform.dataset_root / dataset_dir' and 'dataset_dir'
    dataset_dir = Path(args.dataset_dir)
    try:
        dataset_dir.relative_to(utils_platform.dataset_root)
    except ValueError:
        dataset_dir = utils_platform.dataset_root / dataset_dir
    
    config_path = Path(args.config_path).with_suffix('.py')
    assert config_path.exists(), f'!! Error. Config path `{config_path}` does not exist'
    
    ## Load config from file
    import importlib.util
    spec = importlib.util.spec_from_file_location("config_module", config_path)
    config_module = importlib.util.module_from_spec(spec)
    sys.modules["config_module"] = config_module
    spec.loader.exec_module(config_module)
    config = config_module.config
    
    ## Parse config
    is_jax_dataset = config['dataset_args'].get('jax')
    is_jax_dataset = False if is_jax_dataset is None else is_jax_dataset
    batch_size = config['pt_config']['batch_size']
    if config['cutmix_alpha'] is not None:
        use_cutmix = True
        cutmix_alpha = config['cutmix_alpha']
    else:
        use_cutmix = False
    
    current_time = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    name = f'{current_time}-{args.alias}'
    
    print(f'{name=}')
    if args.no_wandb:
        logging.info('No Weights and Biases')
        pl_logger = False
    else:
        pl_logger = pl.pytorch.loggers.WandbLogger(
            entity="joost_",
            project=args.wandb_project,
            name=name,
            config=config,
        )
        
        # Create a breadcrumb in the dataset file if this data has been
        #   traceably (i.e. it's in WandB) trained
        open(dataset_dir / f'{name}_trained.txt', 'a').close()
    
    
    ## Load datasets
    print("Retrieving file paths")
    start = datetime.now()
    paths_dict = Ptv3Data.get_arc_files(dataset_dir)
    print(datetime.now() - start)
    
    train_paths = paths_dict["train"]
    assert len(train_paths) > 0
    random.shuffle(train_paths)
    
    val_paths = paths_dict["val"]
    if len(val_paths) == 0:
        train_len = int(len(train_paths) * 0.8)
        print(len(train_paths), train_len)
        val_paths = train_paths[train_len:]
        train_paths = train_paths[:train_len]
        print(f"Val path len = 0, now {len(train_paths)=}, {len(val_paths)=}")
    
    test_paths = paths_dict["test"]
    
    if subset != -1:
        print("Getting subset")
        subset_fraction = subset / len(train_paths)
        train_paths = train_paths[: subset]
        val_paths = val_paths[: int(subset_fraction * len(val_paths))]
        test_paths = test_paths[: int(subset_fraction * len(test_paths))]
    
    print(f"train paths {len(train_paths)}")
    print(f"val paths {len(val_paths)}")
    print(f"test paths {len(test_paths)}")
    
    
    logging.debug("Creating train set...")
    train_set = Ptv3Data.ArcDataset(**config['dataset_args'], root_dir=dataset_dir, transform=config['train_transforms'], file_paths=train_paths,)
    logging.debug("Creating val set...")
    val_set = Ptv3Data.ArcDataset(**config['dataset_args'], root_dir=dataset_dir, transform=config['val_transforms'], file_paths=val_paths,)
    
    ## Create dataloaders
    if use_cutmix:
        collator_cutmixTrueFalse = Ptv3Data.ArcCollator(num_classes=config['pt_config']['out_channels'], apply_cutmix=use_cutmix, cutmix_alpha=cutmix_alpha)
    else:
        collator_cutmixTrueFalse = Ptv3Data.ArcCollator(num_classes=config['pt_config']['out_channels'])
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=collator_cutmixTrueFalse,
        persistent_workers=True,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=Ptv3Data.ArcCollator(num_classes=config['pt_config']['out_channels']),
        persistent_workers=True,
        pin_memory=True,
    )
    print(f'Num train data {len(train_loader.dataset)}, num batches {len(train_loader)}')
    print(f'Num val data {len(val_loader.dataset)}, num batches {len(val_loader)}')
    if pl_logger is not False:
        pl_logger.experiment.config.update(
            {
                "train_size": len(train_loader.dataset),
                "val_size": len(val_loader.dataset),
                "train_batches": len(train_loader),
                "val_batches": len(val_loader),
            }
        )


    point_transformer = PointtransformerV3Lightning(config)
    
    callbacks = []
    callbacks.append(pl_callbacks.ModelCheckpoint(dirpath=dataset_dir, filename = name + '_checkpoint_{epoch}_{val_acc:.2f}_ptv3', monitor="val_acc", mode='max'))
    
    trainer = pl.Trainer(
        max_epochs=300,
        enable_progress_bar = not utils_platform.is_submitted_slurm,
        enable_checkpointing = True,
        logger = pl_logger,
        callbacks = callbacks,
        log_every_n_steps = 2,
        
        accelerator = 'auto',
        strategy = 'auto',
        devices = -1
    )
    print('Starting training', flush=True)
    trainer.fit(point_transformer, train_dataloaders=train_loader, val_dataloaders=val_loader)
    