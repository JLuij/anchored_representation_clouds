import torch
import torch.nn as nn
import torchmetrics

import sys
import logging
import warnings
from pathlib import Path
from argparse import ArgumentParser
from tqdm import tqdm

import utils_platform
import models_arc.samplers
import data.utils_mnist
import data.dataset_image as ImageData
import configs.arc_config_core as c


def normalise(img_object):
    img = img_object['image']
    mean = img.mean(dim=[1,2], keepdim=True)
    img_object["mean"] = mean
    std = img.std(dim=[1,2], keepdim=True)
    img_object["std"] = std
    img = (img - mean) / std
    img_object["image"] = img
    return img_object

def forward_pass(model, sampler, move_back_to_cpu=False):
    global shared_dec_loss
    
    if move_back_to_cpu:
        model['encoder'].cuda()
        model['decoder'].cuda()
        sampler.cuda()
    
    model['encoder_optim'].zero_grad()
    if shared_dec is None:
        model['decoder_optim'].zero_grad()
    
    gt, idx = sampler()
    features = model['encoder'](idx)
    colour = model['decoder'](features).permute(1,0)    # [c, prod(dims)] -> [prod(dims), c]
    
    loss = criterion(colour, gt)
    loss.backward(retain_graph=True)
    
    if fit_a_dec:
        shared_dec_loss += loss
    
    model['encoder_optim'].step()
    if shared_dec is None:
        model['decoder_optim'].step()
    
    if move_back_to_cpu:
        model['encoder'].cpu()
        model['decoder'].cpu()


def create_save_dir(step):
    global save_dir
    save_dir = utils_platform.dataset_root / 'placeholder_sets' / f"{'dec' if fit_a_dec else ''}{config.alias}-{step}"
    
    if not save_dir.exists():
        logging.debug(f"Creating dir at {save_dir}")
        save_dir.mkdir(exist_ok=True, parents=True)

def save_single_model(step, model, sampler, save_components, repeat, to_and_fro_cuda, img_object=None):
    create_save_dir(step)
    if not config.get_save_path(save_dir).exists():
        config.save(save_dir)
    
    if not args.no_pred:
        if to_and_fro_cuda:
            model['encoder'].cuda()
            model['decoder'].cuda()
        
        with torch.no_grad():
            full_pred = models_arc.samplers.full_forward_pass(model, sampler.image.shape)
            full_psnr = torchmetrics.functional.image.peak_signal_noise_ratio(
                    full_pred,
                    sampler.image.cpu()).item()
        
        if to_and_fro_cuda:
            model['encoder'].cpu()
            model['decoder'].cpu()
        
        ## Reverse any reversable image transformations
        for transform in list(reversed(img_dataset.image_transforms)):
            try:
                full_pred = transform.reverse(full_pred)
            except Exception as e:
                # logging.warning(f"Transform {transform} is not reversible")
                # print(e)
                pass
        
        print("!! UNNORMALISING !!")
        full_pred = (full_pred * model["std"]) + model["mean"]
    else:
        full_psnr = -1.0
    
    ## Assemble reconstruction path
    repeat_str = f"_{repeat}" if (repeat > 1) else ''
    gt_path = Path(sampler.gt_path)
    stem = f"{gt_path.stem}_{sampler.gt_label}{repeat_str}_psnr{full_psnr:.2f}"
    
    if not args.no_pred:
        img_save_path = save_dir / f'{stem}_pred.png'
        ImageData.save_as_pil(full_pred.permute(1,2,0).cpu(), img_save_path, unnormalise=False)
    else:
        img_save_path = 'no_prediction_png_was_saved'
    
    save_path = save_dir / f"{stem}.pt"
    save_data = {}
    for part in save_components:
        save_data[part] = model[part].state_dict()
    save_data['psnr'] = full_psnr
    save_data['recon_path'] = img_save_path
    torch.save(save_data, save_path)
    
    
    return full_psnr

def save_batch(step, models_list, sampler_list, move_back_to_cpu=True):
    logging.info(f'Saving at step {step}')
    create_save_dir(step)
    config.save(save_dir)
    
    save_components = ['encoder']
    torch.save({'decoder': shared_dec.state_dict()}, save_dir / 'shared-dec.pt')
    
    average_psnr = []
    for model, sampler in zip(models_list, sampler_list):
        psnr = save_single_model(step, model, sampler, save_components, repeat=1, to_and_fro_cuda=move_back_to_cpu)
        average_psnr.append(psnr)
    average_psnr = sum(average_psnr) / len(average_psnr)
    logging.info(f"{average_psnr=:.2f}")
    
    with open(save_dir / f'{average_psnr=:.2f}.txt', 'w'):
        pass
    

def create_and_init_model(img_object):
    print("!! NORMALISING IMAGE !!", flush=True)
    img_object = normalise(img_object)
    print("!! DONE NORMALISING IMAGE !!", flush=True)
    
    image = img_object['image']
    sampler = settings['coord_subsampler'](img_object)
    
    spatial_dims = image.shape[1:]
    if shared_grid is None:
        grid_subset = data.utils.generate_grid_integer_coords((spatial_dims[0], spatial_dims[1])).requires_grad_(False)
    else:
        grid_subset = shared_grid[:spatial_dims[0], :spatial_dims[1]]
    
    ## Initialise encoder
    with torch.no_grad():
        latents_pos, gradient = settings['latent_position_init'](image)
        num_latents = latents_pos.shape[0]
        
        encoder = settings['inr_type'](num_latents, **settings['enc_arguments'])
        encoder.latents_pos.weight.data = latents_pos
    encoder.get_NN_information(grid_subset.view(-1, 2))
    
    enc_optim = settings['enc_optimiser'](encoder.parameters())
    
    if shared_dec is None:
        decoder = settings['dec_type'](**settings['dec_arguments'])
        decoder.initialise_()
        
        dec_optim = settings['dec_optimiser'](decoder.parameters())
    else:
        decoder = shared_dec
        dec_optim = None
    
    model = dict(
        encoder = encoder,
        decoder = decoder,
        encoder_optim = enc_optim,
        decoder_optim = dec_optim,
        mean = img_object["mean"],
        std = img_object["std"],
    )
    return model, sampler



if __name__ == "__main__":
    warnings.filterwarnings("error")        # Crash on e.g. loss shape mismatch
    torch.backends.cuda.matmul.allow_tf32 = True
    
    parser = ArgumentParser()
    parser.add_argument("--config-path", type=str, required=True)
    parser.add_argument("--subset", type=int, default=-1)
    parser.add_argument("--dec-path", type=str, default="")
    parser.add_argument("--fraction", type=utils_platform.cli_partwhole, default="1/1")
    parser.add_argument("--fixed-size", type=utils_platform.cli_tuple_type, default=None, help=
                        "Provide a predetermined max size as \"max_height,max_width\"")
    parser.add_argument("--no-pred", action='store_true', default=False, help="If provided, don't save prediction .pngs")
    args = parser.parse_args()
    
    ## Parse input arguments
    config_path = Path(args.config_path)
    config_path = config_path if str(config_path).endswith(".py") else Path(str(config_path) + ".py")
    assert config_path.exists(), f"Config path does not exist {config_path.as_posix()}"
    
    dec_path = Path(args.dec_path) if (args.dec_path != "") else None
    if dec_path is not None:
        assert dec_path.exists(), f"Decoder path does not exist {dec_path.as_posix()}"
    
    ## Load config to object
    import importlib.util
    spec = importlib.util.spec_from_file_location("config_module", config_path)
    config_module = importlib.util.module_from_spec(spec)
    sys.modules["config_module"] = config_module
    spec.loader.exec_module(config_module)
    config: c.Config = config_module.config
    
    settings = config.parse()
    
    ## Determine whether we have to fit a shared decoder first
    # if dec_share but no path: fit a dec on subset
    # if dec_share and path: load and freeze dec, proceed
    # if no dec_share and no path: proceed
    # if no dec_share and path: error
    fit_a_dec = False
    if config.dec_share:
        shared_dec = settings['dec_type'](**settings['dec_arguments'])
        shared_dec.initialise_()
        
        if dec_path is not None:
            logging.info(f"Using shared frozen decoder at {dec_path}")
            pretrained_dec_statedict = torch.load(dec_path, map_location='cuda')['decoder']
            shared_dec.load_state_dict(pretrained_dec_statedict)
            
            # Freeze dec
            for param in shared_dec.parameters():
                param.requires_grad = False
        else:
            logging.info("Training decoder from scratch")
            fit_a_dec = True
        
        shared_dec_optim = settings['dec_optimiser'](shared_dec.parameters())
    else:
        logging.info("Not using a shared decoder")
        shared_dec = None
        assert dec_path is None, "Dec path can't be specified if dec_share is False"
    
    ## Create image dataset
    img_dataset = settings['img_dataset']
    if fit_a_dec:
        img_loader = ImageData.GetDataloader(
            img_dataset,
            seed=-1,
            fraction_or_num_elements=args.subset,
        )
    else:
        img_loader = ImageData.GetDataloader(
            img_dataset,
            fraction_or_num_elements=args.subset,
            dataset_fraction=args.fraction,
        )
    
    
    ## Pregenerate grid
    if config.img_dataset[0] in [c.DatasetEnum.MNIST, c.DatasetEnum.CIFAR, c.DatasetEnum.FASHIONMNIST]:
        ## Dataset will have images of the same size
        # Load a sample image and get its size
        sample_img = img_dataset.__getitem__(0)['image']
        grid_size = tuple(sample_img.shape[1:])
        logging.info(f"Using constant size dataset {config.img_dataset} of grid size {grid_size}")
        shared_grid = data.utils.generate_grid_integer_coords(grid_size).requires_grad_(False)
        logging.info(f'{shared_grid.shape=}')
    else:
        shared_grid = None
    
    criterion = nn.MSELoss()
    
    if fit_a_dec:
        ## Initialse the models
        models_list = [None] * len(img_loader)
        samplers_list = [None] * len(img_loader)
        
        for i, image_object in enumerate(tqdm(img_loader, desc="Initialising", disable=utils_platform.is_submitted_slurm)):
            model, sampler = create_and_init_model(image_object)
            model['encoder'].cuda()
            model['decoder'].cuda()
            sampler.cuda()
            
            models_list[i] = model
            samplers_list[i] = sampler
        
        ## Fit the models
        step_pbar = tqdm(range(config.num_steps), desc=f"Step count ({len(models_list)} INRs)")
        for step in step_pbar:
            shared_dec_optim.zero_grad()
            shared_dec_loss = 0
            
            model_pbar = tqdm(range(len(models_list)), desc="Model", leave=False, disable=True)
            for i in model_pbar:
                forward_pass(models_list[i], samplers_list[i], move_back_to_cpu=True)
            
            if fit_a_dec:
                shared_dec_loss /= len(models_list)
                shared_dec_loss.backward()
                shared_dec_optim.step()
            
            if (step+1) in settings['save_intermittently_at']:
                save_batch(step+1, models_list, samplers_list, move_back_to_cpu=False)
        save_batch(step+1, models_list, samplers_list, move_back_to_cpu=False)
    
    else:
        save_components = ['encoder']
        if shared_dec is None:
            save_components.append('decoder')
        
        img_pbar = tqdm(img_loader, desc="Fitting INRs serialised")
        for img_object in img_pbar:
            for repeat in range(settings['num_repeats']):
                model, sampler = create_and_init_model(img_object)
                model['encoder'].cuda()
                model['decoder'].cuda()
                sampler.cuda()
                
                step_pbar = tqdm(range(settings['num_steps']), leave=False, disable=True)
                for step in step_pbar:
                    forward_pass(model, sampler, move_back_to_cpu=False)
                    
                    if (step+1) in config.save_intermittently_at:
                        save_single_model(step+1, model, sampler, save_components, repeat=repeat, to_and_fro_cuda=False, img_object=img_object)
                
                save_single_model(step+1, model, sampler, save_components, repeat=repeat, to_and_fro_cuda=False, img_object=img_object)
    
    logging.info('Done!')
