import torch
import torch.nn as nn

from pathlib import Path

import configs.arc_config_core as config_core


def init_arc(
        image_object,
        grid: torch.Tensor,
        parsed_config: dict,
    ):
    if torch.is_tensor(image_object):
        image_object = dict(image=image_object)
    
    image = image_object['image']
    sampler = parsed_config['coord_subsampler'](image_object)
    
    ## Initialise encoder
    with torch.no_grad():
        latents_pos, gradient = parsed_config['latent_position_init'](image)
        num_latents = latents_pos.shape[0]
        
        encoder = parsed_config['inr_type'](num_latents, **parsed_config['enc_arguments'])
        encoder.latents_pos.weight.data = latents_pos
    
    grid_np = grid.view(-1, 2)
    grid_np = grid_np.numpy() if torch.is_tensor(grid) else grid_np
    encoder.get_NN_information(grid_np)
    enc_optim = parsed_config['enc_optimiser'](encoder.parameters())
    
    ## Get decoder
    decoder = parsed_config['dec_type'](**parsed_config['dec_arguments'])
    dec_optim = parsed_config['dec_optimiser'](decoder.parameters())
    
    # Accumulate
    model = dict(
        encoder = encoder,
        decoder = decoder,
        encoder_optim = enc_optim,
        decoder_optim = dec_optim,
    )
    return model, sampler



def fit_arc(image: torch.Tensor, config_path: Path):
    config = config_core.load_config(config_path)
    fit_arc(image, config)

def fit_arc(image: torch.Tensor, config: config_core.Config):
    pass

def forward_pass(model, sampler, shared_dec_loss=None, criterion=nn.MSELoss(), move_back_to_cpu=False):
    shared_dec = True if (shared_dec_loss is not None) else False
    
    if move_back_to_cpu:
        model['encoder'].cuda()
        sampler.cuda()
        if not shared_dec: model['decoder'].cuda()
    
    model['encoder_optim'].zero_grad()
    model['decoder_optim'].zero_grad()
    
    gt, idx = sampler()
    features = model['encoder'](idx)
    colour = model['decoder'](features).permute(1,0)    # [c, prod(dims)] -> [prod(dims), c]
    
    loss = criterion(colour, gt)
    if not shared_dec:
        loss.backward()
    else:
        loss.backward(retain_graph = True)
        shared_dec_loss += loss
    
    model['encoder_optim'].step()
    if not shared_dec: model['decoder_optim'].step()
    
    if move_back_to_cpu:
        model['encoder'].cpu()
        if not shared_dec: model['decoder'].cpu()
