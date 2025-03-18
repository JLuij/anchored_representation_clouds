

import data.utils_ptv3

config = dict(
    pt_config = dict(
        batch_size = 1024,
        out_channels = 10,
        
        learning_rate = 1e-3,
        weight_decay = 0.001,
        
        pointtransformer_args = dict(
            in_channels =  32,
            cls_mode = True,
            stride =         (2,2,2,),
            enc_depths =     (1,    1,      2,    1, ),
            enc_channels =   (64,   64,     64,   64,),
            enc_num_head =   (8,    16,     32,   32,),
            enc_patch_size = (25,   25,     25,   25,),
            mlp_ratio = 2,
            attn_drop = 0.0,
            proj_drop = 0.0,
            drop_path = 0.3,
            
            enable_rpe = True,
            enable_cpe = True,
            enable_flash = False,
        ), 
    ),
    
    cutmix_alpha = 1.0,     # None for no cutmix, else float in [0, 1]
    dataset_args = dict(
        grid_size = 2,
        include_if_path_lambda = None,
        train=True,
        at_step=500,
    ),
    
    train_transforms = [
        data.utils_ptv3.PointDropout(0.3),
        data.utils_ptv3.NormalisePerLatentDim('/home/path/to/fashionmnist_pngs/normalisation_statistics_500.pt'),
    ],
    val_transforms = [
        data.utils_ptv3.NormalisePerLatentDim('/home/path/to/fashionmnist_pngs/normalisation_statistics_500.pt'),
    ],
    test_transforms = [
        data.utils_ptv3.NormalisePerLatentDim('/home/path/to/fashionmnist_pngs/normalisation_statistics_500_test.pt'),
    ]
    
)
