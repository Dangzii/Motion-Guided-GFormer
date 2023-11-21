import numpy as np
import torch.nn as nn

config = dict(
    # Basic Config
    enable_backends_cudnn_benchmark=True,
    max_epochs=3000 + 1,
    log_period=0.1,
    save_dir=r"../checkpoints/",
    save_period=1,
    n_save=5,
    log_dir=r"../log/",
    tag="",
    warm_up_epoch=5,

    # Dataset
    train_pipeline=dict(
        dataloader=dict(batch_size=64, num_workers=8, drop_last=True, pin_memory=True, shuffle=True, ),

        dataset=dict(type="dataset", sel=False, path=r"/data/ChaLearn2013/trainsubset.h5"),

        transforms=[
            dict(type="Interpolation",num_joints=19, sample_rate=40),
            dict(type="ToTensor", ),
        ],

    ),

    test_pipeline=dict(
    dataloader=dict(batch_size=64, num_workers=8, drop_last=False, pin_memory=True, shuffle=True, ),

    dataset=dict(type="dataset", sel=False, path=r"/data/ChaLearn2013/testsubset.h5"),

    transforms=[
        dict(type="Interpolation",num_joints=19,sample_rate=40),
        dict(type="ToTensor", ),
    ],

),

    # Model
    model=dict(
        arch=dict(type="Finetune"),
        encoder=dict(type="GlobalLocalFormer2", img_size=(19,40), patch_size=(2,2), 
                     num_stages=2, num_heads=[4,4,4], path_dim=12, every_sample=True,
                     embed_dim=64, embed_dims=[64,64,64], depths=[3,3,12], sample_ratio=[0.05, 0.025], 
                     qkv_bias=True, no_embed_class=False, num_classes=20,
                     drop_rate=0., attn_drop_rate=0., drop_path_rate=0., sig_head_ratio=2,
                     mlp_ratios=[4,8,4], norm_layer=nn.BatchNorm1d, max_T=1.0, decay_alpha=0.99998,
                     num_person=1, num_point=19, head_dropout=0.2),
        losses=dict(type="CrossEntropy", dropout_rate=0.5, weight=0.02),
        # losses=dict(type="MaskandCE", weight=0, ratio=0.3),
    ),

    multi_gpu=False,
    max_num_devices=1, 

    # Solver
    #lr_scheduler = dict(type="ExponentialLR",gamma=0.99997), # cycle_momentum=False if optimizer==Adam
    lr_scheduler=dict(type="CyclicLR", base_lr=1e-7, max_lr=1e-4, step_size_up=1060, mode='triangular',
                      cycle_momentum=True),  # cycle_momentum=False if optimizer==Adam
    #lr_scheduler = dict(type="MultiStepLR",  milestones=[100,200,300,400,500], gamma=0.1),

    # optimizer = dict(type="Adam",lr=1e-3,weight_decay=1e-5),
    optimizer=dict(type="SGD", lr=3e-3, weight_decay=1e-5),
    # warm_up = dict(length=2000,min_lr=1e-7,max_lr=3e-5,froze_num_lyers=0)

    find_lr=dict(init_value=1e-7, final_value=0.01, beta=0.98, ),

)

if __name__ == "__main__":
    pass