import torch
from . import arch as ARCH
from copy import deepcopy


def build_model(cfg, pretrain_path=r""):
    cfg = deepcopy(cfg)
    arch_cfg = cfg['model']['arch']
    arch_type = arch_cfg.pop("type")
    if hasattr(ARCH,arch_type):
        model = getattr(ARCH,arch_type)(cfg)
    else:
        raise KeyError("`arch_type` not found. Got {}".format(arch_type))
    
    if pretrain_path:
        checkpoint_model = torch.load(pretrain_path, map_location='cpu')
        print("Load pre-trained checkpoint from: %s" % pretrain_path)
        msg = model.load_state_dict(checkpoint_model['model'], strict=False)
        print(msg)
    return model