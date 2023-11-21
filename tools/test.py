import sys
sys.path.append('../')
from zmq import device
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import logging 
logging.getLogger("requests").setLevel(logging.WARNING)
import torch.nn as nn
import torch
from argparse import ArgumentParser
import datetime
import os
from copy import deepcopy



from configs import merage_from_arg,load_arg
from src.model import build_model
from src.solver import make_optimizer,wrapper_lr_scheduler
from src.engine import do_test
from src.utils import get_free_device_ids,find_lr
from src.data.dataloader import build_dataloader





if __name__ == "__main__":
    
    arg = vars(load_arg())
    print(arg)

    config_file = arg["CONFIG_FILE"]
    config_file = config_file.replace("../","").replace(".py","").replace('/','.')

    exec(r"from {} import config as cfg".format(config_file))

    cfg = merage_from_arg(cfg,arg)
    print(cfg)
    cfg = deepcopy(cfg)

    # train_dataloader = build_dataloader(cfg['train_pipeline'])
    test_dataloader = build_dataloader(cfg['test_pipeline'])

    model = build_model(cfg,pretrain_path=arg['load_path'])
    optimizer = make_optimizer(cfg['optimizer'],model)
    lr_scheduler = wrapper_lr_scheduler(cfg['lr_scheduler'],optimizer)


    if arg['device']: 
        free_device_ids = arg['device']
    else:
        free_device_ids = get_free_device_ids()

    max_num_devices = cfg['max_num_devices']
    if len(free_device_ids)>=max_num_devices:
        free_device_ids = free_device_ids[:max_num_devices]

    print("free_device_ids :",free_device_ids)
    master_device = 0
    print(master_device)
    model.cuda(master_device)
    if cfg['multi_gpu']:
        model = nn.DataParallel(model,device_ids=free_device_ids).cuda(master_device)
    
    if cfg['enable_backends_cudnn_benchmark']:
        print("enable backends cudnn benchmark")
        torch.backends.cudnn.benchmark = True


    do_test(cfg,model=model,val_loader=test_dataloader,metrics=None,device=free_device_ids)


