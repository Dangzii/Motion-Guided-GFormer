from . import dataset as DATASETS
from . import sign_dataset as SIGN_DATASETS

from copy import deepcopy

def build_dataset(cfg_dataset,transforms):
    '''
    Description:
    '''
    cfg_dataset = deepcopy(cfg_dataset)
    dataset_type = cfg_dataset.pop("type")
    dataset_kwags = cfg_dataset
    
    if hasattr(DATASETS,dataset_type):
        dataset = getattr(DATASETS,dataset_type)(**dataset_kwags,transforms=transforms)
    elif hasattr(SIGN_DATASETS,dataset_type):
        dataset = getattr(SIGN_DATASETS,dataset_type)(**dataset_kwags,transforms=transforms)
    else:
        raise ValueError("\'type\' of dataset is not defined. Got {}".format(dataset_type))
    return dataset