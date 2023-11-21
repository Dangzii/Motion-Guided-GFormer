from . import transforms as TRANSFORMS
from . import body_normalization
from . import hand_normalization
from copy import deepcopy

def build_transforms(cfg_transforms):
    '''
    Description:
    '''
    cfg_transforms = deepcopy(cfg_transforms)
    transforms_list = list()
    for transform_item in cfg_transforms:
        transforms_type = transform_item.pop("type")
        transforms_kwags = transform_item
        if hasattr(TRANSFORMS,transforms_type):
            transform = getattr(TRANSFORMS,transforms_type)(**transforms_kwags)
            transforms_list.append(transform)
            print(transform)
        else:
            raise ValueError("\'type\' of transforms is not defined. Got {}".format(transforms_type))

    return TRANSFORMS.Compose(transforms_list)
