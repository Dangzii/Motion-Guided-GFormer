
import torch
import numbers
import warnings
import types
import cv2 as cv
import numpy as np
import random
from . import functional as F
import math


class RandomChoice(object):
    """
    Apply transformations randomly picked from a list with a given probability
    Args:
        transforms: a list of transformations
        p: probability
    """
    def __init__(self,p,transforms):
        self.p = p
        self.transforms = transforms
    def __call__(self,sample):
        if len(self.transforms) < 1:
            raise TypeError("transforms(list) should at least have one transformation")
        for t in self.transforms:
            if np.random.uniform(0,1) < self.p:
                sample = t(sample)
        return sample

    def __repr__(self):
        return self.__class__.__name__+"(p={})".format(self.p)


class Compose(object):
    '''
    Description: Compose several transforms together
    Args (type): 
        transforms (list): list of transforms
        sample (ndarray or dict):
    return: 
        sample (ndarray or dict)
    '''
    def __init__(self,transforms):
        self.transforms = transforms
    def __call__(self,sample):
        for t in self.transforms:
            sample = t(sample)
        return sample
    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string

class ToTensor(object):
    '''
    Description: Convert ndarray in sample to Tensors.
    Args (type): 
        sample (ndarray or dict)
    return: 
        Converted sample.
    '''
    def __call__(self,sample):
        return F.to_tensor(sample)
    def __repr__(self):
        return self.__class__.__name__ + "()"


class Lambda(object):
    '''
    Description: Apply a user-defined lambda as a transform.
    Args (type): lambd (function): Lambda/function to be used for transform.
    Return: 
    '''
    def __init__(self, lambd):
        assert isinstance(lambd, types.LambdaType)
        self.lambd = lambd

    def __call__(self, img):
        return self.lambd(img)

    def __repr__(self):
        return self.__class__.__name__ + '()'





class Joint2Bone(object):
    def __init__(self,num_joints,sample_rate,dataset):
        self.num_joint = num_joints
        self.sample_rate = sample_rate
        self.dataset = dataset
    
    def __call__(self, sample):
        return F.joint2bone(sample,self.num_joint,self.sample_rate,self.dataset)
        
    def __repr__(self):
        format_string = self.__class__.__name__ + "(dataset={0})".format(self.dataset)
        return format_string


class Resizehand(object):
    def __init__(self,num_joints,sample_rate):
        self.num_joint = num_joints
        self.sample_rate = sample_rate

    
    def __call__(self, sample):
        return F.resize(sample,self.num_joint,self.sample_rate)


class RandomShift(object):
    def __init__(self,p=1,shift_limit=5,final_frame_nb=39):
        # shift_limit : max shifit
        # final_frame_nb : frame number
        assert (shift_limit>=0) and (shift_limit<=final_frame_nb/5)
        self.p = p
        self.shift_limit = shift_limit
        self.final_frame_nb = final_frame_nb

    def __call__(self,sample):
        if np.random.random() <= self.p:
            return F.randomshift(sample,self.shift_limit,self.final_frame_nb)
        else:
            return sample
    def __repr__(self):
        return self.__class__.__name__ + "(p={},shift_limit={},final_frame_nb={})".format(self.p,self.shift_limit,self.final_frame_nb)

class GaussNoise(object):
    def __init__(self,p=1,scale=0.001,percentage=0.1):
        self.scale = scale
        self.p = p
        self.percentage = percentage
    
    def __call__(self,sample):
        if np.random.random() <= self.p:
            return F.gaussnoise(sample,self.scale,self.percentage)
        else:
            return sample
    def __repr__(self):
        return self.__class__.__name__ + "(p={},scale={})".format(self.p,self.scale)


class Interpolation(object):
    def __init__(self,num_joints=22,sample_rate=66):
        self.num_joints = num_joints
        self.sample_rate = sample_rate
    
    def __call__(self,sample):
        return F.interpolation(sample,self.num_joints,self.sample_rate)

class FramesandJoints(object):
    def __init__(self,num_joints=22,sample_rate=66,p=0.5):
        self.num_joints = num_joints
        self.sample_rate = sample_rate
        self.p = p
    
    def __call__(self,sample):
        return F.framesandjoints(sample,self.num_joints,self.sample_rate,self.p)

class Valid_Crop_Resize(object):
    def __init__(self,valid_frame_num,p_interval,window):
        self.valid_frame_num = valid_frame_num
        self.p_interval = p_interval
        self.window = window
    
    def __call__(self,sample):
        return F.valid_crop_resize(sample,self.valid_frame_num,self.p_interval,self.window)

# class RandomRotation(object):
#     def __init__(self,angles=[np.pi/36, np.pi/18, np.pi/36]):
#         self.angles = angles
    
#     def __call__(self,sample):
#         pass

#     def __repr__(self):
#         return self.__class__.__name__ + "(angles={})".format(str(tuple(self.angles)))

class Normalize(object):
    '''
    Description: Normalize a tensor with mean and standard deviation.
    Args (type): 
        mean (tuple): Sequence of means for each channel.
        std (tuple): Sequence of std for each channel.
    Return: 
        Converted sample
    '''
    def __init__(self,mean,std,inplace=False):
        self.mean = mean
        self.std = std
        self.inplace = inplace
    def __call__(self,sample):
        #Convert to tensor
        mean = torch.tensor(self.mean,dtype=torch.float32)
        std = torch.tensor(self.std,dtype=torch.float32)
        return F.normalize(sample,mean,std,inplace=self.inplace)
    def __repr__(self):
        format_string = self.__class__.__name__ + "(mean={0},std={1})".format(self.mean,self.std)
        return format_string
