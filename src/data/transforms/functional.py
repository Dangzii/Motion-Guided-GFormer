import numpy as np
import torch
import cv2 as cv
from PIL import Image,ImageOps,ImageEnhance,ImageFilter
import warnings
import numbers
import collections
import torch.nn.functional as F
import random

def _is_tensor_image(image):
    '''
    Description:  Return whether image is torch.tensor and the number of dimensions of image.
    Reutrn : True or False.
    '''
    return torch.is_tensor(image) and image.ndimension()==3

def _is_numpy_image(image):
    '''
    Description: Return whether image is np.ndarray and the number of dimensions of image
    Return: True or False.
    '''
    return isinstance(image,np.ndarray) and (image.ndim in {2,3} )

def _is_numpy(landmarks):
    '''
    Description: Return whether landmarks is np.ndarray.
    Return: True or False
    '''
    return isinstance(landmarks,np.ndarray)


def to_tensor(sample):
    '''
    Description: Convert sample.values() to Tensor.
    Args (type): sample : {image:ndarray,target:int}
    Return: Converted sample
    '''
    # image,target = sample['image'],sample['target']
    raw_data,label = sample['raw_data'],sample['label']


    raw_data = torch.from_numpy(raw_data)
    # label = torch.from_numpy(label)
    sample['raw_data'] = raw_data
    return sample
   

def randomshift(sample,shift_limit,final_frame_nb=39):
    src_samples = sample['raw_data']
    num, dim = src_samples.shape # 1,1287
    dst_samples = np.zeros(src_samples.shape)
    trans_nb_list = np.random.randint(-shift_limit,shift_limit,num)

    for k in range(num):
        trans_nb = trans_nb_list[k]
        if trans_nb==0:
            dst_samples[k,:] = src_samples[k,:]
        elif trans_nb<0:
            src_arr = src_samples[k,:]
            dst_arr_l = np.zeros(dim)
            trans_nb = -trans_nb
            for i in range(dim//final_frame_nb):
                # left shift.
                dst_arr_l[i*final_frame_nb:(i+1)*final_frame_nb-trans_nb] = \
                    src_arr[i*final_frame_nb+trans_nb:(i+1)*final_frame_nb]
                dst_arr_l[(i+1)*final_frame_nb-trans_nb:(i+1)*final_frame_nb] = \
                    src_arr[(i+1)*final_frame_nb-1]
            dst_samples[k,:] = dst_arr_l
        else:
            src_arr = src_samples[k,:]
            dst_arr_r = np.zeros(dim)
            for i in range(dim//final_frame_nb):
                # right shift.
                dst_arr_r[i*final_frame_nb+trans_nb:(i+1)*final_frame_nb] = \
                    src_arr[i*final_frame_nb:(i+1)*final_frame_nb-trans_nb]
                dst_arr_r[i*final_frame_nb:i*final_frame_nb+trans_nb] = \
                    src_arr[i*final_frame_nb]
            dst_samples[k,:] = dst_arr_r
    sample['raw_data'] = dst_samples
    return sample

def gaussnoise(sample,scale,percentage):
    raw_data = sample['raw_data']
    totals = raw_data.shape[1]
    noise_mat = np.random.normal(scale=scale,size=raw_data.shape)
    zeros = np.random.randint(0,totals,int((1-percentage)*totals))
    noise_mat[:,zeros] = 0
    # import pdb; pdb.set_trace()
    sample['raw_data'] = raw_data + noise_mat
    return sample

def interpolation(sample,num_joints,sample_rate):
    raw_data = sample['raw_data']
    num_channels = (raw_data.shape[1]/num_joints/39)
    raw_data = raw_data.reshape((num_joints,int(num_channels),39))
    raw_data = raw_data.transpose((2,0,1)) # 39 x num_joints x num_channels
    raw_data = cv.resize(raw_data,(num_joints,sample_rate))
    raw_data = raw_data.transpose((1,2,0))
    raw_data = raw_data.reshape(-1)
    sample['raw_data'] = raw_data[None,:]
    return sample

def framesandjoints(sample,num_joints,sample_rate,p):
    raw_data = sample['raw_data']
    num_channels = (raw_data.shape[1]/num_joints/39)
    raw_data = raw_data.reshape((num_joints,int(num_channels),39))
    raw_data = raw_data.transpose((2,0,1)) # 39 x num_joints x num_channels
    raw_data = cv.resize(raw_data,(num_joints,sample_rate))
    #idx = np.random.permutation(num_joints)
    idx = np.arange(num_joints)
    if torch.rand(1) < p:
        i1, i2 = np.random.choice(idx, 2)
        idx[i1], idx[i2] = i2, i1
    raw_data = raw_data[:,idx,:]
    raw_data = raw_data.transpose((1,2,0))
    raw_data = raw_data.reshape(-1)
    sample['raw_data'] = raw_data[None,:]
    return sample


def valid_crop_resize(data_numpy,valid_frame_num,p_interval,window):
    # input: C,T,V,M
    C, T, V, M = data_numpy.shape
    begin = 0
    end = valid_frame_num
    valid_size = end - begin

    #crop
    if len(p_interval) == 1:
        p = p_interval[0]
        bias = int((1-p) * valid_size/2)
        data = data_numpy[:, begin+bias:end-bias, :, :]# center_crop
        cropped_length = data.shape[1]
    else:
        p = np.random.rand(1)*(p_interval[1]-p_interval[0])+p_interval[0]
        cropped_length = np.minimum(np.maximum(int(np.floor(valid_size*p)),64), valid_size)# constraint cropped_length lower bound as 64
        bias = np.random.randint(0,valid_size-cropped_length+1)
        data = data_numpy[:, begin+bias:begin+bias+cropped_length, :, :]
        if data.shape[1] == 0:
            print(cropped_length, bias, valid_size)

    # resize
    data = torch.tensor(data,dtype=torch.float)
    data = data.permute(0, 2, 3, 1).contiguous().view(C * V * M, cropped_length)
    data = data[None, None, :, :]
    data = F.interpolate(data, size=(C * V * M, window), mode='bilinear',align_corners=False).squeeze() # could perform both up sample and down sample
    data = data.contiguous().view(C, V, M, window).permute(0, 3, 1, 2).contiguous().numpy()

    return data


def normalize(sample,mean,std,inplace=False,num_joints=19):
    '''
    Description: Normalize a tensor image with mean and standard deviation.
    Args (type): 
        sample (torch.Tensor or dict): 
            sample(torch.Tensor):Classification
            sample({"image":image,"landmarks":landmarks}):Detection
            sample({"image":image,"mask":mask}):Segmentation
        mean (sequnence): Sequence of means for each channel.
        std (sequence): Sequence of standard devication for each channel.
    Return: 
        Converted sample
    '''
    raw_data, label = sample["raw_data"], sample["label"]
    num_channels = (raw_data.shape[1]/num_joints/40)
    raw_data = raw_data.reshape((num_joints,int(num_channels),40))
    raw_data = raw_data.permute((1,0,2)) # num_channels,Joint,Frames
    if not inplace:
        raw_data = raw_data.clone()
         
        #check dtype and device
        dtype = raw_data.dtype
        device = raw_data.device
        mean = torch.as_tensor(mean,dtype=dtype,device=device)
        std = torch.as_tensor(std,dtype=dtype,device=device)
        raw_data = raw_data.sub_(mean[:,None,None]).div_(std[:,None,None])
        raw_data = raw_data.permute((1,0,2)).reshape(-1)
        sample = {"raw_data":raw_data, "label":label}
        return sample
    else:
        raise TypeError("sample should be a torch image or a dict with keys image and landmarks/mask. Got {}".format(sample.keys()))


def joint2bone(sample,num_joints,sample_rate,dataset):
    if dataset == 'chalearn13':
        pairs = ((0, 15), (15, 16), (16, 17), (17, 18), (0, 11), (11, 12), (12, 13),
                (13, 14), (0, 1), (1, 7), (7, 8), (8, 9), (9, 10), (1, 3), (3, 4),
                (4, 5), (5, 6), (1, 2))
    elif dataset == 'nturgb+d':
        pairs = ((0, 1), (1, 20), (2, 20), (3, 2), (4, 20), (5, 4), (6, 5), (7, 6), (8, 20), (9, 8),
                (10, 9), (11, 10), (12, 0), (13, 12), (14, 13), (15, 14), (16, 0), (17, 16), (18, 17),
                (19, 18), (21, 22), (20, 20), (22, 7), (23, 24), (24, 11))
    raw_data = sample['raw_data']
    # num_channels = (raw_data.shape[1]/num_joints/sample_rate)
    raw_data = raw_data.reshape((num_joints,3,sample_rate))
    raw_data = raw_data.transpose((2,0,1)) # T x V x C
    bone = np.zeros_like(raw_data, dtype=np.float32)
    for v1, v2 in pairs:
        bone[..., v1, :] = raw_data[..., v1, :] - raw_data[..., v2, :]
    raw_data = np.concatenate((raw_data, bone),axis=-1)
    raw_data = raw_data.transpose((1,2,0))
    raw_data = raw_data.reshape(-1)
    sample['raw_data'] = raw_data[None,:]
    return sample

def resize(sample,num_joints,sample_rate):
    raw_data = sample['raw_data']
    raw_data = raw_data.reshape((num_joints,3,sample_rate)).transpose(0,2,1).reshape((num_joints*sample_rate, 3))
    max, min = np.max(raw_data,axis=0), np.min(raw_data,axis=0)
    # x_max,x_min = raw_data[:,0,:].max() , raw_data[:,0,:].min()
    # y_max,y_min = raw_data[:,1,:].max() , raw_data[:,1,:].min() 
    # z_max,z_min = raw_data[:,2,:].max() , raw_data[:,2,:].min()
    # max = np.concatenate(x_max,y_max,z_max) # 3
    # min = np.concatenate(x_min,y_min,z_min) # 3
    data = (raw_data - min) / (max - min)
    data = data.reshape((num_joints, sample_rate, 3)).transpose(0,2,1).reshape(-1)
    sample['raw_data'] = data[None,:]
    return sample

# def randomrotation(sample,angles):
#     def rxf(a):
#         x = np.array([1, 0, 0, 0,
#                     0, np.cos(a), np.sin(a), 0,
#                     0, -np.sin(a), np.cos(a), 0,
#                     0, 0, 0, 1])
#         return x.reshape(4,4)

#     def ryf(b):
#         y = np.array([np.cos(b), 0, -np.sin(b), 0,
#                     0, 1, 0, 0,
#                     np.sin(b), 0, np.cos(b), 0,
#                     0, 0, 0, 1])
#         return y.reshape(4,4)

#     def rzf(c):
#         z = np.array([np.cos(c), np.sin(c), 0, 0,
#                     -np.sin(c), np.cos(c), 0, 0,
#                     0, 0, 1, 0,
#                     0, 0, 0, 1])
#         return z.reshape(4,4)

#     raw_data = sample['raw_data']
#     x_angles = np.random.uniform(-1,1)*r_angle[0]
#     y_angles = np.random.uniform(-1,1)*r_angle[1]
#     z_angles = np.random.uniform(-1,1)*r_angle[2]
#     angles[:,0] = x_angles[:,0]
#     angles[:,1] = y_angles[:,0]
#     angles[:,2] = z_angles[:,0]