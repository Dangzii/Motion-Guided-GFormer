from  torch.utils.data import Dataset
import os
import sys
import numpy as np
import h5py
import cv2 as cv
from copy import deepcopy
from collections import defaultdict
import torch
from ..transforms import functional as F



def load_h5py(path):
    with h5py.File(path,"r") as hf:
        x = np.array(hf.get('matrix'))
        x = np.transpose(x,(0,1,3,2)).reshape(x.shape[0],-1)
        y = np.array(hf.get('label'))
        print(x.shape)
        return x,y
    
    
def load_SHREC(path):
    with h5py.File(path,"r") as hf:
        x = np.array(hf.get('matrix'))
        y_14 = np.array(hf.get('labels_14'))
        y_28 = np.array(hf.get('labels_28'))
        print(x.shape)
        return x,y_14,y_28

def sel_temp_point(batch_xs, sel_temp_points_id=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], sel_points_id=[i for i in range(20)], final_frame_nb=39,two_dim_flag=False):
    '''
    select joints
    '''
    batch_xs_temp = np.array([[]]*batch_xs.shape[0])
    if two_dim_flag:
        for i in sel_temp_points_id:
            p_id = sel_points_id.index(i)
            batch_xs_temp = np.append(batch_xs_temp, \
                batch_xs[:,p_id*(final_frame_nb*2):(p_id+1)*(final_frame_nb*2)], axis=1)
        return batch_xs_temp
    else:
        for i in sel_temp_points_id:
            p_id = sel_points_id.index(i)
            batch_xs_temp = np.append(batch_xs_temp, \
                batch_xs[:,p_id*(final_frame_nb*3):(p_id+1)*(final_frame_nb*3)], axis=1)
        return batch_xs_temp, len(sel_temp_points_id)


class balance_dataset(Dataset):
    '''
    sample (sample_per_classes*num_classes) to bulid a balance dataset to form a mini-batch
    '''
    def __init__(self,root_path,samples_per_classes=4,len_dataset=100,transforms=None):
        self.samples_per_classes = samples_per_classes
        self.len_dataset = len_dataset
        self.transforms = transforms
        data_arr,label_arr = load_h5py(root_path)
        data_arr = sel_temp_point(data_arr)
        self.data_arr = data_arr
        self.label_arr = np.argmax(label_arr,axis=1)

        self.instance_dict = defaultdict(list)
        for i in range(len(self.label_arr)):
            class_id = self.label_arr[i]
            self.instance_dict[class_id].append(i)
        self.class_ids = list(self.instance_dict.keys())
        

    def __len__(self):
        return self.len_dataset
    
    def __getitem__(self,idx):
        # idx = np.random.choice(self.class_ids,size=1)[0] # 0-19 选一个id
        # print(idx)
        sampler_indices = np.random.choice(self.instance_dict[idx],size=self.samples_per_classes)
        sampler_indices = np.array(sampler_indices).reshape(-1)
        batch_raw_data,batch_label = self.data_arr[sampler_indices],self.label_arr[sampler_indices]
        # print(self.step,batch_label)
        # self.step += 1

        batch_sampler = [{"raw_data":batch_raw_data[i][None,:],"label":batch_label[i]} for i in range(len(batch_label))]
        if self.transforms:
            batch_sampler = [self.transforms(sample) for sample in batch_sampler]
        
        batch_raw_data,batch_label = list(),list()
        for i in range(len(batch_sampler)):
            batch_raw_data.append(batch_sampler[i]['raw_data'].unsqueeze(0))
            batch_label.append(batch_sampler[i]['label'])
        batch_raw_data = torch.cat(batch_raw_data,dim=0).float()
        batch_label = torch.tensor(batch_label)

        return batch_raw_data,batch_label



class dataset(Dataset):
    def __init__(self, path, sel=False, transforms=None):
        data_arr,label_arr = load_h5py(os.path.abspath(sys.path[-1])+path)
        if sel:
            data_arr, self.joint = sel_temp_point(data_arr)
        else:
            self.joint = 19
        self.data_arr = data_arr
        self.label_arr = np.argmax(label_arr,axis=1)
        self.transforms = transforms
    
    def __len__(self):
        return len(self.data_arr)


    def __getitem__(self,idx):
        raw_data,label = (self.data_arr[idx,:])[None,:],self.label_arr[idx]
        sample = {'raw_data':raw_data,'label':label}
        # import pdb; pdb.set_trace()

        if self.transforms:
            sample = self.transforms(sample)
        raw_data,label = sample['raw_data'],sample['label']
        # import pdb; pdb.set_trace()
        raw_data = raw_data
        raw_data = raw_data.view(self.joint, -1, 40)   # NJCT
        raw_data = raw_data.permute(1, 0, 2).contiguous().float() # NCJT
        return raw_data.float(), label


class SHRECdataset(Dataset):
    def __init__(self, root_path, transforms=None, labelnum=14):
        self.data_arr, self.label14_arr, self.label28_arr = load_SHREC(root_path)
        self.transforms = transforms
        self.labelnum = labelnum

    def __len__(self):
        return len(self.data_arr)

    def __getitem__(self, idx):
        data = (self.data_arr[idx])[None,:]
        if self.labelnum == 14:
            label = self.label14_arr[idx]-1
        else:
            label = self.label28_arr[idx]-1
        raw_data = data.reshape(-1)[None,:]
        sample = {'raw_data': raw_data, 'label': label}
        # import pdb; pdb.set_trace()
        
        if self.transforms:
            sample = self.transforms(sample)
        raw_data, label = sample['raw_data'], sample['label']
        # import pdb; pdb.set_trace()
        raw_data = raw_data.view(22, 3, -1)  # NJCT
        raw_data = raw_data.permute(1, 0, 2).contiguous().float()
        return raw_data.float(), label
    


