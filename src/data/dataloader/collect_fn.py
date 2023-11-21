import torch

def my_collate_fn(batch_list):
    image_list = list()
    label_list = list()
    for i in range(len(batch_list)):
        image,labels = batch_list[i]
        image_list.append(image.squeeze(0))
        label_list.append(labels)
    return torch.cat(image_list,dim=0),torch.cat(label_list,dim=0)
