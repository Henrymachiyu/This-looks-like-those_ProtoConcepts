import os
import torch
import torch.nn.functional as F
import numpy as np
import math

def list_of_distances(X, Y):
    return torch.sum((torch.unsqueeze(X, dim=2) - torch.unsqueeze(Y.t(), dim=0)) ** 2, dim=1)

def make_one_hot(target, target_one_hot):
    target = target.view(-1,1)
    target_one_hot.zero_()
    target_one_hot.scatter_(dim=1, index=target, value=1.)

def makedir(path):
    '''
    if path does not exist in the file system, create it
    '''
    if not os.path.exists(path):
        os.makedirs(path)

def print_and_write(str, file):
    print(str)
    file.write(str + '\n')

def find_high_activation_crop(activation_map, percentile=90):
    threshold = np.percentile(activation_map, percentile)
    mask = np.ones(activation_map.shape)
    mask[activation_map < threshold] = 0
    lower_y, upper_y, lower_x, upper_x = 0, 0, 0, 0
    for i in range(mask.shape[0]):
        if np.amax(mask[i]) > 0.5:
            lower_y = i
            break
    for i in reversed(range(mask.shape[0])):
        if np.amax(mask[i]) > 0.5:
            upper_y = i
            break
    for j in range(mask.shape[1]):
        if np.amax(mask[:,j]) > 0.5:
            lower_x = j
            break
    for j in reversed(range(mask.shape[1])):
        if np.amax(mask[:,j]) > 0.5:
            upper_x = j
            break
    return lower_y, upper_y+1, lower_x, upper_x+1

def upscale_rf(protoL_rf_info, proto_act_img_j, img_size): 
    n = protoL_rf_info[0]
    j = protoL_rf_info[1]
    r = protoL_rf_info[2]
    start = math.floor(protoL_rf_info[3])

    x_min = start
    x_max = start + (proto_act_img_j.size(0) - 1) * j

    mesh = torch.tensor([-1 + i * 2. / (j * (proto_act_img_j.size(0) - 1)) for i in range(img_size)])
    mesh_x, mesh_y = torch.meshgrid(mesh, mesh, indexing='ij')
    grid = torch.stack((mesh_y, mesh_x), dim=2).unsqueeze(0)
    gridsample_in = proto_act_img_j.unsqueeze(0).unsqueeze(1)
    upsampled_act_img_j = F.grid_sample(gridsample_in, grid, mode='bicubic', padding_mode='zeros', align_corners=True)

    return upsampled_act_img_j
