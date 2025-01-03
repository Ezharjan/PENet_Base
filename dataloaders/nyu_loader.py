import os
import os.path
import glob
import fnmatch  # pattern matching
import numpy as np
from numpy import linalg as LA
from random import choice
from PIL import Image
import torch
import torch.utils.data as data
import torchvision.transforms as T
import torchvision.transforms.functional as TF

import cv2
from dataloaders import transforms
import CoordConv
import random
import math
import json
import h5py
from PIL import Image
seed = 7240 # NLSPN
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.cuda.manual_seed_all(seed)

def get_sparse_depth1(dep, num_sample,modal_missing, epoch):
    channel, height, width = dep.shape

    assert channel == 1

    idx_nnz = torch.nonzero(dep.view(-1) > 0.0001, as_tuple=False)

    num_idx = len(idx_nnz)
    idx_sample = torch.randperm(num_idx)[:num_sample]

    idx_nnz = idx_nnz[idx_sample[:]]

    mask = torch.zeros((channel*height*width))
    mask[idx_nnz] = 1.0
    mask = mask.view((channel, height, width))

    dep_sp = dep * mask.type_as(dep)

    _, height, width = dep_sp.shape
    k = math.log2(max(1, (epoch-9)))
    # print("enhancement k is:", k)
    rect_height, rect_width = int(k * height//20), int(k * width//20)
    top_left_x = random.randint(0, width - rect_width)
    top_left_y = random.randint(0, height - rect_height)
    dep_sp[top_left_y:top_left_y+rect_height, top_left_x:top_left_x+rect_width] = 0
    if modal_missing == True:
        dep_sp = np.zeros_like(dep_sp)

    return dep_sp

def get_sparse_depth2(dep, num_sample,modal_missing, epoch):
    channel, height, width = dep.shape

    assert channel == 1

    idx_nnz = torch.nonzero(dep.view(-1) > 0.0001, as_tuple=False)

    num_idx = len(idx_nnz)
    idx_sample = torch.randperm(num_idx)[:num_sample]

    idx_nnz = idx_nnz[idx_sample[:]]

    mask = torch.zeros((channel*height*width))
    mask[idx_nnz] = 1.0
    mask = mask.view((channel, height, width))

    dep_sp = dep * mask.type_as(dep)

    _, height, width = dep_sp.shape
    k = math.log2(max(1, (epoch-9)))
    # print("enhancement k is:", k)
    rect_height, rect_width = int(k * height//5), int(k * width//5)
    top_left_x = random.randint(0, width - rect_width)
    top_left_y = random.randint(0, height - rect_height)
    dep_sp[top_left_y:top_left_y+rect_height, top_left_x:top_left_x+rect_width] = 0
    if modal_missing == True:
        dep_sp = np.zeros_like(dep_sp)

    return dep_sp

def position_transform(position, args):
    # s = np.random.uniform(1.0, 1.5) # random scaling
    # angle = np.random.uniform(-5.0, 5.0) # random rotation degrees
    oheight = 240
    owidth = 320

    if position is not None:
        bottom_crop_only = transforms.Compose([transforms.BottomCrop((oheight, owidth))])
        position = bottom_crop_only(position)

    return position


"""
NYUDepthV2 json file has a following format:

{
    "train": [
        {
            "filename": "train/bedroom_0078/00066.h5"
        }, ...
    ],
    "val": [
        {
            "filename": "train/study_0008/00351.h5"
        }, ...
    ],
    "test": [
        {
            "filename": "val/official/00001.h5"
        }, ...
    ]
}

Reference : https://github.com/XinJCheng/CSPN/blob/master/nyu_dataset_loader.py
"""
to_tensor = transforms.ToTensor()
to_float_tensor = lambda x: to_tensor(x).float()

class NYU(data.Dataset):
    def __init__(self, mode, args,epoch=0):
        super(NYU, self).__init__()

        self.args = args
        self.mode = mode
        

        if mode != 'train' and mode != 'val' and mode != 'test':
            raise NotImplementedError

        # For NYUDepthV2, crop size is fixed
        height, width = (240, 320)
        crop_size = (228, 304)

        self.height = height
        self.width = width
        self.crop_size = crop_size
        self.epoch = epoch
        # self.position_transform = position_transform(mode, args)
        modal_missing_rate = 0
        if random.random() < modal_missing_rate:
            self.modal_missing = True
        else:
            self.modal_missing = False

        # Camera intrinsics [fx, fy, cx, cy]
        self.K = torch.Tensor([
            5.1885790117450188e+02 / 2.0,
            5.1946961112127485e+02 / 2.0,
            3.2558244941119034e+02 / 2.0 - 8.0,
            2.5373616633400465e+02 / 2.0 - 6.0
        ])

        self.augment = True   # Data augmentation

        with open(self.args.nyu_json) as json_file:
            json_data = json.load(json_file)
            self.sample_list = json_data[mode]

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        path_file = os.path.join(self.args.dir_nyu,
                                 self.sample_list[idx]['filename'])

        f = h5py.File(path_file, 'r')
        rgb_h5 = f['rgb'][:].transpose(1, 2, 0)
        dep_h5 = f['depth'][:]

        rgb = Image.fromarray(rgb_h5, mode='RGB')
        dep = Image.fromarray(dep_h5.astype('float32'), mode='F')
        position = CoordConv.AddCoordsNp(self.height, self.width)
        position = position.call()

        if self.augment and self.mode == 'train':
            _scale = np.random.uniform(1.0, 1.5)
            scale = int(self.height * _scale)
            degree = np.random.uniform(-5.0, 5.0)
            flip = np.random.uniform(0.0, 1.0)

            if flip > 0.5:
                rgb = TF.hflip(rgb)
                dep = TF.hflip(dep)

            rgb = TF.rotate(rgb, angle=degree)
            dep = TF.rotate(dep, angle=degree)

            t_rgb = T.Compose([
                T.Resize(scale),
                T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                T.CenterCrop(self.crop_size),
                T.ToTensor(),
                T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])

            t_dep = T.Compose([
                T.Resize(scale),
                T.CenterCrop(self.crop_size),
                # self.ToNumpy(),
                T.ToTensor()
            ])

            rgb = t_rgb(rgb)
            dep = t_dep(dep)

            dep = dep / _scale

            K = self.K.clone()
            K[0] = K[0] * _scale
            K[1] = K[1] * _scale
        else:
            t_rgb = T.Compose([
                T.Resize(self.height),
                T.CenterCrop(self.crop_size),
                T.ToTensor(),
                T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])

            t_dep = T.Compose([
                T.Resize(self.height),
                T.CenterCrop(self.crop_size),
                # self.ToNumpy(),
                T.ToTensor()
            ])

            rgb = t_rgb(rgb)
            dep = t_dep(dep)

            K = self.K.clone().repeat(3,1)
        num_sample = 500
        dep_sp1 = get_sparse_depth1(dep, num_sample, self.modal_missing, self.epoch)
        dep_sp2 = get_sparse_depth2(dep, num_sample, self.modal_missing, self.epoch)

        # position = self.position_transform(position, self.args)
        

        output = {'rgb': rgb, 'd': dep_sp1, 'd2': dep_sp2, 'gt': dep, 'position': position,'K': K}

        return output

    