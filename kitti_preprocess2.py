import argparse
import os
import copy
import random
import numpy as np
from PIL import Image
import cv2

import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import time
from torch.utils.data import DataLoader

from dataloaders.kitti_loader2 import load_calib, input_options, KittiDepth
from metrics import AverageMeter, Result
import criteria
import helper
import vis_utils

from model import ENet
from model import PENet_C1_train
from model import PENet_C2_train
#from model import PENet_C4_train (Not Implemented)
from model import PENet_C1
from model import PENet_C2
from model import PENet_C4

# use our model to regenerate depth map for train when simulating depth lost
# the regenerated map will be saved to local

parser = argparse.ArgumentParser(description='Sparse-to-Dense')
parser.add_argument('-n',
                    '--network-model',
                    type=str,
                    default="e",
                    choices=["e", "pe"],
                    help='choose a model: enet or penet'
                    )
parser.add_argument('--workers',
                    default=4,
                    type=int,
                    metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs',
                    default=100,
                    type=int,
                    metavar='N',
                    help='number of total epochs to run (default: 100)')
parser.add_argument('--start-epoch',
                    default=0,
                    type=int,
                    metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--start-epoch-bias',
                    default=0,
                    type=int,
                    metavar='N',
                    help='manual epoch number bias(useful on restarts)')
parser.add_argument('-c',
                    '--criterion',
                    metavar='LOSS',
                    default='l2',
                    choices=criteria.loss_names,
                    help='loss function: | '.join(criteria.loss_names) +
                    ' (default: l2)')
parser.add_argument('-b',
                    '--batch-size',
                    default=1,
                    type=int,
                    help='mini-batch size (default: 1)')
parser.add_argument('--lr',
                    '--learning-rate',
                    default=1e-3,
                    type=float,
                    metavar='LR',
                    help='initial learning rate (default 1e-5)')
parser.add_argument('--weight-decay',
                    '--wd',
                    default=1e-6,
                    type=float,
                    metavar='W',
                    help='weight decay (default: 0)')
parser.add_argument('--print-freq',
                    '-p',
                    default=10,
                    type=int,
                    metavar='N',
                    help='print frequency (default: 10)')
parser.add_argument('--resume',
                    default='',
                    type=str,
                    metavar='PATH',
                    help='path to latest checkpoint (default: none)')
########################
parser.add_argument('--round',
                    type=int,
                    default = 1,
                    help="simulate which client to train(from 1 to 10)"
                    )
#################
parser.add_argument('--data-folder',
                    default='../dataset/KITTI/kitti_depth/depth',
                    type=str,
                    metavar='PATH',
                    help='data folder (default: none)')
parser.add_argument('--data-folder-rgb',
                    default='../dataset/KITTI/kitti_raw',
                    type=str,
                    metavar='PATH',
                    help='data folder rgb (default: none)')
parser.add_argument('--data-folder-save',
                    default='../dataset/KITTI/kitti_depth/submit_test/',
                    type=str,
                    metavar='PATH',
                    help='data folder test results(default: none)')
parser.add_argument('-i',
                    '--input',
                    type=str,
                    default='rgbd',
                    choices=input_options,
                    help='input: | '.join(input_options))
parser.add_argument('--val',
                    type=str,
                    default="select",
                    choices=["select", "full"],
                    help='full or select validation set')
parser.add_argument('--jitter',
                    type=float,
                    default=0.1,
                    help='color jitter for images')
parser.add_argument('--rank-metric',
                    type=str,
                    default='rmse',
                    choices=[m for m in dir(Result()) if not m.startswith('_')],
                    help='metrics for which best result is saved')

parser.add_argument('-e', '--evaluate', default='', type=str, metavar='PATH')
parser.add_argument('-f', '--freeze-backbone', action="store_true", default=False,
                    help='freeze parameters in backbone')
parser.add_argument('--test', action="store_true", default=False,
                    help='save result kitti test dataset for submission')
parser.add_argument('--cpu', action="store_true", default=False, help='run on cpu')

#random cropping
parser.add_argument('--not-random-crop', action="store_true", default=False,
                    help='prohibit random cropping')
parser.add_argument('-he', '--random-crop-height', default=320, type=int, metavar='N',
                    help='random crop height')
parser.add_argument('-w', '--random-crop-width', default=1216, type=int, metavar='N',
                    help='random crop height')

#geometric encoding
parser.add_argument('-co', '--convolutional-layer-encoding', default="xyz", type=str,
                    choices=["std", "z", "uv", "xyz"],
                    help='information concatenated in encoder convolutional layers')

#dilated rate of DA-CSPN++
parser.add_argument('-d', '--dilation-rate', default="2", type=int,
                    choices=[1, 2, 4],
                    help='CSPN++ dilation rate')

args = parser.parse_args()
args.result = os.path.join('..', 'results')
args.use_rgb = ('rgb' in args.input)
args.use_d = 'd' in args.input
args.use_g = 'g' in args.input
args.val_h = 352
args.val_w = 1216
# args.resume = '../results/standard/checkpoint-10.pth.tar'
args.resume = '../basement/model_best.pth.tar'
print(args)
args.rgb_lost = False
args.d_lost = False
args.network_model = 'e'

cuda = torch.cuda.is_available() and not args.cpu
if cuda:
    import torch.backends.cudnn as cudnn
    cudnn.benchmark = True
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print("=> using '{}' for computation.".format(device))

checkpoint = torch.load(args.resume, map_location=device)
global_weights = checkpoint['model']
model = ENet(args).to(device)
model.eval()
try:
    model.load_state_dict(global_weights)
    print("Global weights loaded successfully!")
except RuntimeError as e:
    print(f"Model Fail to Load: {e}")

###
# dataset = KittiDepth(split='train', args=args)
# dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

def save_prediction(image, path):
    image = image.squeeze().cpu().numpy()
    image = (image * 256).astype('uint16')   # 6.22 added
    # new_size = (375, 1242)
    # new_size = (1242, 375)  # for cv2
    # image = cv2.resize(image, new_size) # 375, 1242
    # image = makeSparse(image)
    os.makedirs(os.path.dirname(path), exist_ok = True)
    Image.fromarray(image).save(path)

def makeSparse(co):
    co[co <1000] = 0
    cnt_co = np.count_nonzero(co)
    index_co = np.nonzero(co)
    # print(index_co)
    out_cnt = int(cnt_co * 0.91)
    random_indices = np.random.choice(cnt_co, out_cnt, replace = False)
    co[index_co[0][random_indices],index_co[1][random_indices]] =0
    return co
    
for user in range(1,21):
    args.round = user
    dataset = KittiDepth(split='val', args=args)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    with torch.no_grad():
        for idx, item in enumerate(dataloader):
            inputs = {key: val.to(device) for key, val in item.items() if val is not None}
            _,_, pred = model(inputs)
            
            sparse_path = dataset.paths['d'][idx]
            
            save_path = sparse_path.replace('/KITTI/','/KITTI5/')
            print(save_path)
            
            save_prediction(pred, save_path)
