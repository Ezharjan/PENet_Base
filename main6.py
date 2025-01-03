import argparse
import os

import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import time

from dataloaders.kitti_loader4 import load_calib, input_options, KittiDepth
from dataloaders.nyu_loader import NYU
from metrics import AverageMeter, Result
import criteria
import helper
import vis_utils

from model4 import ECLNet
from model4 import PECLNet, PECLNet_train
# from model import PENet_C1_train
# from model import PENet_C2_train
# #from model import PENet_C4_train (Not Implemented)
# from model import PENet_C1
# from model import PENet_C2
# from model import PENet_C4

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
                    default=60,
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
parser.add_argument('--data', default='kitti', type=str,choices=['kitti', 'nyu'],
                    help='choose a dataset: kitti or nyu')

args = parser.parse_args()
args.result = os.path.join('..', 'results')
args.nyu_json = 'dataloaders/nyu.json'
args.dir_nyu = '../nyudepthv2/'
args.use_rgb = ('rgb' in args.input)
args.use_d = 'd' in args.input
args.use_g = 'g' in args.input
args.val_h = 352
args.val_w = 1216
args.part =50
args.modal_missing_rate = 0.2
print(args)

cuda = torch.cuda.is_available() and not args.cpu
if cuda:
    import torch.backends.cudnn as cudnn
    cudnn.benchmark = True
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print("=> using '{}' for computation.".format(device))

if args.data == 'kitti':
    dataLoad = KittiDepth
elif args.data == 'nyu':
    dataLoad = NYU

# define loss functions
depth_criterion = criteria.Huber_Combine() 
# cl_criterion = criteria.info_nce_loss_depth_maps() 
MaskedMSELoss = criteria.MaskedMSELoss()

scaler = torch.cuda.amp.GradScaler()  #
ConsineSimilarity = torch.nn.CosineSimilarity(dim=1).cuda()

#multi batch
multi_batch_size = 1
def iterate(mode, args, loader, model, optimizer, logger, epoch):
    actual_epoch = epoch - args.start_epoch + args.start_epoch_bias

    block_average_meter = AverageMeter()
    block_average_meter.reset(False)
    average_meter = AverageMeter()
    meters = [block_average_meter, average_meter]

    # switch to appropriate mode
    assert mode in ["train", "val", "eval", "test_prediction", "test_completion"], \
        "unsupported mode: {}".format(mode)
    if mode == 'train':
        model.train()
        lr = helper.adjust_learning_rate(args.lr, optimizer, actual_epoch, args)
    else:
        model.eval()
        lr = 0

    torch.cuda.empty_cache()
    total_batches = len(loader)
    stop_k = 1
    if epoch in range(0, 2):
        stop_k = 0.6
    elif epoch in range(2, 4):
        stop_k = 0.7
    elif epoch in range(4, 6):
        stop_k = 0.8
    elif epoch in range(6, 8):
        stop_k = 0.9
    else:
        stop_k = 1
    for i, batch_data in enumerate(loader):
        if i >= stop_k * total_batches:
            print("Reach the limit of {} total data, break training".format(stop_k))
            break
        dstart = time.time()
        batch_data = {
            key: val.to(device)
            for key, val in batch_data.items() if val is not None
        }            

        gt = batch_data[
            'gt'] if mode != 'test_prediction' and mode != 'test_completion' else None
        data_time = time.time() - dstart

        pred = None
        start = None
        gpu_time = 0

        #start = time.time()
        #pred = model(batch_data)
        #gpu_time = time.time() - start

        #'''
        if(args.network_model == 'e'):
            start = time.time()
            with torch.cuda.amp.autocast():
                output1, output2, pred = model(batch_data) # z, p1, p2
        else:
            start = time.time()
            pred = model(batch_data)

        if(args.evaluate):
            gpu_time = time.time() - start
        #'''

        depth_loss, photometric_loss, smooth_loss, mask = 0, 0, 0, None

        # inter loss_param
        cl_loss, loss = 0, 0

        # round1, round2, round3 = 0, 0, None   # 1, 3, None
        # if(actual_epoch <= round1):
        #     w_st1, w_st2 = 0.2, 0.2
        # elif(actual_epoch <= round2):
        #     w_st1, w_st2 = 0.05, 0.05
        # else:
        #     w_st1, w_st2 = 0, 0


        if mode == 'train':
            # Loss 1: the direct depth supervision from ground truth label
            # mask=1 indicates that a pixel does not ground truth labels
            if args.network_model == 'e':
                l1 = 0.8
                l2 = 1-l1
                cl_loss = (MaskedMSELoss(pred, output1.detach()).mean() + MaskedMSELoss(pred, output2.detach()).mean()) * 0.5
                depth_loss = depth_criterion(pred, gt)
                Loss = l1* depth_loss + l2* cl_loss
                scaler.scale(Loss).backward()

                #optimizer.step()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                print("depth_loss:", depth_loss, "cl_loss:",cl_loss ,"epoch:", epoch, " ", i, "/", len(loader) )
            else:
                depth_loss = MaskedMSELoss(pred, gt)
                depth_loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                print("PEdepth_loss:", depth_loss, "epoch:", epoch, " ", i, "/", len(loader) )

        if mode == "test_completion":
            str_i = str(i)
            path_i = str_i.zfill(10) + '.png'
            path = os.path.join(args.data_folder_save, path_i)
            vis_utils.save_depth_as_uint16png_upload(pred, path)

        if(not args.evaluate):
            gpu_time = time.time() - start
        # measure accuracy and record loss
        with torch.no_grad():
            mini_batch_size = next(iter(batch_data.values())).size(0)
            result = Result()
            if mode != 'test_prediction' and mode != 'test_completion':
                result.evaluate(pred.data, gt.data, photometric_loss)
                [
                    m.update(result, gpu_time, data_time, mini_batch_size)
                    for m in meters
                ]

                if mode != 'train':
                    logger.conditional_print(mode, i, epoch, lr, len(loader),
                                     block_average_meter, average_meter)
                logger.conditional_save_img_comparison(mode, i, batch_data, pred,
                                                   epoch)
                logger.conditional_save_pred(mode, i, pred, epoch)

    avg = logger.conditional_save_info(mode, average_meter, epoch)
    is_best = logger.rank_conditional_save_best(mode, avg, epoch)
    if is_best and not (mode == "train"):
        logger.save_img_comparison_as_best(mode, epoch)
    logger.conditional_summarize(mode, avg, is_best)

    return avg, is_best

def main():
    global args
    checkpoint = None
    is_eval = False
    if args.evaluate:
        args_new = args
        if os.path.isfile(args.evaluate):
            print("=> loading checkpoint '{}' ... ".format(args.evaluate),
                  end='')
            checkpoint = torch.load(args.evaluate, map_location=device)
            #args = checkpoint['args']
            args.start_epoch = checkpoint['epoch'] + 1
            args.data_folder = args_new.data_folder
            args.val = args_new.val
            is_eval = True

            print("Completed.")
        else:
            is_eval = True
            print("No model found at '{}'".format(args.evaluate))
            #return

    elif args.resume:  # optionally resume from a checkpoint
        args_new = args
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}' ... ".format(args.resume),
                  end='')
            checkpoint = torch.load(args.resume, map_location=device)

            args.start_epoch = checkpoint['epoch'] + 1
            args.data_folder = args_new.data_folder
            args.val = args_new.val
            print("Completed. Resuming from epoch {}.".format(
                checkpoint['epoch']))
        else:
            print("No checkpoint found at '{}'".format(args.resume))
            return

    print("=> creating model and optimizer ... ", end='')
    model = None
    penet_accelerated = False
    if (args.network_model == 'e'):
        model = ECLNet(args).to(device)
    if (args.network_model == 'pe'):
        if (is_eval == False):
            model = PECLNet_train(args).to(device)
        else:
            model = PECLNet(args).to(device)
            penet_accelerated = True
    
    if (penet_accelerated == True):
        model.encoder3.requires_grad = False
        model.encoder5.requires_grad = False
        model.encoder7.requires_grad = False

    model_named_params = None
    model_bone_params = None
    model_new_params = None
    optimizer = None

    if checkpoint is not None:
        #print(checkpoint.keys())
        if (args.freeze_backbone == True):
            model.backbone.load_state_dict(checkpoint['model'])
        else:
            model.load_state_dict(checkpoint['model'], strict=False)
        #optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> checkpoint state loaded.")

    logger = helper.logger(args)
    if checkpoint is not None:
        logger.best_result = checkpoint['best_result']
        del checkpoint
    print("=> logger created.")

    test_dataset = None
    test_loader = None
    if (args.test):
        test_dataset = dataLoad('test_completion', args)
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=1,
            pin_memory=True)
        iterate("test_completion", args, test_loader, model, None, logger, 0)
        return

    val_dataset = dataLoad('val', args)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        pin_memory=True)  # set batch size to be 1 for validation
    print("\t==> val_loader size:{}".format(len(val_loader)))

    if is_eval == True:
        for p in model.parameters():
            p.requires_grad = False

        result, is_best = iterate("val", args, val_loader, model, None, logger,
                              args.start_epoch - 1)
        return

    if (args.freeze_backbone == True):
        for p in model.backbone.parameters():
            p.requires_grad = False
        model_named_params = [
            p for _, p in model.named_parameters() if p.requires_grad
        ]
        optimizer = torch.optim.Adam(model_named_params, lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.99))
    elif (args.network_model == 'pe'):
        model_bone_params = [
            p for _, p in model.backbone.named_parameters() if p.requires_grad
        ]
        model_new_params = [
            p for _, p in model.named_parameters() if p.requires_grad
        ]
        model_new_params = list(set(model_new_params) - set(model_bone_params))
        optimizer = torch.optim.Adam([{'params': model_bone_params, 'lr': args.lr / 10}, {'params': model_new_params}],
                                     lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.99))
    else:
        model_named_params = [
            p for _, p in model.named_parameters() if p.requires_grad
        ]
        optimizer = torch.optim.Adam(model_named_params, lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.99))
    print("completed.")

    model = torch.nn.DataParallel(model)

    # Data loading code
    print("=> creating data loaders ... ")
    if not is_eval:
        train_dataset = dataLoad('train', args)
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=args.batch_size,
                                                   shuffle=True,
                                                   num_workers=args.workers,
                                                   pin_memory=True,
                                                   sampler=None)
        print("\t==> train_loader size:{}".format(len(train_loader)))

    print("=> starting main loop ...")
    for epoch in range(args.start_epoch, args.epochs):
        print("=> starting training epoch {} ..".format(epoch))
        iterate("train", args, train_loader, model, optimizer, logger, epoch)  # train for one epoch

        # validation memory reset
        for p in model.parameters():
            p.requires_grad = False
        result, is_best = iterate("val", args, val_loader, model, None, logger, epoch)  # evaluate on validation set

        for p in model.parameters():
            p.requires_grad = True
        if (args.freeze_backbone == True):
            for p in model.module.backbone.parameters():
                p.requires_grad = False

        helper.save_checkpoint({ # save checkpoint
            'epoch': epoch,
            'model': model.module.state_dict(),
            'best_result': logger.best_result,
            'optimizer' : optimizer.state_dict(),
            'args' : args,
        }, is_best, epoch, logger.output_directory)


if __name__ == '__main__':
    main()