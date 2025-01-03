# def average_weights(w):
#     """
#     Returns the average of the weights.
#     """
#     w_avg = copy.deepcopy(w[0])
#     for key in w_avg.keys():
#         for i in range(1, len(w)):
#             w_avg[key] += w[i][key]
#         w_avg[key] = torch.div(w_avg[key], len(w))
#     return w_avg

# elif args.resume:  # optionally resume from a checkpoint
#     args_new = args
#     if os.path.isfile(args.resume):
#         print("=> loading checkpoint '{}' ... ".format(args.resume),
#               end='')
#         checkpoint = torch.load(args.resume, map_location=device)

#         args.start_epoch = checkpoint['epoch'] + 1
#         args.data_folder = args_new.data_folder
#         args.val = args_new.val
#         print("Completed. Resuming from epoch {}.".format(
#             checkpoint['epoch']))
#     else:
#         print("No checkpoint found at '{}'".format(args.resume))
#         return

import torch

# 假设 model 是你的模型，可以根据实际情况替换为具体的模型实例化代码
# model = ...  # 这里应该是你的模型定义，例如 model = MyModel()

# 加载两个 checkpoint 文件
checkpoint1 = torch.load('./ckpt/checkpoint-8.pth.tar')
checkpoint2 = torch.load('./ckpt/checkpoint-8.pth(1).tar')

print(checkpoint1.keys())
# 提取网络参数
state_dict1 = checkpoint1['model']
state_dict2 = checkpoint2['model']

# 初始化新的状态字典
averaged_state_dict = {}

# 对每个参数键进行遍历并计算平均值
for key in state_dict1:
    # 确保两个 state_dict 中都包含相同的键
    if key in state_dict2:
        # 取平均值
        averaged_state_dict[key] = (state_dict1[key] + state_dict2[key]) / 2
    else:
        raise KeyError(f"Key {key} not found in both checkpoints")

# 将平均后的状态字典加载到模型中
# model.load_state_dict(averaged_state_dict)

# 保存新的 checkpoint 文件
torch.save({
    'state_dict': averaged_state_dict
}, '3.pth.tar')

