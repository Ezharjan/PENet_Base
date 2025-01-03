import os
import glob
import numpy as np
from random import sample
from PIL import Image
import torch
import torch.utils.data as data
from dataloaders import transforms
import CoordConv

class KittiDepth(data.Dataset):
    def __init__(self, split, args, userNum):
        self.args = args
        self.split = split
        paths, transform = get_paths_and_transform(split, args)
        self.all_parts = self.divide_into_parts(paths)  # Stores all parts for debugging
        # self.selected_parts = self.select_random_parts(num_parts=userNum)  # 只存储选中的部分
        # self.paths = self.select_random_parts(num_parts=userNum)
        self.paths = self.all_parts
        self.transform = transform
        self.K = load_calib()

    def divide_into_parts(self, paths):
        total_len = len(paths['rgb'])
        part_size = total_len // 10
        indices = np.arange(total_len)
        parts_indices = [indices[i * part_size: (i + 1) * part_size] for i in range(10)]
        parts = [{key: [paths[key][idx] for idx in part_indices] for key in paths}
                 for part_indices in parts_indices]
        return parts
    
    def select_random_parts(self, num_parts):        # 随机选择指定数量的部分        
        return sample(self.all_parts, num_parts)

    # def select_random_parts(self, num_parts):
    #     selected_parts = sample(self.all_parts, num_parts)
    #     selected_paths = {key: [] for key in self.all_parts[0]}
    #     for part in selected_parts:
    #         for key in part:
    #             selected_paths[key].extend(part[key])
    #     return selected_paths

    def __getraw__(self, index):
        rgb = rgb_read(self.paths['rgb'][index]) if self.paths['rgb'][index] is not None else None
        sparse = depth_read(self.paths['d'][index]) if self.paths['d'][index] is not None else None
        target = depth_read(self.paths['gt'][index]) if self.paths['gt'][index] is not None else None
        return rgb, sparse, target

    def __getitem__(self, index):
        rgb, sparse, target = self.__getraw__(index)
        position = CoordConv.AddCoordsNp(self.args.val_h, self.args.val_w).call()
        rgb, sparse, target, position = self.transform(rgb, sparse, target, position, self.args)
        rgb, gray = handle_gray(rgb, self.args)
        candidates = {"rgb": rgb, "d": sparse, "gt": target, "g": gray, 'position': position, 'K': self.K}
        items = {key: to_float_tensor(val) for key, val in candidates.items() if val is not None}
        return items

    def __len__(self):
        return len(self.paths['gt'])

def save_parts_to_files(parts):
    for i, part in enumerate(parts):
        with open(f"part_{i+1}.txt", "w") as f:
            for key in part:
                f.write(f"{key.upper()} FILES:\n")
                f.writelines(f"{filename}\n" for filename in part[key])
                f.write("\n")

def main():
    args = YourArgumentsHere()
    split = 'train'
    userNum = 3
    dataset = KittiDepth(split, args, userNum)

    # Saving each part's files to separate text files for debugging
    save_parts_to_files(dataset.all_parts)

if __name__ == "__main__":
    main()
