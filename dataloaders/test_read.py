import os
import glob

paths_with_titles =[]
glob_d = []
glob_gt = []
rgb = []
round =1 
split = 'train'
with open(f"selected_part_{round}.txt","r")as file:
    for line in file:
        line = line.strip()
        if line.endswith(':'):
            current_title = line[0:-1]
        elif line:
            paths_with_titles.append((current_title, line))
for title, path in paths_with_titles:
    if title == 'd':
        glob_d.append(path)
    if title == 'gt':
        glob_gt.append(path)
    if title == 'rgb':
        rgb.append(path)
def get_rgb_paths(p):
    return rgb
assert glob_d, "Fail to open selected_part txt file. Please check your code!!!!!"
if glob_gt is not None and split !='train':
        # train or val-full or val-select
        paths_d = sorted(glob.glob(glob_d))
        paths_gt = sorted(glob.glob(glob_gt))
        paths_rgb = [get_rgb_paths(p) for p in paths_gt]
elif glob_gt is not None and split =='train':
    paths_d = glob_d
    paths_gt = glob_gt
    paths_rgb = get_rgb_paths(1)
else:
    # test only has d or rgb
    paths_rgb = sorted(glob.glob(glob_rgb))
    paths_gt = [None] * len(paths_rgb)
    if split == "test_prediction":
        paths_d = [None] * len(
            paths_rgb)  # test_prediction has no sparse depth
    else:
        paths_d = sorted(glob.glob(glob_d))
    
    
    
    
    
    
print(len(paths_d))
print(len(paths_gt))
print(len(paths_rgb))
# print(paths_d)