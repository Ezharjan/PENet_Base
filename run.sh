#!/bin/bash


CUDA_VISIBLE_DEVICES="0,1" python main6.py -b 6 -n e  --resume ../results/input=rgbd.criterion=l2.lr=0.001.bs=6.round=1.time=2024-12-30@16-17/checkpoint-4.pth.tar > ../log/siamout_0103_50.txt
wait
