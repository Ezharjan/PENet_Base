#!/bin/bash


python main6.py -b 1 -n e --evaluate ../results/input=rgbd.criterion=l2.lr=0.001.bs=6.round=1.time=2024-11-11@18-49/checkpoint-30.pth.tar > ../log/val1113.txt
wait