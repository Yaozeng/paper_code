import sys
import os.path
import argparse
import numpy as np
import scipy.io
import pdb
import h5py
import json
import re
import math
import pickle as cPickle


predict_file = json.load(open('/data1/YZ/new_model/cocoresults2/val_baseline17_1280_epoch8.json', 'r'))
gt_file= cPickle.load(open("/data1/YZ/new_model/data/cache/coco_val_target.pkl", 'rb'))
label2ans=cPickle.load(open("/data1/YZ/new_model/data/cache/coco_label2ans.pkl", 'rb'))



# calculate the accuracy for each type:
acc = 0
acc0 = 0
acc1 = 0
acc2 = 0
acc3 = 0

count = 0
count0 = 0
count1 = 0
count2 = 0
count3 = 0
f1 = open('gt_ans_save2.txt', 'w')
f2 = open('pd_ans_save2.txt', 'w')

for i in range(len(predict_file)):
    pre_id = predict_file[i]['question_id']
    gt_id = gt_file[i]['question_id']

    if not pre_id == gt_id:
        raise AssertionError()

    pre_ans = predict_file[i]['answer'].lower()
    gt_ans = label2ans[gt_file[i]['labels']].lower()

    ques_type = int(gt_file[i]['types'])
    if pre_ans == gt_ans:
        acc += 1
    count += 1
    if ques_type == 0:
        if pre_ans == gt_ans:
            acc0 += 1
        count0 += 1
    elif ques_type == 1:
        if pre_ans == gt_ans:
            acc1 += 1
        count1 += 1
    elif ques_type == 2:
        if pre_ans == gt_ans:
            acc2 += 1
        count2 += 1
    elif ques_type == 3:
        if pre_ans == gt_ans:
            acc3 += 1
        count3 += 1


    # write the gt and answer
    f1.write(gt_ans + '\n')
    f2.write(pre_ans + '\n')
print(acc0/count0)
print(acc1/count1)
print(acc2/count2)
print(acc3/count3)
print(acc/count)