"""
This code is modified from Hengyuan Hu's repository.
https://github.com/hengyuan-hu/bottom-up-attention-vqa
"""
import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
import numpy as np

from dataset import Dictionary, VQAFeatureDataset, VisualGenomeFeatureDataset,COCOFeatureDataset
import base_model
from train import *
from train_coco import train_coco
from utils import *
from utils import trim_collate
from dataset import tfidf_from_questions
os.environ['CUDA_VISIBLE_DEVICES']="0,1"
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_hid', type=int, default=1280)
    parser.add_argument('--model', type=str, default='baseline17')
    parser.add_argument('--use_both', type=bool, default=False)
    parser.add_argument('--output', type=str, default='saved_models/coco')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--seed', type=int, default=1204, help='random seed')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    utils.create_dir(args.output)
    logger = utils.Logger(os.path.join(args.output, 'log.txt'))
    logger.write(args.__repr__())

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    batch_size = args.batch_size

    dictionary = Dictionary.load_from_file('data/cocodictionary.pkl')
    train_dset = COCOFeatureDataset('train', dictionary, adaptive=False)
    val_dset = COCOFeatureDataset('val', dictionary, adaptive=False)

    if args.use_both:
        trainval_dset = ConcatDataset([train_dset, val_dset])
        train_loader = DataLoader(trainval_dset, batch_size, shuffle=True, num_workers=1, collate_fn=utils.trim_collate)
        eval_loader = None
    else:
        train_loader = DataLoader(train_dset, batch_size, shuffle=True, num_workers=1, collate_fn=utils.trim_collate)
        eval_loader = DataLoader(val_dset, batch_size, shuffle=False, num_workers=1, collate_fn=utils.trim_collate)

    constructor = 'build_%s' % args.model
    model = getattr(base_model, constructor)(train_dset, args.num_hid).cuda()
    model.apply(weights_init)

    tfidf = None
    weights = None
    model.w_emb.init_embedding('data/cocoglove6b_init_300d.npy', tfidf, weights)

    model = nn.DataParallel(model).cuda()

    optim = None
    #train_loader = DataLoader(train_dset, batch_size, shuffle=True, num_workers=1, collate_fn=utils.trim_collate)
    #eval_loader = DataLoader(val_dset, batch_size, shuffle=False, num_workers=1, collate_fn=utils.trim_collate)
    #train(model, train_loader, eval_loader, args.output)
    #train2(model, train_loader, eval_loader, 13, args.output, opt=None, s_epoch=0)
    #train4(model, train_loader, eval_loader, 12, args.output, s_epoch=0)
    #train5(model, train_loader, eval_loader, 12, args.output, s_epoch=0)
    #train6(model, train_loader, eval_loader, 12, args.output, s_epoch=0)
    #train7(model, train_loader, eval_loader, 12, args.output)
    train_coco(model, train_loader, eval_loader, 12, args.output, s_epoch=0)
