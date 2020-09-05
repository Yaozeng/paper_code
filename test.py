"""
This code is modified from Hengyuan Hu's repository.
https://github.com/hengyuan-hu/bottom-up-attention-vqa
"""
import argparse
import json
#import progressbar
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
import numpy as np

from dataset import Dictionary, VQAFeatureDataset
import base_model
import utils
import csv, sys, base64, copy, os
import scipy.misc
import matplotlib
from tqdm import tqdm

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.patches as patches
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageEnhance


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_hid', type=int, default=1280)
    parser.add_argument('--model', type=str, default='baseline0')
    parser.add_argument('--model2', type=str, default='baseline17')
    parser.add_argument('--split', type=str, default='val')
    parser.add_argument('--input', type=str, default='saved_models/origin')
    parser.add_argument('--input2', type=str, default='saved_models/exp10')
    parser.add_argument('--output', type=str, default='results1')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--debug', type=bool, default=True)
    parser.add_argument('--logits', type=bool, default=False)
    parser.add_argument('--index', type=int, default=0)
    parser.add_argument('--epoch', type=int, default=11)
    args = parser.parse_args()
    return args
def get_question(q, dataloader):
    str = []
    dictionary = dataloader.dataset.dictionary
    for i in range(q.size(0)):
        str.append(dictionary.idx2word[q[i]] if q[i] < len(dictionary.idx2word) else '_')
    return ' '.join(str)


def get_answer(p, dataloader):
    _m, idx = p.max(0)
    return dataloader.dataset.label2ans[idx[0]]

def get_logits(model, dataloader):
    N = len(dataloader.dataset)
    M = dataloader.dataset.num_ans_candidates
    pred = torch.FloatTensor(N, M).zero_()
    qIds = torch.IntTensor(N).zero_()
    image_ids = torch.IntTensor(N).zero_()
    idx = 0
    #bar = progressbar.ProgressBar(max_value=N)
    for v, b, q, i ,image_id in iter(dataloader):
        #bar.update(idx)
        batch_size = v.size(0)
        v = Variable(v, volatile=True).cuda()
        b = Variable(b, volatile=True).cuda()
        q = Variable(q, volatile=True).cuda()
        logits,att1,att2= model(v, b, q, None)
        pred[idx:idx+batch_size,:].copy_(logits.data)
        qIds[idx:idx+batch_size].copy_(i)
        idx += batch_size
        if args.debug:
            for j in range(0,batch_size,20):
                plt.figure(1)
                plt.clf()
                plt.text(0.0,0.5,get_question(q.data[j], dataloader))
                plt.axis('off')
                plt.savefig("visualize/{}_q.jpg".format(i[j]))
                plt.clf()
                plt.text(0.5,0.5,get_answer(logits.data[j], dataloader))
                plt.axis('off')
                plt.savefig("visualize/{}_a.jpg".format(i[j]))
                plt.clf()
                # print(b.data[5])
                # print(att1.data[0])
                # print(image_id[0])
                img = Image.open("data/test2015/COCO_test2015_{}.jpg".format(str(image_id[j]).zfill(12)))
                w, h = img.size
                # raw_img = load_image_into_numpy_array(img)
                # img = plt.imread("data/test2015/COCO_test2015_{}.jpg".format(str(image_id[0]).zfill(12)))
                # image=Image.fromarray(raw_img)
                draw = ImageDraw.Draw(img)
                #print(att1.data[4].cpu().numpy().squeeze())
                att1_np=att1.data[j].cpu().numpy().squeeze()
                index_sort = np.argsort(att1_np)[::-1][:2]
                axis=np.arange(36)
                attN1=att1_np[index_sort]
                plt.bar(axis,att1_np)
                plt.savefig("visualize/{}_att.jpg".format(i[j]))
                plt.clf()
                #print(index_sort)
                #print(b.data[4].cpu().numpy())
                boxesN = b.data[j].cpu().numpy()[index_sort]
                #print(boxesN)
                # boxesN = [list(map(int, box)) for box in boxesN]
                # print(boxesN)
                for ii,e in enumerate(boxesN):
                    # print(e[0])
                    # print(e[1])
                    # print(e[2])
                    # print(e[3])
                    (left, right, top, bottom) = (e[0] * w, e[2] * w,
                                                  e[1] * h, e[3] * h)
                    # print(left)
                    # print(right)
                    # print(top)
                    # print(bottom)
                    draw.line([(left, top), (left, bottom), (right, bottom),
                               (right, top), (left, top)], width=4, fill='red')
                    try:
                        font = ImageFont.truetype('font/arial.ttf', 30)
                    except IOError:
                        font = ImageFont.load_default()

                        # If the total height of the display strings added to the top of the bounding
                        # box exceeds the top of the image, stack the strings below the bounding box
                        # instead of above.
                    display_str_heights = [font.getsize(str(round(ds,2)))[1] for ds in attN1]
                    # Each display_str has a top and bottom margin of 0.05x.
                    total_display_str_height = (1 + 2 * 0.05) * sum(display_str_heights)

                    if top > total_display_str_height:
                        text_bottom = top
                    else:
                        text_bottom = bottom + total_display_str_height
                    # Reverse list and print from bottom to top.
                    text_width, text_height = font.getsize(str(round(attN1[ii],2)))
                    margin = np.ceil(0.05 * text_height)
                    draw.rectangle([(left, text_bottom - text_height - 2 * margin), (left + text_width,
                                                                              text_bottom)],
                            fill='red')
                    draw.text(
                            (left + margin, text_bottom - text_height - margin),
                            str(round(attN1[ii], 2)),
                            fill='black',
                            font=font)
                plt.imshow(np.array(img.convert('RGB')))
                plt.axis('off')
                plt.savefig("visualize/{}_img.jpg".format(i[j]))
                plt.clf()

                img = Image.open("data/test2015/COCO_test2015_{}.jpg".format(str(image_id[j]).zfill(12)))
                draw = ImageDraw.Draw(img)
                att2_np=att2.data[j].cpu().numpy().squeeze()
                index_sort2 = np.argsort(att2_np)[::-1][:2]
                attN2 = att2_np[index_sort2]
                plt.bar(axis,att2_np)
                plt.savefig("visualize/{}_att2.jpg".format(i[j]))
                plt.clf()
                boxesN2 = b.data[j].cpu().numpy()[index_sort2]
                # boxesN2 = [list(map(int, box)) for box in boxesN2]
                for ii,e in enumerate(boxesN2):
                    # print(e[0])
                    # print(e[1])
                    # print(e[2])
                    # print(e[3])
                    (left, right, top, bottom) = (e[0] * w, e[2] * w,
                                                  e[1] * h, e[3] * h)
                    # print(left)
                    # print(right)
                    # print(top)
                    # print(bottom)
                    draw.line([(left, top), (left, bottom), (right, bottom),
                               (right, top), (left, top)], width=4, fill='red')
                    try:
                        font = ImageFont.truetype('font/arial.ttf', 30)
                    except IOError:
                        font = ImageFont.load_default()

                        # If the total height of the display strings added to the top of the bounding
                        # box exceeds the top of the image, stack the strings below the bounding box
                        # instead of above.
                    display_str_heights = [font.getsize(str(round(ds,2)))[1] for ds in attN2]
                    # Each display_str has a top and bottom margin of 0.05x.
                    total_display_str_height = (1 + 2 * 0.05) * sum(display_str_heights)

                    if top > total_display_str_height:
                        text_bottom = top
                    else:
                        text_bottom = bottom + total_display_str_height
                    # Reverse list and print from bottom to top.
                    text_width, text_height = font.getsize(str(round(attN2[ii],2)))
                    margin = np.ceil(0.05 * text_height)
                    draw.rectangle([(left, text_bottom - text_height - 2 * margin), (left + text_width,
                                                                              text_bottom)],
                            fill='red')
                    draw.text(
                            (left + margin, text_bottom - text_height - margin),
                            str(round(attN2[ii], 2)),
                            fill='black',
                            font=font)
                plt.imshow(np.array(img.convert('RGB')))
                plt.axis('off')
                # plt.imsave("test.jpg",img)
                #plt.savefig("visualize/{}.jpg".format(i[j]))
                plt.savefig("visualize/{}_img2.jpg".format(i[j]))
                plt.clf()
                #bar.update(idx)
        break
    return pred, qIds,image_ids
def get_logits2(model, model2,dataloader):
    #N = len(dataloader.dataset)
    #M = dataloader.dataset.num_ans_candidates
    #pred = torch.FloatTensor(N, M).zero_()
    #qIds = torch.IntTensor(N).zero_()
    #image_ids = torch.IntTensor(N).zero_()
    idx = 0
    #bar = progressbar.ProgressBar(max_value=N)
    for v, b, q, i ,question_id,image_id in iter(dataloader):
        #bar.update(idx)
        batch_size = v.size(0)
        v = Variable(v, volatile=True).cuda()
        b = Variable(b, volatile=True).cuda()
        q = Variable(q, volatile=True).cuda()
        #logits,att= model(v, b, q, None)
       # logits1, att1, att2 = model2(v, b, q, None)
        #pred[idx:idx+batch_size,:].copy_(logits.data)
        #qIds[idx:idx+batch_size].copy_(i)
        idx += batch_size
        if args.debug:
            for j in range(0,batch_size,100):
                print(question_id[j])
                print(image_id[j])
"""
                m1, ans1 = logits.data[j].max(0)
                m2, ans2= i[j].cuda().max(0)
                m3, ans3 = logits1.data[j].max(0)
                if (ans1[0] != ans2[0]) and (ans3[0] == ans2[0]):
                    plt.figure(1)
                    plt.clf()
                    plt.text(0.0, 0.5, get_question(q.data[j], dataloader))
                    plt.axis('off')
                    plt.savefig("visualize2/{}_q.jpg".format(question_id[j]))
                    plt.clf()
                    plt.text(0.0, 0.5, get_answer(logits.data[j], dataloader))
                    plt.text(0.5, 0.5, get_answer(i[j], dataloader))
                    plt.axis('off')
                    plt.savefig("visualize2/{}_a.jpg".format(question_id[j]))
                    plt.clf()
                    # print(b.data[5])
                    # print(att1.data[0])
                    # print(image_id[0])
                    img = Image.open("/data1/im2txt/im2txt/data/1/raw-data/val2014/COCO_val2014_{}.jpg".format(
                        str(image_id[j]).zfill(12)))
                    w, h = img.size
                    # raw_img = load_image_into_numpy_array(img)
                    # img = plt.imread("data/test2015/COCO_test2015_{}.jpg".format(str(image_id[0]).zfill(12)))
                    # image=Image.fromarray(raw_img)
                    draw = ImageDraw.Draw(img)
                    # print(att1.data[4].cpu().numpy().squeeze())
                    att1_np = att.data[j].cpu().numpy().squeeze()
                    index_sort = np.argsort(att1_np)[::-1][:2]
                    axis = np.arange(36)
                    attN1 = att1_np[index_sort]
                    plt.bar(axis, att1_np)
                    plt.savefig("visualize2/{}_att.jpg".format(question_id[j]))
                    plt.clf()
                    # print(index_sort)
                    # print(b.data[4].cpu().numpy())
                    boxesN = b.data[j].cpu().numpy()[index_sort]
                    # print(boxesN)
                    # boxesN = [list(map(int, box)) for box in boxesN]
                    # print(boxesN)
                    for ii, e in enumerate(boxesN):
                        # print(e[0])
                        # print(e[1])
                        # print(e[2])
                        # print(e[3])
                        (left, right, top, bottom) = (e[0] * w, e[2] * w,
                                                      e[1] * h, e[3] * h)
                        # print(left)
                        # print(right)
                        # print(top)
                        # print(bottom)
                        draw.line([(left, top), (left, bottom), (right, bottom),
                                   (right, top), (left, top)], width=4, fill='red')
                        try:
                            font = ImageFont.truetype('font/arial.ttf', 30)
                        except IOError:
                            font = ImageFont.load_default()

                            # If the total height of the display strings added to the top of the bounding
                            # box exceeds the top of the image, stack the strings below the bounding box
                            # instead of above.
                        display_str_heights = [font.getsize(str(round(ds, 2)))[1] for ds in attN1]
                        # Each display_str has a top and bottom margin of 0.05x.
                        total_display_str_height = (1 + 2 * 0.05) * sum(display_str_heights)

                        if top > total_display_str_height:
                            text_bottom = top
                        else:
                            text_bottom = bottom + total_display_str_height
                        # Reverse list and print from bottom to top.
                        text_width, text_height = font.getsize(str(round(attN1[ii], 2)))
                        margin = np.ceil(0.05 * text_height)
                        draw.rectangle([(left, text_bottom - text_height - 2 * margin), (left + text_width,
                                                                                         text_bottom)],
                                       fill='red')
                        draw.text(
                            (left + margin, text_bottom - text_height - margin),
                            str(round(attN1[ii], 2)),
                            fill='black',
                            font=font)
                    plt.imshow(np.array(img.convert('RGB')))
                    plt.axis('off')
                    plt.savefig("visualize2/{}_img.jpg".format(question_id[j]))
                    plt.clf()

                    plt.figure(2)
                    plt.clf()
                    plt.text(0.0, 0.5, get_question(q.data[j], dataloader))
                    plt.axis('off')
                    plt.savefig("visualize3/{}_q.jpg".format(question_id[j]))
                    plt.clf()
                    plt.text(0.0, 0.5, get_answer(logits1.data[j], dataloader))
                    plt.text(0.5, 0.5, get_answer(i[j], dataloader))
                    plt.axis('off')
                    plt.savefig("visualize3/{}_a.jpg".format(question_id[j]))
                    plt.clf()
                    # print(b.data[5])
                    # print(att1.data[0])
                    # print(image_id[0])
                    img = Image.open("/data1/im2txt/im2txt/data/1/raw-data/val2014/COCO_val2014_{}.jpg".format(str(image_id[j]).zfill(12)))
                    w, h = img.size
                    # raw_img = load_image_into_numpy_array(img)
                    # img = plt.imread("data/test2015/COCO_test2015_{}.jpg".format(str(image_id[0]).zfill(12)))
                    # image=Image.fromarray(raw_img)
                    draw = ImageDraw.Draw(img)
                    # print(att1.data[4].cpu().numpy().squeeze())
                    att1_np = att1.data[j].cpu().numpy().squeeze()
                    index_sort = np.argsort(att1_np)[::-1][:2]
                    axis = np.arange(36)
                    attN1 = att1_np[index_sort]
                    plt.bar(axis, att1_np)
                    plt.savefig("visualize3/{}_att.jpg".format(question_id[j]))
                    plt.clf()
                    # print(index_sort)
                    # print(b.data[4].cpu().numpy())
                    boxesN = b.data[j].cpu().numpy()[index_sort]
                    # print(boxesN)
                    # boxesN = [list(map(int, box)) for box in boxesN]
                    # print(boxesN)
                    for ii, e in enumerate(boxesN):
                        # print(e[0])
                        # print(e[1])
                        # print(e[2])
                        # print(e[3])
                        (left, right, top, bottom) = (e[0] * w, e[2] * w,
                                                      e[1] * h, e[3] * h)
                        # print(left)
                        # print(right)
                        # print(top)
                        # print(bottom)
                        draw.line([(left, top), (left, bottom), (right, bottom),
                                   (right, top), (left, top)], width=4, fill='red')
                        try:
                            font = ImageFont.truetype('font/arial.ttf', 30)
                        except IOError:
                            font = ImageFont.load_default()

                            # If the total height of the display strings added to the top of the bounding
                            # box exceeds the top of the image, stack the strings below the bounding box
                            # instead of above.
                        display_str_heights = [font.getsize(str(round(ds, 2)))[1] for ds in attN1]
                        # Each display_str has a top and bottom margin of 0.05x.
                        total_display_str_height = (1 + 2 * 0.05) * sum(display_str_heights)

                        if top > total_display_str_height:
                            text_bottom = top
                        else:
                            text_bottom = bottom + total_display_str_height
                        # Reverse list and print from bottom to top.
                        text_width, text_height = font.getsize(str(round(attN1[ii], 2)))
                        margin = np.ceil(0.05 * text_height)
                        draw.rectangle([(left, text_bottom - text_height - 2 * margin), (left + text_width,
                                                                                         text_bottom)],
                                       fill='red')
                        draw.text(
                            (left + margin, text_bottom - text_height - margin),
                            str(round(attN1[ii], 2)),
                            fill='black',
                            font=font)
                    plt.imshow(np.array(img.convert('RGB')))
                    plt.axis('off')
                    plt.savefig("visualize3/{}_img.jpg".format(question_id[j]))
                    plt.clf()

                    img = Image.open("/data1/im2txt/im2txt/data/1/raw-data/val2014/COCO_val2014_{}.jpg".format(str(image_id[j]).zfill(12)))
                    draw = ImageDraw.Draw(img)
                    att2_np = att2.data[j].cpu().numpy().squeeze()
                    index_sort2 = np.argsort(att2_np)[::-1][:2]
                    attN2 = att2_np[index_sort2]
                    plt.bar(axis, att2_np)
                    plt.savefig("visualize3/{}_att2.jpg".format(question_id[j]))
                    plt.clf()
                    boxesN2 = b.data[j].cpu().numpy()[index_sort2]
                    # boxesN2 = [list(map(int, box)) for box in boxesN2]
                    for ii, e in enumerate(boxesN2):
                        # print(e[0])
                        # print(e[1])
                        # print(e[2])
                        # print(e[3])
                        (left, right, top, bottom) = (e[0] * w, e[2] * w,
                                                      e[1] * h, e[3] * h)
                        # print(left)
                        # print(right)
                        # print(top)
                        # print(bottom)
                        draw.line([(left, top), (left, bottom), (right, bottom),
                                   (right, top), (left, top)], width=4, fill='red')
                        try:
                            font = ImageFont.truetype('font/arial.ttf', 30)
                        except IOError:
                            font = ImageFont.load_default()

                            # If the total height of the display strings added to the top of the bounding
                            # box exceeds the top of the image, stack the strings below the bounding box
                            # instead of above.
                        display_str_heights = [font.getsize(str(round(ds, 2)))[1] for ds in attN2]
                        # Each display_str has a top and bottom margin of 0.05x.
                        total_display_str_height = (1 + 2 * 0.05) * sum(display_str_heights)

                        if top > total_display_str_height:
                            text_bottom = top
                        else:
                            text_bottom = bottom + total_display_str_height
                        # Reverse list and print from bottom to top.
                        text_width, text_height = font.getsize(str(round(attN2[ii], 2)))
                        margin = np.ceil(0.05 * text_height)
                        draw.rectangle([(left, text_bottom - text_height - 2 * margin), (left + text_width,
                                                                                         text_bottom)],
                                       fill='red')
                        draw.text(
                            (left + margin, text_bottom - text_height - margin),
                            str(round(attN2[ii], 2)),
                            fill='black',
                            font=font)
                    plt.imshow(np.array(img.convert('RGB')))
                    plt.axis('off')
                    # plt.imsave("test.jpg",img)
                    # plt.savefig("visualize/{}.jpg".format(i[j]))
                    plt.savefig("visualize3/{}_img2.jpg".format(question_id[j]))
                    plt.clf()
"""
"""
                img = Image.open("data/test2015/COCO_test2015_{}.jpg".format(str(image_id[j]).zfill(12)))
                draw = ImageDraw.Draw(img)
                att2_np=att2.data[j].cpu().numpy().squeeze()
                index_sort2 = np.argsort(att2_np)[::-1][:2]
                attN2 = att2_np[index_sort2]
                plt.bar(axis,att2_np)
                plt.savefig("visualize/{}_att2.jpg".format(i[j]))
                plt.clf()
                boxesN2 = b.data[j].cpu().numpy()[index_sort2]
                # boxesN2 = [list(map(int, box)) for box in boxesN2]
                for ii,e in enumerate(boxesN2):
                    # print(e[0])
                    # print(e[1])
                    # print(e[2])
                    # print(e[3])
                    (left, right, top, bottom) = (e[0] * w, e[2] * w,
                                                  e[1] * h, e[3] * h)
                    # print(left)
                    # print(right)
                    # print(top)
                    # print(bottom)
                    draw.line([(left, top), (left, bottom), (right, bottom),
                               (right, top), (left, top)], width=4, fill='red')
                    try:
                        font = ImageFont.truetype('font/arial.ttf', 30)
                    except IOError:
                        font = ImageFont.load_default()

                        # If the total height of the display strings added to the top of the bounding
                        # box exceeds the top of the image, stack the strings below the bounding box
                        # instead of above.
                    display_str_heights = [font.getsize(str(round(ds,2)))[1] for ds in attN2]
                    # Each display_str has a top and bottom margin of 0.05x.
                    total_display_str_height = (1 + 2 * 0.05) * sum(display_str_heights)

                    if top > total_display_str_height:
                        text_bottom = top
                    else:
                        text_bottom = bottom + total_display_str_height
                    # Reverse list and print from bottom to top.
                    text_width, text_height = font.getsize(str(round(attN2[ii],2)))
                    margin = np.ceil(0.05 * text_height)
                    draw.rectangle([(left, text_bottom - text_height - 2 * margin), (left + text_width,
                                                                              text_bottom)],
                            fill='red')
                    draw.text(
                            (left + margin, text_bottom - text_height - margin),
                            str(round(attN2[ii], 2)),
                            fill='black',
                            font=font)
                plt.imshow(np.array(img.convert('RGB')))
                plt.axis('off')
                # plt.imsave("test.jpg",img)
                #plt.savefig("visualize/{}.jpg".format(i[j]))
                plt.savefig("visualize/{}_img2.jpg".format(i[j]))
                plt.clf()
                #bar.update(idx)
"""
    #return pred, qIds,image_ids


def make_json(logits, qIds, dataloader):
    utils.assert_eq(logits.size(0), len(qIds))
    results = []
    for i in range(logits.size(0)):
        result = {}
        result['question_id'] = qIds[i]
        result['answer'] = get_answer(logits[i], dataloader)
        results.append(result)
    return results

if __name__ == '__main__':
    args = parse_args()

    torch.backends.cudnn.benchmark = True

    dictionary = Dictionary.load_from_file('data/dictionary.pkl')
    eval_dset = VQAFeatureDataset(args.split, dictionary, adaptive=False)

    n_device = torch.cuda.device_count()
    batch_size = args.batch_size * n_device

    constructor = 'build_%s' % args.model
    constructor2 = 'build_%s' % args.model2
    model = getattr(base_model, constructor)(eval_dset, args.num_hid).cuda()
    model2 = getattr(base_model, constructor2)(eval_dset, args.num_hid).cuda()
    eval_loader =  DataLoader(eval_dset, batch_size, shuffle=False, num_workers=1, collate_fn=utils.trim_collate)

    def process(args, model,model2, eval_loader):
        model_path2 = args.input2+'/model%s.pth' % \
            ('' if 0 > args.epoch else '_epoch%d' % args.epoch)
        model_path=args.input+'/model.pth'

        print('loading %s' % model_path)
        model_data = torch.load(model_path)

        model = nn.DataParallel(model).cuda()
        model.load_state_dict(model_data.get('model_state', model_data))

        model_data2 = torch.load(model_path2)

        model2 = nn.DataParallel(model2).cuda()
        model2.load_state_dict(model_data2.get('model_state', model_data))

        model.train(False)

        get_logits2(model, model2,eval_loader)
        """
        results = make_json(logits, qIds, eval_loader)
        model_label = '%s_%s' % (args.model,args.num_hid)

        if args.logits:
            utils.create_dir('logits/'+model_label)
            torch.save(logits, 'logits/'+model_label+'/logits%d.pth' % args.index)
        
        utils.create_dir(args.output)
        if 0 <= args.epoch:
            model_label += '_epoch%d' % args.epoch

        with open(args.output+'/%s_%s.json' \
            % (args.split, model_label), 'w') as f:
            json.dump(results, f)
        """

    process(args, model, model2,eval_loader)
