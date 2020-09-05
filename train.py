"""
This code is modified from Hengyuan Hu's repository.
https://github.com/hengyuan-hu/bottom-up-attention-vqa
"""
import os
import time
#import itertools
import torch
import torch.nn as nn
import utils
from torch.autograd import Variable
import torch.optim.lr_scheduler as lr_scheduler
from torch.optim.lr_scheduler import LambdaLR
#from bisect import bisect
#from tensorboardX import SummaryWriter

wu_iters=6934
wu_factor=0.2
lr_ratio=0.5
def instance_bce_with_logits(logits, labels):
    assert logits.dim() == 2

    loss = nn.functional.binary_cross_entropy_with_logits(logits, labels)
    loss *= labels.size(1)
    return loss


def compute_score_with_logits(logits, labels):
    logits = torch.max(logits, 1)[1].data # argmax
    one_hots = torch.zeros(*labels.size()).cuda()
    one_hots.scatter_(1, logits.view(-1, 1), 1)
    scores = (one_hots * labels)
    return scores

def lr_lambda_fun(i_iter):
    if i_iter <= wu_iters:
        alpha = float(i_iter) / float(wu_iters)
        return wu_factor * (1. - alpha) + alpha
    else:
        if i_iter<=10401:
            return 1
        elif i_iter<=13868:
            return 0.5
        elif i_iter<=31203:
            return pow(lr_ratio, 2)
        else:
            return pow(lr_ratio, 3)


def get_optim_scheduler(optimizer):
    return LambdaLR(optimizer, lr_lambda=lr_lambda_fun)


def train(model, train_loader, eval_loader,output):
    lr_default=0.01
    grad_clip = .25
    epoch=0
    i_iter=0
    max_iter=45071
    utils.create_dir(output)
    optim = torch.optim.Adamax(filter(lambda p: p.requires_grad, model.parameters()), lr=lr_default)
    scheduler = get_optim_scheduler(optim)
    logger = utils.Logger(os.path.join(output, 'log.txt'))
    best_eval_score = 0
    utils.print_model(model, logger)

    while i_iter<max_iter:
        total_loss = 0
        train_score = 0
        total_norm = 0
        count_norm = 0
        epoch=epoch+1
        N = len(train_loader.dataset)
        logger.write('lr: %.4f' % optim.param_groups[0]['lr'])
        t=time.time()

        for i, (v, b, q, a) in enumerate(train_loader):
            i_iter=i_iter+1
            if i_iter>max_iter:
                break
            scheduler.step(i_iter)
            optim.zero_grad()
            v = Variable(v).cuda()
            b = Variable(b).cuda()
            q = Variable(q).cuda()
            a = Variable(a).cuda()

            pred= model(v, b, q, a)
            loss = instance_bce_with_logits(pred, a)
            loss.backward()
            total_norm += nn.utils.clip_grad_norm(model.parameters(), grad_clip)
            count_norm += 1

            batch_score = compute_score_with_logits(pred, a.data).sum()
            total_loss += loss.data[0] * v.size(0)
            train_score += batch_score
            #print('batch_score: %.2f' % (batch_score))
            #print(train_score)
            optim.step()

        total_loss /= N
        train_score = 100 * train_score / N
        if None != eval_loader:
            model.train(False)
            eval_score = evaluate(model, eval_loader)
            model.train(True)
        logger.write('epoch: %d, time: %.2f' % (epoch, time.time()-t))
        logger.write('\ttrain_loss: %.2f, norm: %.4f, score: %.2f' % (total_loss, total_norm/count_norm, train_score))

        if eval_loader is not None:
            logger.write('\teval score: %.2f' % (100 * eval_score))
        if (eval_loader is not None and eval_score > best_eval_score):
            model_path = os.path.join(output, 'model_epoch%d.pth' % (epoch))
            utils.save_model(model_path, model, iter, optim)
            if eval_loader is not None:
                best_eval_score = eval_score

def train2(model, train_loader, eval_loader, num_epochs, output, opt=None, s_epoch=0):
    lr_default = 2e-3 if eval_loader is not None else 7e-4
    lr_decay_step = 2
    lr_decay_rate = .5
    lr_decay_epochs = range(4,10,lr_decay_step)
    gradual_warmup_steps = [1.0 * lr_default, 2.0 * lr_default, 4.0 * lr_default, 5.0 * lr_default]
    saving_epoch = 3
    grad_clip = .25

    utils.create_dir(output)
    optim = torch.optim.Adamax(filter(lambda p: p.requires_grad, model.parameters()), lr=lr_default) \
        if opt is None else opt
    logger = utils.Logger(os.path.join(output, 'log.txt'))
    best_eval_score = 0

    utils.print_model(model, logger)
    logger.write('optim: adamax lr=%.4f, decay_step=%d, decay_rate=%.2f, grad_clip=%.2f' % \
        (lr_default, lr_decay_step, lr_decay_rate, grad_clip))

    for epoch in range(s_epoch, num_epochs):
        total_loss = 0
        train_score = 0
        total_norm = 0
        count_norm = 0
        t = time.time()
        N = len(train_loader.dataset)
        if epoch < len(gradual_warmup_steps):
            optim.param_groups[0]['lr'] = gradual_warmup_steps[epoch]
            logger.write('gradual warmup lr: %.4f' % optim.param_groups[0]['lr'])
        elif epoch in lr_decay_epochs:
            optim.param_groups[0]['lr'] *= lr_decay_rate
            logger.write('decreased lr: %.4f' % optim.param_groups[0]['lr'])
        else:
            logger.write('lr: %.4f' % optim.param_groups[0]['lr'])

        for i, (v, b, q, a) in enumerate(train_loader):
            v = Variable(v).cuda()
            b = Variable(b).cuda()
            q = Variable(q).cuda()
            a = Variable(a).cuda()

            pred= model(v, b, q, a)
            loss = instance_bce_with_logits(pred, a)
            loss.backward()
            total_norm += nn.utils.clip_grad_norm(model.parameters(), grad_clip)
            count_norm += 1
            optim.step()
            optim.zero_grad()

            batch_score = compute_score_with_logits(pred, a.data).sum()
            total_loss += loss.data[0] * v.size(0)
            train_score += batch_score

        total_loss /= N
        train_score = 100 * train_score / N
        if None != eval_loader:
            model.train(False)
            eval_score= evaluate(model, eval_loader)
            model.train(True)

        logger.write('epoch %d, time: %.2f' % (epoch, time.time()-t))
        logger.write('\ttrain_loss: %.2f, norm: %.4f, score: %.2f' % (total_loss, total_norm/count_norm, train_score))
        if eval_loader is not None:
            logger.write('\teval score: %.2f (%.2f)' % (100 * eval_score))

        if (eval_loader is not None and eval_score > best_eval_score) or (eval_loader is None and epoch >= saving_epoch):
            model_path = os.path.join(output, 'model_epoch%d.pth' % epoch)
            utils.save_model(model_path, model, epoch, optim)
            if eval_loader is not None:
                best_eval_score = eval_score
def train3(model, train_loader, eval_loader, num_epochs, output,s_epoch=0):
    lr_default=0.001
    grad_clip = .25
    utils.create_dir(output)
    lr_decay_step = 2
    lr_decay_rate = .5
    lr_decay_epochs = range(9, 12, lr_decay_step)
    gradual_warmup_steps = [0.5 * lr_default, 1.0 * lr_default,2.0*lr_default]
    optim = torch.optim.Adamax(filter(lambda p: p.requires_grad, model.parameters()), lr=lr_default)
    logger = utils.Logger(os.path.join(output, 'log.txt'))
    best_eval_score = 0
    utils.print_model(model, logger)
    for epoch in range(s_epoch, num_epochs):
        total_loss = 0
        train_score = 0
        total_norm = 0
        count_norm = 0
        t = time.time()
        N = len(train_loader.dataset)
        if epoch < len(gradual_warmup_steps):
            optim.param_groups[0]['lr'] = gradual_warmup_steps[epoch]
            logger.write('gradual warmup lr: %.4f' % optim.param_groups[0]['lr'])
        elif epoch in lr_decay_epochs:
            optim.param_groups[0]['lr'] *= lr_decay_rate
            logger.write('decreased lr: %.4f' % optim.param_groups[0]['lr'])
        else:
            logger.write('lr: %.4f' % optim.param_groups[0]['lr'])
        for i, (v, b, q, a) in enumerate(train_loader):
            v = Variable(v).cuda()
            b = Variable(b).cuda()
            q = Variable(q).cuda()
            a = Variable(a).cuda()

            pred= model(v, b, q, a)
            loss = instance_bce_with_logits(pred, a)
            loss.backward()
            total_norm += nn.utils.clip_grad_norm(model.parameters(), grad_clip)
            count_norm += 1
            optim.step()
            optim.zero_grad()

            batch_score = compute_score_with_logits(pred, a.data).sum()
            total_loss += loss.data[0] * v.size(0)
            train_score += batch_score

        total_loss /= N
        train_score = 100 * train_score / N
        if None != eval_loader:
            model.train(False)
            eval_score,eval_loss= evaluate(model, eval_loader)
            model.train(True)

        logger.write('epoch %d, time: %.2f' % (epoch, time.time()-t))
        logger.write('\ttrain_loss: %.2f, norm: %.4f, score: %.2f' % (total_loss, total_norm/count_norm, train_score))
        if eval_loader is not None:
            logger.write('\teval score: %.2f,eval loss:%.2f' % (100 * eval_score,eval_loss))

        if (eval_loader is not None and eval_score > best_eval_score):
            model_path = os.path.join(output, 'model_epoch%d.pth' % epoch)
            utils.save_model(model_path, model, epoch, optim)
            if eval_loader is not None:
                best_eval_score = eval_score

#final train function
def train4(model, train_loader, eval_loader, num_epochs, output,s_epoch=0):
    lr_default=0.001
    grad_clip = .25
    utils.create_dir(output)
    lr_decay_step = 2
    lr_decay_rate = .5
    lr_decay_epochs = range(9, 12, lr_decay_step)
    gradual_warmup_steps = [0.5 * lr_default, 1.0 * lr_default,2.0*lr_default]
    optim = torch.optim.Adamax(filter(lambda p: p.requires_grad, model.parameters()), lr=lr_default)
    logger = utils.Logger(os.path.join(output, 'log.txt'))
    best_eval_score = 0
    utils.print_model(model, logger)
    for epoch in range(s_epoch, num_epochs):
        total_loss = 0
        train_score = 0
        total_norm = 0
        count_norm = 0
        t = time.time()
        N = len(train_loader.dataset)
        if epoch < len(gradual_warmup_steps):
            optim.param_groups[0]['lr'] = gradual_warmup_steps[epoch]
            logger.write('gradual warmup lr: %.4f' % optim.param_groups[0]['lr'])
        elif epoch in lr_decay_epochs:
            optim.param_groups[0]['lr'] *= lr_decay_rate
            logger.write('decreased lr: %.4f' % optim.param_groups[0]['lr'])
        else:
            logger.write('lr: %.4f' % optim.param_groups[0]['lr'])
        for i, (v, b, q, a,image_id) in enumerate(train_loader):
            v = Variable(v).cuda()
            b = Variable(b).cuda()
            q = Variable(q).cuda()
            a = Variable(a).cuda()

            pred,att_1,att_2= model(v, b, q, a)
            loss = instance_bce_with_logits(pred, a)
            loss.backward()
            total_norm += nn.utils.clip_grad_norm(model.parameters(), grad_clip)
            count_norm += 1
            optim.step()
            optim.zero_grad()

            batch_score = compute_score_with_logits(pred, a.data).sum()
            total_loss += loss.data[0] * v.size(0)
            train_score += batch_score

        total_loss /= N
        train_score = 100 * train_score / N
        if None != eval_loader:
            model.train(False)
            eval_score,eval_loss= evaluate(model, eval_loader)
            model.train(True)

        logger.write('epoch %d, time: %.2f' % (epoch, time.time()-t))
        logger.write('\ttrain_loss: %.2f, norm: %.4f, score: %.2f' % (total_loss, total_norm/count_norm, train_score))
        if eval_loader is not None:
            logger.write('\teval score: %.2f,eval loss:%.2f' % (100 * eval_score,eval_loss))

        if (eval_loader is not None and eval_score > best_eval_score) or (eval_loader is None and epoch>=0):
            model_path = os.path.join(output, 'model_epoch%d.pth' % epoch)
            utils.save_model(model_path, model, epoch, optim)
            if eval_loader is not None:
                best_eval_score = eval_score

def train5(model, train_loader, eval_loader, num_epochs, output,s_epoch=0):
    lr_default=0.001
    grad_clip = .25
    utils.create_dir(output)
    lr_decay_step = 2
    lr_decay_rate = .5
    lr_decay_epochs = range(9, 12, lr_decay_step)
    gradual_warmup_steps = [0.5 * lr_default, 1.0 * lr_default,2.0*lr_default]
    optim = torch.optim.Adamax(filter(lambda p: p.requires_grad, model.parameters()), lr=lr_default)
    logger = utils.Logger(os.path.join(output, 'log.txt'))
    best_eval_score = 0
    utils.print_model(model, logger)
    for epoch in range(s_epoch, num_epochs):
        total_loss = 0
        train_score = 0
        total_norm = 0
        count_norm = 0
        t = time.time()
        N = len(train_loader.dataset)
        if epoch < len(gradual_warmup_steps):
            optim.param_groups[0]['lr'] = gradual_warmup_steps[epoch]
            logger.write('gradual warmup lr: %.4f' % optim.param_groups[0]['lr'])
        elif epoch in lr_decay_epochs:
            optim.param_groups[0]['lr'] *= lr_decay_rate
            logger.write('decreased lr: %.4f' % optim.param_groups[0]['lr'])
        else:
            logger.write('lr: %.4f' % optim.param_groups[0]['lr'])
        for i, (v, b, q, a,image_id) in enumerate(train_loader):
            v = Variable(v).cuda()
            b = Variable(b).cuda()
            q = Variable(q).cuda()
            a = Variable(a).cuda()

            pred,att= model(v, b, q, a)
            loss = instance_bce_with_logits(pred, a)
            loss.backward()
            total_norm += nn.utils.clip_grad_norm(model.parameters(), grad_clip)
            count_norm += 1
            optim.step()
            optim.zero_grad()

            batch_score = compute_score_with_logits(pred, a.data).sum()
            total_loss += loss.data[0] * v.size(0)
            train_score += batch_score

        total_loss /= N
        train_score = 100 * train_score / N
        if None != eval_loader:
            model.train(False)
            eval_score,eval_loss= evaluate2(model, eval_loader)
            model.train(True)

        logger.write('epoch %d, time: %.2f' % (epoch, time.time()-t))
        logger.write('\ttrain_loss: %.2f, norm: %.4f, score: %.2f' % (total_loss, total_norm/count_norm, train_score))
        if eval_loader is not None:
            logger.write('\teval score: %.2f,eval loss:%.2f' % (100 * eval_score,eval_loss))

        if (eval_loader is not None and eval_score > best_eval_score) or (eval_loader is None and epoch>=0):
            model_path = os.path.join(output, 'model_epoch%d.pth' % epoch)
            utils.save_model(model_path, model, epoch, optim)
            if eval_loader is not None:
                best_eval_score = eval_score
def train6(model, train_loader, eval_loader, num_epochs, output,s_epoch=0):
    lr_default=0.001
    grad_clip = .25
    utils.create_dir(output)
    lr_decay_step = 2
    lr_decay_rate = .5
    lr_decay_epochs = range(9, 12, lr_decay_step)
    gradual_warmup_steps = [0.5 * lr_default, 1.0 * lr_default,2.0*lr_default]
    optim = torch.optim.Adamax(filter(lambda p: p.requires_grad, model.parameters()), lr=lr_default)
    logger = utils.Logger(os.path.join(output, 'log.txt'))
    best_eval_score = 0
    utils.print_model(model, logger)
    for epoch in range(s_epoch, num_epochs):
        total_loss = 0
        train_score = 0
        total_norm = 0
        count_norm = 0
        t = time.time()
        N = len(train_loader.dataset)
        if epoch < len(gradual_warmup_steps):
            optim.param_groups[0]['lr'] = gradual_warmup_steps[epoch]
            logger.write('gradual warmup lr: %.4f' % optim.param_groups[0]['lr'])
        elif epoch in lr_decay_epochs:
            optim.param_groups[0]['lr'] *= lr_decay_rate
            logger.write('decreased lr: %.4f' % optim.param_groups[0]['lr'])
        else:
            logger.write('lr: %.4f' % optim.param_groups[0]['lr'])
        for i, (v, b, q, a,image_id) in enumerate(train_loader):
            v = Variable(v).cuda()
            b = Variable(b).cuda()
            q = Variable(q).cuda()
            a = Variable(a).cuda()

            pred= model(v, b, q, a)
            loss = instance_bce_with_logits(pred, a)
            loss.backward()
            total_norm += nn.utils.clip_grad_norm(model.parameters(), grad_clip)
            count_norm += 1
            optim.step()
            optim.zero_grad()

            batch_score = compute_score_with_logits(pred, a.data).sum()
            total_loss += loss.data[0] * v.size(0)
            train_score += batch_score

        total_loss /= N
        train_score = 100 * train_score / N
        if None != eval_loader:
            model.train(False)
            eval_score,eval_loss= evaluate3(model, eval_loader)
            model.train(True)

        logger.write('epoch %d, time: %.2f' % (epoch, time.time()-t))
        logger.write('\ttrain_loss: %.2f, norm: %.4f, score: %.2f' % (total_loss, total_norm/count_norm, train_score))
        if eval_loader is not None:
            logger.write('\teval score: %.2f,eval loss:%.2f' % (100 * eval_score,eval_loss))

        if (eval_loader is not None and eval_score > best_eval_score) or (eval_loader is None and epoch>=0):
            model_path = os.path.join(output, 'model_epoch%d.pth' % epoch)
            utils.save_model(model_path, model, epoch, optim)
            if eval_loader is not None:
                best_eval_score = eval_score
def train7(model, train_loader, eval_loader, num_epochs, output):
    utils.create_dir(output)
    optim = torch.optim.Adamax(model.parameters())
    logger = utils.Logger(os.path.join(output, 'log.txt'))
    best_eval_score = 0

    for epoch in range(num_epochs):
        total_loss = 0
        train_score = 0
        t = time.time()

        for i, (v, b, q, a,image_id) in enumerate(train_loader):
            v = Variable(v).cuda()
            b = Variable(b).cuda()
            q = Variable(q).cuda()
            a = Variable(a).cuda()

            pred,att= model(v, b, q, a)
            loss = instance_bce_with_logits(pred, a)
            loss.backward()
            nn.utils.clip_grad_norm(model.parameters(), 0.25)
            optim.step()
            optim.zero_grad()

            batch_score = compute_score_with_logits(pred, a.data).sum()
            total_loss += loss.data[0] * v.size(0)
            train_score += batch_score

        total_loss /= len(train_loader.dataset)
        train_score = 100 * train_score / len(train_loader.dataset)
        model.train(False)
        eval_score,val_loss = evaluate2(model, eval_loader)
        model.train(True)

        logger.write('epoch %d, time: %.2f' % (epoch, time.time()-t))
        logger.write('\ttrain_loss: %.2f, score: %.2f' % (total_loss, train_score))
        logger.write('\teval score: %.2f ,loss:%.2f' % (100 * eval_score, val_loss))

        if eval_score > best_eval_score:
            model_path = os.path.join(output, 'model.pth')
            torch.save(model.state_dict(), model_path)
            best_eval_score = eval_score
def evaluate(model, dataloader):
    score = 0
    total_loss=0
    for v, b, q, a ,image_id in iter(dataloader):
        v = Variable(v).cuda()
        b = Variable(b).cuda()
        a = Variable(a).cuda()
        q = Variable(q, volatile=True).cuda()
        pred ,att1,att2= model(v, b, q, None)
        loss = instance_bce_with_logits(pred, a)
        total_loss += loss.data[0] * v.size(0)
        batch_score = compute_score_with_logits(pred,a.data).sum()
        score += batch_score
    score = score / len(dataloader.dataset)
    total_loss=total_loss/len(dataloader.dataset)
    return score,total_loss
def evaluate2(model, dataloader):
    score = 0
    total_loss=0
    for v, b, q, a ,image_id in iter(dataloader):
        v = Variable(v).cuda()
        b = Variable(b).cuda()
        a = Variable(a).cuda()
        q = Variable(q, volatile=True).cuda()
        pred,att= model(v, b, q, None)
        loss = instance_bce_with_logits(pred, a)
        total_loss += loss.data[0] * v.size(0)
        batch_score = compute_score_with_logits(pred,a.data).sum()
        score += batch_score
    score = score / len(dataloader.dataset)
    total_loss=total_loss/len(dataloader.dataset)
    return score,total_loss
def evaluate3(model, dataloader):
    score = 0
    total_loss=0
    for v, b, q, a ,image_id in iter(dataloader):
        v = Variable(v).cuda()
        b = Variable(b).cuda()
        a = Variable(a).cuda()
        q = Variable(q, volatile=True).cuda()
        pred= model(v, b, q, None)
        loss = instance_bce_with_logits(pred, a)
        total_loss += loss.data[0] * v.size(0)
        batch_score = compute_score_with_logits(pred,a.data).sum()
        score += batch_score
    score = score / len(dataloader.dataset)
    total_loss=total_loss/len(dataloader.dataset)
    return score,total_loss