
import os
import time
#import itertools
import torch
import torch.nn as nn
import utils
from torch.autograd import Variable
import torch.optim.lr_scheduler as lr_scheduler
from torch.optim.lr_scheduler import LambdaLR

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

def train_coco(model, train_loader, eval_loader, num_epochs, output,s_epoch=0):
    lr_default=0.001
    grad_clip = .25
    utils.create_dir(output)
    lr_decay_step = 2
    lr_decay_rate = .5
    lr_decay_epochs = range(8, 12, lr_decay_step)
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
        for i, (v, b, q, a,question_id,image_id,types) in enumerate(train_loader):
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
def evaluate(model, dataloader):
    score = 0
    total_loss=0
    for v, b, q, a,question_id,image_id,types in iter(dataloader):
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