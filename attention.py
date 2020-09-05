"""
Bilinear Attention Networks
Jin-Hwa Kim, Jaehyun Jun, Byoung-Tak Zhang
https://arxiv.org/abs/1805.07932

This code is written by Jin-Hwa Kim.
"""
import torch
import torch.nn as nn
from torch.nn.utils.weight_norm import weight_norm
from fc import FCNet,get_norm,nonlinear_layer
from bc import *
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, v_dim, q_dim, num_hid):
        super(Attention, self).__init__()
        self.nonlinear = FCNet([v_dim + q_dim, num_hid],0.0,"weight","ReLU")
        self.linear = weight_norm(nn.Linear(num_hid, 1), dim=None)

    def forward(self, v, q):
        """
        v: [batch, k, vdim]
        q: [batch, qdim]
        """
        logits = self.logits(v, q)
        w = nn.functional.softmax(logits, 1)
        return w

    def logits(self, v, q):
        num_objs = v.size(1)
        q = q.unsqueeze(1).repeat(1, num_objs, 1)
        vq = torch.cat((v, q), 2)
        joint_repr = self.nonlinear(vq)
        logits = self.linear(joint_repr)
        return logits

class BiAttention(nn.Module):
    def __init__(self, x_dim, y_dim, z_dim, glimpse, dropout=[.2,.5]):
        super(BiAttention, self).__init__()

        self.glimpse = glimpse
        self.logits = weight_norm(BCNet(x_dim, y_dim, z_dim, glimpse, dropout=dropout, k=3), \
            name='h_mat', dim=None)

    def forward(self, v, q, v_mask=True):
        """
        v: [batch, k, vdim]
        q: [batch, qdim]
        """
        p, logits = self.forward_all(v, q, v_mask)
        return p, logits

    def forward_all(self, v, q, v_mask=True):
        v_num = v.size(1)
        q_num = q.size(1)
        logits = self.logits(v,q) # b x g x v x q

        if v_mask:
            mask = (0 == v.abs().sum(2)).unsqueeze(1).unsqueeze(3).expand(logits.size())
            logits.data.masked_fill_(mask.data, -float('inf'))

        p = nn.functional.softmax(logits.view(-1, self.glimpse, v_num * q_num), 2)
        return p.view(-1, self.glimpse, v_num, q_num), logits
class BiAttention2(nn.Module):
    def __init__(self, x_dim, y_dim, hid_num,dropout=[.2,.5]):
        super(BiAttention2, self).__init__()
        self.logits = weight_norm(BCNet2(x_dim, y_dim,hid_num,dropout=dropout))

    def forward(self, v, q):
        """
        v: [batch, k, vdim]
        q: [batch, qdim]
        """
        p, = self.forward_all(v, q)
        return p

    def forward_all(self, v, q):
        v_num = v.size(1)
        q_num = q.size(1)
        logits = self.logits(v,q) # b x v x q
        p = nn.functional.softmax(logits.view(-1,v_num * q_num), 1)
        return p.view(-1, v_num, q_num)
class Att_3(nn.Module):
    def __init__(self, v_dim, q_dim, num_hid, norm, act, dropout=0.0):
        super(Att_3, self).__init__()
        norm_layer = get_norm(norm)
        self.v_proj = FCNet([v_dim, num_hid], dropout= dropout, norm= norm, act= act)
        self.q_proj = FCNet([q_dim, num_hid], dropout= dropout, norm= norm, act= act)
        self.nonlinear = FCNet([num_hid, num_hid], dropout= dropout, norm= norm, act= act)
        self.linear = norm_layer(nn.Linear(num_hid, 1), dim=None)

    def forward(self, v, q):
        """
        v: [batch, k, vdim]
        q: [batch, qdim]
        """
        logits = self.logits(v, q)
        w = nn.functional.softmax(logits, 1)
        return w

    def logits(self, v, q):
        batch, k, _ = v.size()
        v_proj = self.v_proj(v) # [batch, k, num_hid]
        q_proj = self.q_proj(q).unsqueeze(1).repeat(1, k, 1) # [batch, k, num_hid]
        joint_repr = v_proj * q_proj
        joint_repr = self.nonlinear(joint_repr)
        logits = self.linear(joint_repr)
        return logits

class concatenate_attention(nn.Module):
    def __init__(self, image_feat_dim, txt_rnn_embeding_dim, hidden_size):
        super(concatenate_attention, self).__init__()
        self.image_feat_dim = image_feat_dim
        self.txt_embeding_dim = txt_rnn_embeding_dim
        self.Fa = FCNet([image_feat_dim + txt_rnn_embeding_dim, hidden_size],0.0,"weight","LeakyReLU")
        self.lc = nn.Linear(hidden_size, 1)

    def forward(self, image_feat, question_embedding):
        _, num_location, _ = image_feat.shape
        question_embedding_expand = torch.unsqueeze(
            question_embedding, 1).expand(-1, num_location, -1)
        concat_feature = torch.cat(
            (image_feat, question_embedding_expand), dim=2)
        raw_attention = self.lc(self.Fa(concat_feature))
        # softmax across locations
        attention = F.softmax(raw_attention, dim=1)
        return attention


class project_attention(nn.Module):
    def __init__(self,
                 image_feat_dim,
                 txt_rnn_embeding_dim,
                 hidden_size,
                 dropout=0.2):
        super(project_attention, self).__init__()
        self.image_feat_dim = image_feat_dim
        self.txt_embeding_dim = txt_rnn_embeding_dim
        self.Fa_image = FCNet([image_feat_dim, hidden_size],0.0,"weight","LeakyReLU")
        self.Fa_txt = FCNet([txt_rnn_embeding_dim, hidden_size],0.0,"weight","LeakyReLU")
        self.dropout = nn.Dropout(dropout)
        self.lc = nn.Linear(hidden_size, 1)

    def compute_raw_att(self, image_feat, question_embedding):
        _, num_location, _ = image_feat.shape
        image_fa = self.Fa_image(image_feat)
        question_fa = self.Fa_txt(question_embedding)
        question_fa_expand = torch.unsqueeze(
            question_fa, 1).expand(-1, num_location, -1)
        joint_feature = image_fa * question_fa_expand
        joint_feature = self.dropout(joint_feature)
        raw_attention = self.lc(joint_feature)
        return raw_attention

    def forward(self, image_feat, question_embedding):
        raw_attention = self.compute_raw_att(image_feat, question_embedding)
        # softmax across locations
        attention = F.softmax(raw_attention, dim=1)
        return attention


class doubel_project_attention(nn.Module):
    def __init__(self,
                 image_feat_dim,
                 txt_rnn_embeding_dim,
                 hidden_size, dropout=0.2):
        super(doubel_project_attention, self).__init__()
        self.att1 = project_attention(
            image_feat_dim, txt_rnn_embeding_dim, hidden_size, dropout)
        self.att2 = project_attention(
            image_feat_dim, txt_rnn_embeding_dim, hidden_size, dropout)
        self.image_feat_dim = image_feat_dim
        self.txt_embeding_dim = txt_rnn_embeding_dim

    def forward(self, image_feat, question_embedding):
        att1 = self.att1.compute_raw_att(image_feat, question_embedding)
        att2 = self.att2.compute_raw_att(image_feat, question_embedding)
        raw_attention = att1 + att2
        # softmax across locations
        attention = F.softmax(raw_attention, dim=1)
        return attention

class top_down_attention(nn.Module):
    def __init__(self, modal_combine_module, normalization, transform_module):
        super(top_down_attention, self).__init__()
        self.modal_combine = modal_combine_module
        self.normalization = normalization
        self.transform = transform_module
        self.out_dim = self.transform.out_dim

    @staticmethod
    def _mask_attentions(attention, image_locs):
        batch_size, num_loc, n_att = attention.data.shape
        tmp1 = torch.unsqueeze(
            torch.arange(0, num_loc).type(torch.LongTensor),
            dim=0).expand(batch_size, num_loc)
        tmp1 = tmp1.cuda()
        tmp2 = torch.unsqueeze(image_locs.data, 1).expand(batch_size, num_loc)
        mask = torch.ge(tmp1, tmp2)
        mask = torch.unsqueeze(mask, 2).expand_as(attention)
        attention.data.masked_fill_(mask, 0)
        return attention

    def forward(self, image_feat, question_embedding, image_locs=None):
        # N x K x joint_dim
        joint_feature = self.modal_combine(image_feat, question_embedding)
        # N x K x n_att
        raw_attention = self.transform(joint_feature)

        if self.normalization.lower() == 'softmax':
            attention = F.softmax(raw_attention, dim=1)
            if image_locs is not None:
                masked_attention = self._mask_attentions(attention, image_locs)
                masked_attention_sum = torch.sum(masked_attention,
                                                 dim=1, keepdim=True)
                masked_attention = masked_attention / masked_attention_sum
            else:
                masked_attention = attention

        elif self.normalization.lower() == 'sigmoid':
            attention = F.sigmoid(raw_attention)
            masked_attention = attention
            if image_locs is not None:
                masked_attention = self._mask_attentions(attention, image_locs)

        return masked_attention

class SelfAttention(nn.Module):
    def __init__(self, v_dim, q_dim, num_hid):
        super(SelfAttention, self).__init__()
        self.nonlinear = FCNet([v_dim + q_dim, num_hid],0.0,"weight","ReLU")
        self.linear = weight_norm(nn.Linear(num_hid, 1), dim=None)

    def forward(self, v, q):
        """
        v: [batch, k, vdim]
        q: [batch, qdim]
        """
        logits = self.logits(v, q)
        w = nn.functional.softmax(logits, 1)
        return w

    def logits(self, v, q):
        num_objs = v.size(1)
        q = q.unsqueeze(1).repeat(1, num_objs, 1)
        vq = torch.cat((v, q), 2)
        joint_repr = self.nonlinear(vq)
        logits = self.linear(joint_repr)
        return logits