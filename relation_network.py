import torch
import torch.nn as nn
from torch.nn.utils.weight_norm import weight_norm
from attention import Att_3
from fc import FCNet
from bc import BCNet
from op import *
import torch.nn.functional as F
class RN(nn.Module):
    def __init__(self,v_dim,weight,norm,dropout):
        super(RN, self).__init__()
        self.linear1=FCNet([v_dim*2, v_dim],dropout,weight,norm)
        self.linear2=FCNet([v_dim, v_dim],dropout,weight,norm)
        self.linear3=FCNet([v_dim, v_dim],dropout,weight,norm)

    def forward(self, v):
        batch=v.size(1)
        v_split=torch.split(v,1,dim=1)
        output=[]
        for i in range(batch):
            x_i=[]
            for j in range(batch):
                v_tuple=torch.cat((v_split[i].squeeze(1),v_split[j].squeeze(1)),dim=1)
                x_i.append(v_tuple)
            v_tuple=torch.stack(x_i, dim=1)
            v_tuple = self.linear1(v_tuple)
            v_tuple = self.linear2(v_tuple)
            x_new=torch.sum(v_tuple,dim=1)
            output.append(x_new)
        x_out=torch.stack(output, dim=1)
        x_out=self.linear3(x_out)
        return x_out

class RN2(nn.Module):
    def __init__(self,vdim,qdim):
        super(RN2, self).__init__()
        self.compress_q_1 = MyLinear(qdim, 310, p=0.5, af='relu')
        self.expand_q_1 = MyLinear(310, vdim, p=0.5, af='sigmoid')

        self.compress_q_2 = MyLinear(qdim, 310, p=0.5, af='relu')
        self.expand_q_2 = MyLinear(310, vdim, p=0.5, af='sigmoid')

    def forward(self, v,q):
        x_out=self.decare_cat(v,v,q)
        return x_out.sum(1)
    def decare_cat(self, block1, block2, guidance):
        b, m, d = block1.size()
        v_feature_1 = block1.view(-1, m, 1, d).repeat(1, 1, m, 1)  # b*m*m*d
        v_feature_2 = block2.view(-1, 1, m, d).repeat(1, m, 1, 1)  # b*m*m*d
        q_feature_1 = self.expand_q_1(self.compress_q_1(guidance))
        q_feature_2 = self.expand_q_2(self.compress_q_2(guidance))

        v_feature_low_12 = bmul(v_feature_1, q_feature_1) + bmul(v_feature_2, q_feature_2)  # b*m*m*2d
        return v_feature_low_12
class RN3(nn.Module):
    def __init__(self,v_dim,weight,norm,dropout):
        super(RN3, self).__init__()
        self.linear1=FCNet([v_dim*2, v_dim],dropout,weight,norm)
        self.linear2=FCNet([v_dim, v_dim],dropout,weight,norm)

    def forward(self, v):
        batch=v.size(1)
        v_split=torch.split(v,1,dim=1)
        output=[]
        for i in range(batch):
            x_i=[]
            for j in range(batch):
                v_tuple=torch.cat((v_split[i].squeeze(1),v_split[j].squeeze(1)),dim=1)
                x_i.append(v_tuple)
            v_tuple=torch.stack(x_i, dim=1)
            v_tuple = self.linear1(v_tuple)
            x_new=torch.sum(v_tuple,dim=1)
            output.append(x_new)
        x_out=torch.stack(output, dim=1)
        x_out=self.linear2(x_out)
        return x_out

class RN4(nn.Module):
    def __init__(self,v_dim,q_dim,weight,norm,dropout):
        super(RN4, self).__init__()
        self.linear1=FCNet([v_dim*2+q_dim, v_dim],dropout,weight,norm)
        self.linear2=FCNet([v_dim, v_dim],dropout,weight,norm)
        self.linear3 = FCNet([q_dim, q_dim], dropout, weight, norm)

    def forward(self, v,q):
        q_emb=self.linear3(q)
        batch=v.size(1)
        v_split=torch.split(v,1,dim=1)
        output=[]
        for i in range(batch):
            x_i=[]
            for j in range(batch):
                v_tuple=torch.cat((v_split[i].squeeze(1),v_split[j].squeeze(1),q_emb),dim=1)
                x_i.append(v_tuple)
            v_tuple=torch.stack(x_i, dim=1)
            v_tuple = self.linear1(v_tuple)
            x_new=torch.sum(v_tuple,dim=1)
            output.append(x_new)
        x_out=torch.stack(output, dim=1)
        x_out=self.linear2(x_out)
        return x_out

class RN5(nn.Module):
    def __init__(self,v_dim,q_dim,weight,norm,dropout):
        super(RN5, self).__init__()
        self.linear1=FCNet([v_dim+q_dim, v_dim],0.0,weight,norm)
        self.linear2=FCNet([v_dim, v_dim],dropout,weight,norm)

    def forward(self, v,q):
        batch=v.size(1)
        v_split=torch.split(v,1,dim=1)
        output=[]
        for i in range(batch):
            x_i=[]
            for j in range(batch):
                v_add=v_split[i].squeeze(1)+v_split[j].squeeze(1)
                v_tuple=torch.cat((v_add,q),dim=1)
                x_i.append(v_tuple)
            v_tuple=torch.stack(x_i, dim=1)
            v_tuple = self.linear1(v_tuple)
            x_new=torch.sum(v_tuple,dim=1)
            output.append(x_new)
        x_out=torch.stack(output, dim=1)
        x_out=self.linear2(x_out)
        return x_out
class RN6(nn.Module):
    def __init__(self,v_dim,q_dim,weight,norm,dropout):
        super(RN6, self).__init__()
        self.linear1=FCNet([v_dim+q_dim, v_dim],dropout,weight,norm)
        self.linear2=FCNet([v_dim, v_dim],dropout,weight,norm)
        self.linear3 = FCNet([q_dim, q_dim], dropout, weight, norm)

    def forward(self, v,q):
        q_emb=self.linear3(q)
        batch=v.size(1)
        v_split=torch.split(v,1,dim=1)
        output=[]
        for i in range(batch):
            x_i=[]
            for j in range(batch):
                v_tmp=v_split[i].squeeze(1)*v_split[j].squeeze(1)
                v_tuple=torch.cat((v_tmp,q_emb),dim=1)
                x_i.append(v_tuple)
            v_tuple=torch.stack(x_i, dim=1)
            v_tuple = self.linear1(v_tuple)
            x_new=torch.sum(v_tuple,dim=1)
            output.append(x_new)
        x_out=torch.stack(output, dim=1)
        x_out=self.linear2(x_out)
        return x_out

class RN7(nn.Module):
    def __init__(self,v_dim,q_dim,weight,norm,dropout):
        super(RN7, self).__init__()
        self.linear1=FCNet([v_dim+q_dim, v_dim],dropout,weight,norm)
        self.linear2=FCNet([v_dim, v_dim],dropout,weight,norm)
        self.linear3 = FCNet([q_dim, q_dim], dropout, weight, norm)

    def forward(self, v,q):
        q_emb=self.linear3(q)
        batch=v.size(1)
        v_split=torch.split(v,1,dim=1)
        output=[]
        for i in range(batch):
            x_i=[]
            for j in range(batch):
                v_tmp=v_split[i].squeeze(1)+v_split[j].squeeze(1)
                v_tuple=torch.cat((v_tmp,q_emb),dim=1)
                x_i.append(v_tuple)
            v_tuple=torch.stack(x_i, dim=1)
            v_tuple = self.linear1(v_tuple)
            x_new=torch.sum(v_tuple,dim=1)
            output.append(x_new)
        x_out=torch.stack(output, dim=1)
        x_out=self.linear2(x_out)
        return x_out

class RN8(nn.Module):
    def __init__(self,v_dim,q_dim,weight,norm,dropout):
        super(RN8, self).__init__()
        #self.linear1=FCNet([v_dim, v_dim],dropout,weight,norm)
        self.linear2=FCNet([v_dim, v_dim],dropout,weight,norm)
        self.linear3 = FCNet([q_dim, v_dim], dropout, weight, norm)

    def forward(self, v,q):
        q_emb=self.linear3(q)
        batch=v.size(1)
        v_split=torch.split(v,1,dim=1)
        output=[]
        for i in range(batch):
            x_i=[]
            for j in range(batch):
                v_tmp=v_split[i].squeeze(1)+v_split[j].squeeze(1)
                v_tmp=v_tmp*q_emb
                x_i.append(v_tmp)
            v_tuple=torch.stack(x_i, dim=1)
            #v_tuple = self.linear1(v_tuple)
            x_new=torch.sum(v_tuple,dim=1)
            output.append(x_new)
        x_out=torch.stack(output, dim=1)
        x_out=self.linear2(x_out)
        return x_out

class RN9(nn.Module):
    def __init__(self,v_dim,q_dim,weight,norm,dropout):
        super(RN9, self).__init__()
        self.linear1=FCNet([v_dim, v_dim],dropout,weight,norm)
        self.linear2=FCNet([v_dim, v_dim],dropout,weight,norm)
        self.linear3 = FCNet([q_dim, v_dim], dropout, weight, norm)
        self.linear4 = FCNet([q_dim, v_dim], dropout, weight, norm)

    def forward(self, v,q):
        q_emb=self.linear3(q)
        q_emb2 = self.linear3(q)
        batch=v.size(1)
        v_split=torch.split(v,1,dim=1)
        output=[]
        for i in range(batch):
            x_i=[]
            for j in range(batch):
                v_tmp=v_split[i].squeeze(1)*q_emb+v_split[j].squeeze(1)*q_emb2
                x_i.append(v_tmp)
            v_tuple=torch.stack(x_i, dim=1)
            v_tuple = self.linear1(v_tuple)
            x_new=torch.sum(v_tuple,dim=1)
            output.append(x_new)
        x_out=torch.stack(output, dim=1)
        x_out=self.linear2(x_out)
        return x_out

class RN10(nn.Module):
    def __init__(self,v_dim,q_dim,weight,norm,dropout):
        super(RN10, self).__init__()
        self.linear1=FCNet([v_dim, v_dim],dropout,weight,norm)
        self.linear2=FCNet([v_dim, v_dim],dropout,weight,norm)
        self.linear3 = FCNet([q_dim, v_dim], dropout, weight, norm)

    def forward(self, v,q):
        q_emb=self.linear3(q)
        batch=v.size(1)
        v_split=torch.split(v,1,dim=1)
        output=[]
        for i in range(batch):
            x_i=[]
            for j in range(batch):
                v_tmp=v_split[i].squeeze(1)*v_split[j].squeeze(1)
                v_tmp=v_tmp*q_emb
                x_i.append(v_tmp)
            v_tuple=torch.stack(x_i, dim=1)
            v_tuple = self.linear1(v_tuple)
            x_new=torch.sum(v_tuple,dim=1)
            output.append(x_new)
        x_out=torch.stack(output, dim=1)
        x_out=self.linear2(x_out)
        return x_out
class RN11(nn.Module):
    def __init__(self,v_dim,q_dim,weight,norm,dropout):
        super(RN11, self).__init__()
        self.linear1=FCNet([v_dim*2+q_dim, v_dim],dropout,weight,norm)
        #self.linear2=FCNet([v_dim, v_dim],dropout,weight,norm)
        self.linear3 = FCNet([q_dim, q_dim], dropout, weight, norm)

    def forward(self, v,q):
        q_emb=self.linear3(q)
        batch=v.size(1)
        v_split=torch.split(v,1,dim=1)
        output=[]
        for i in range(batch):
            x_i=[]
            for j in range(batch):
                v_tuple=torch.cat((v_split[i].squeeze(1),v_split[j].squeeze(1),q_emb),dim=1)
                x_i.append(v_tuple)
            v_tuple=torch.stack(x_i, dim=1)
            #v_tuple = self.linear1(v_tuple)
            x_new=torch.sum(v_tuple,dim=1)
            output.append(x_new)
        x_out=torch.stack(output, dim=1)
        x_out=self.linear1(x_out)
        return x_out
