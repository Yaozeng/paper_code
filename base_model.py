"""
Bilinear Attention Networks
Jin-Hwa Kim, Jaehyun Jun, Byoung-Tak Zhang
https://arxiv.org/abs/1805.07932

This code is written by Jin-Hwa Kim.
"""
from attention import *
from language_model import *
from classifier import SimpleClassifier
from fc import FCNet
from gate import *
from relation_network import *
import torch.nn as nn

class BaseModel(nn.Module):
    def __init__(self, w_emb, q_emb, v_att1, v_att2,q_net, v_net,rn,sfu,classifier):
        super(BaseModel, self).__init__()
        self.w_emb = w_emb
        self.q_emb = q_emb
        self.v_att1 = v_att1
        self.v_att2 = v_att2
        self.q_net = q_net
        self.v_net = v_net
        self.rn=rn
        self.sfu=sfu
        self.classifier = classifier

    def forward(self, v, b, q, labels):
        w_emb = self.w_emb(q)
        q_emb = self.q_emb(w_emb) # [batch, q_dim]

        rn_out = self.rn(v)

        att1 = self.v_att1(v, q_emb)
        att2 = self.v_att2(v, q_emb)
        att=att1+att2
        v_tmp = (att * v).sum(1)  # [batch, v_dim]

        att3 = self.v_att1(rn_out, q_emb)
        att4 = self.v_att2(rn_out, q_emb)
        att5 = att3 + att4
        v_tmp2 = (att5 * rn_out).sum(1)  # [batch, v_dim]

        v_emb=self.sfu(v_tmp,v_tmp2)

        q_repr = self.q_net(q_emb)
        v_repr = self.v_net(v_emb)
        joint_repr = q_repr * v_repr
        logits = self.classifier(joint_repr)
        return logits
class BaseModel1(nn.Module):
    def __init__(self, w_emb, q_emb, v_att,q_net, v_net,rn,sfu,classifier):
        super(BaseModel1, self).__init__()
        self.w_emb = w_emb
        self.q_emb = q_emb
        self.v_att = v_att
        self.q_net = q_net
        self.v_net = v_net
        self.rn=rn
        self.sfu=sfu
        self.classifier = classifier

    def forward(self, v, b, q, labels):
        w_emb = self.w_emb(q)
        q_emb = self.q_emb(w_emb) # [batch, q_dim]

        rn_out = self.rn(v)

        att1 = self.v_att(v, q_emb)
        v_tmp = (att1 * v).sum(1)  # [batch, v_dim]

        att2 = self.v_att(rn_out, q_emb)
        v_tmp2 = (att2 * rn_out).sum(1)  # [batch, v_dim]

        v_emb=self.sfu(v_tmp,v_tmp2)

        q_repr = self.q_net(q_emb)
        v_repr = self.v_net(v_emb)
        joint_repr = q_repr * v_repr
        logits = self.classifier(joint_repr)
        return logits

class BaseModel2(nn.Module):
    def __init__(self, w_emb, q_emb, v_att,q_net, v_net,rn,sfu,classifier):
        super(BaseModel2, self).__init__()
        self.w_emb = w_emb
        self.q_emb = q_emb
        self.v_att = v_att
        self.q_net = q_net
        self.v_net = v_net
        self.rn=rn
        self.sfu=sfu
        self.classifier = classifier

    def forward(self, v, b, q, labels):
        w_emb = self.w_emb(q)
        q_emb = self.q_emb(w_emb) # [batch, q_dim]

        rn_out = self.rn(v,q_emb)

        att1 = self.v_att(v, q_emb)
        v_tmp = (att1 * v).sum(1)  # [batch, v_dim]

        att2 = self.v_att(rn_out, q_emb)
        v_tmp2 = (att2 * rn_out).sum(1)  # [batch, v_dim]

        v_emb=self.sfu(v_tmp,v_tmp2)

        q_repr = self.q_net(q_emb)
        v_repr = self.v_net(v_emb)
        joint_repr = q_repr * v_repr
        logits = self.classifier(joint_repr)
        return logits
class BaseModel3(nn.Module):
    def __init__(self, w_emb, q_emb, v_att,q_net, v_net,rn,gate,classifier):
        super(BaseModel3, self).__init__()
        self.w_emb = w_emb
        self.q_emb = q_emb
        self.v_att = v_att
        self.q_net = q_net
        self.v_net = v_net
        self.rn=rn
        self.sfu=gate
        self.classifier = classifier

    def forward(self, v, b, q, labels):
        w_emb = self.w_emb(q)
        q_emb = self.q_emb(w_emb) # [batch, q_dim]

        rn_out = self.rn(v,q_emb)

        att1 = self.v_att(v, q_emb)
        v_tmp = (att1 * v).sum(1)  # [batch, v_dim]

        att2 = self.v_att(rn_out, q_emb)
        v_tmp2 = (att2 * rn_out).sum(1)  # [batch, v_dim]

        v_emb=self.sfu(v_tmp,v_tmp2)

        q_repr = self.q_net(q_emb)
        v_repr = self.v_net(v_emb)
        joint_repr = q_repr * v_repr
        logits = self.classifier(joint_repr)
        return logits

#final model struct
class BaseModel4(nn.Module):
    def __init__(self, w_emb, q_emb, v_att,q_net, v_net,rn,classifier):
        super(BaseModel4, self).__init__()
        self.w_emb = w_emb
        self.q_emb = q_emb
        self.v_att = v_att
        self.q_net = q_net
        self.v_net = v_net
        self.rn=rn
        self.classifier = classifier

    def forward(self, v, b, q, labels):
        w_emb = self.w_emb(q)
        q_emb = self.q_emb(w_emb) # [batch, q_dim]

        rn_out = self.rn(v,q_emb)

        att1 = self.v_att(v, q_emb)
        v_tmp = (att1 * v).sum(1)  # [batch, v_dim]

        att2 = self.v_att(rn_out, q_emb)
        v_tmp2 = (att2 * rn_out).sum(1)  # [batch, v_dim]

        v_emb=torch.cat((v_tmp,v_tmp2),dim=1)

        q_repr = self.q_net(q_emb)
        v_repr = self.v_net(v_emb)
        joint_repr = q_repr * v_repr
        logits = self.classifier(joint_repr)
        return logits,att1,att2
class BaseModel5(nn.Module):
    def __init__(self, w_emb, q_emb, v_att,q_net, v_net,rn,classifier):
        super(BaseModel5, self).__init__()
        self.w_emb = w_emb
        self.q_emb = q_emb
        self.v_att = v_att
        self.q_net = q_net
        self.v_net = v_net
        self.rn=rn
        self.classifier = classifier

    def forward(self, v, b, q, labels):
        w_emb = self.w_emb(q)
        q_emb = self.q_emb(w_emb) # [batch, q_dim]

        rn_out = self.rn(v,q_emb)

        att1 = self.v_att(v, q_emb)
        v_tmp = (att1 * v).sum(1)  # [batch, v_dim]

        att2 = self.v_att(rn_out, q_emb)
        v_tmp2 = (att2 * rn_out).sum(1)  # [batch, v_dim]

        v_emb=torch.cat((v_tmp,v_tmp2),dim=1)

        q_repr = self.q_net(q_emb)
        v_repr = self.v_net(v_emb)
        joint_repr = q_repr * v_repr
        logits = self.classifier(joint_repr)
        return logits
class BaseModel6(nn.Module):
    def __init__(self, w_emb, q_emb, v_att,q_net, v_net,rn,gate,classifier):
        super(BaseModel6, self).__init__()
        self.w_emb = w_emb
        self.q_emb = q_emb
        self.v_att = v_att
        self.rn=rn
        self.sfu=gate
        self.classifier = classifier

    def forward(self, v, b, q, labels):
        w_emb = self.w_emb(q)
        q_emb = self.q_emb.forward_all(w_emb)

        rn_out = self.rn(v,q_emb)

        v_tmp = self.v_att(v, q_emb)

        v_tmp2 = self.v_att(rn_out, q_emb)

        v_emb=self.sfu(v_tmp,v_tmp2)

        logits = self.classifier(v_emb)
        return logits
class BaseModel7(nn.Module):
    def __init__(self, w_emb, q_emb, v_att,q_net, v_net,rn,classifier):
        super(BaseModel7, self).__init__()
        self.w_emb = w_emb
        self.q_emb = q_emb
        self.v_att = v_att
        self.q_net = q_net
        self.v_net = v_net
        self.rn=rn
        self.classifier = classifier

    def forward(self, v, b, q, labels):
        w_emb = self.w_emb(q)
        q_emb = self.q_emb(w_emb) # [batch, q_dim]

        rn_out = self.rn(v,q_emb)

        att1 = self.v_att(v, q_emb)
        v_tmp = att1 * v

        att2 = self.v_att(rn_out, q_emb)
        v_tmp2 = att2 * rn_out

        v_emb=torch.cat((v_tmp,v_tmp2),dim=2)
        v_emb=v_emb.sum(1)

        q_repr = self.q_net(q_emb)
        v_repr = self.v_net(v_emb)
        joint_repr = q_repr * v_repr
        logits = self.classifier(joint_repr)
        return logits
class BaseModel8(nn.Module):
    def __init__(self, w_emb, q_emb, v_att,q_net, v_net,rn,classifier):
        super(BaseModel8, self).__init__()
        self.w_emb = w_emb
        self.q_emb = q_emb
        self.v_att = v_att
        self.q_net = q_net
        self.v_net = v_net
        self.rn=rn
        self.classifier = classifier

    def forward(self, v, b, q, labels):
        w_emb = self.w_emb(q)
        q_emb = self.q_emb(w_emb) # [batch, q_dim]

        rn_out = self.rn(v,q_emb)

        att1 = self.v_att(v, q_emb)
        v_tmp = att1 * v

        att2 = self.v_att(rn_out, q_emb)
        v_tmp2 = att2 * rn_out

        v_emb=torch.cat((v_tmp,v_tmp2),dim=2)
        v_emb=v_emb.sum(1)

        q_repr = self.q_net(q_emb)
        v_repr = self.v_net(v_emb)
        joint_repr = q_repr * v_repr
        logits = self.classifier(joint_repr)
        return logits
class BaseModel9(nn.Module):
    def __init__(self, w_emb, q_emb, v_att,q_net, v_net,rn,sfu,classifier):
        super(BaseModel9, self).__init__()
        self.w_emb = w_emb
        self.q_emb = q_emb
        self.v_att = v_att
        self.q_net = q_net
        self.v_net = v_net
        self.rn=rn
        self.sfu=sfu
        self.classifier = classifier

    def forward(self, v, b, q, labels):
        w_emb = self.w_emb(q)
        q_emb = self.q_emb(w_emb) # [batch, q_dim]

        rn_out = self.rn(v,q_emb)

        att1 = self.v_att(v, q_emb)
        v_tmp =  att1 * v

        att2 = self.v_att(rn_out, q_emb)
        v_tmp2 = att2 * rn_out

        v_emb=self.sfu(v_tmp,v_tmp2)
        v_emb=v_emb.sum(1)

        q_repr = self.q_net(q_emb)
        v_repr = self.v_net(v_emb)
        joint_repr = q_repr * v_repr
        logits = self.classifier(joint_repr)
        return logits
class BaseModel10(nn.Module):
    def __init__(self, w_emb, q_emb, v_att,q_net, v_net,rn,sfu,classifier):
        super(BaseModel10, self).__init__()
        self.w_emb = w_emb
        self.q_emb = q_emb
        self.v_att = v_att
        self.q_net = q_net
        self.v_net = v_net
        self.rn=rn
        self.sfu=sfu
        self.classifier = classifier

    def forward(self, v, b, q, labels):
        w_emb = self.w_emb(q)
        q_emb = self.q_emb(w_emb) # [batch, q_dim]

        rn_out = self.rn(v,q_emb)
        v_com=self.sfu(rn_out,v)

        att = self.v_att(v_com, q_emb)
        v_emb =  (att * v_com).sum(1)

        q_repr = self.q_net(q_emb)
        v_repr = self.v_net(v_emb)
        joint_repr = q_repr * v_repr
        logits = self.classifier(joint_repr)
        return logits
class BaseModel12(nn.Module):
    def __init__(self, w_emb, q_emb, v_att,q_net, v_net,rn,sfu,classifier):
        super(BaseModel12, self).__init__()
        self.w_emb = w_emb
        self.q_emb = q_emb
        self.v_att = v_att
        self.q_net = q_net
        self.v_net = v_net
        self.rn=rn
        self.sfu=sfu
        self.classifier = classifier

    def forward(self, v, b, q, labels):
        w_emb = self.w_emb(q)
        q_emb = self.q_emb(w_emb)  # [batch, q_dim]

        rn_out = self.rn(v, q_emb)

        att1 = self.v_att(v, q_emb)
        v_tmp = (att1 * v).sum(1)  # [batch, v_dim]

        att2 = self.v_att(rn_out, q_emb)
        v_tmp2 = (att2 * v).sum(1)  # [batch, v_dim]

        v_emb = self.sfu(v_tmp,v_tmp2)

        q_repr = self.q_net(q_emb)
        v_repr = self.v_net(v_emb)
        joint_repr = q_repr * v_repr
        logits = self.classifier(joint_repr)
        return logits
class BaseModel11(nn.Module):
    def __init__(self, w_emb, q_emb, v_att,q_net, v_net,rn,classifier):
        super(BaseModel11, self).__init__()
        self.w_emb = w_emb
        self.q_emb = q_emb
        self.v_att = v_att
        self.q_net = q_net
        self.v_net = v_net
        self.rn=rn
        self.classifier = classifier

    def forward(self, v, b, q, labels):
        w_emb = self.w_emb(q)
        q_emb = self.q_emb(w_emb) # [batch, q_dim]

        rn_out = self.rn(v,q_emb)
        v_com=torch.cat((v,rn_out),dim=2)

        att=self.v_att(v_com,q_emb)
        v_emb=(v_com*att).sum(1)

        q_repr = self.q_net(q_emb)
        v_repr = self.v_net(v_emb)
        joint_repr = q_repr * v_repr
        logits = self.classifier(joint_repr)
        return logits,att
class BaseModel13(nn.Module):
    def __init__(self, w_emb, q_emb, v_att,q_net, v_net,classifier):
        super(BaseModel13, self).__init__()
        self.w_emb = w_emb
        self.q_emb = q_emb
        self.v_att = v_att
        self.q_net = q_net
        self.v_net = v_net

        self.classifier = classifier

    def forward(self, v, b, q, labels):
        w_emb = self.w_emb(q)
        q_emb = self.q_emb(w_emb) # [batch, q_dim]

        att = self.v_att(v, q_emb)
        v_emb = (att * v).sum(1)  # [batch, v_dim]

        q_repr = self.q_net(q_emb)
        v_repr = self.v_net(v_emb)
        joint_repr = q_repr * v_repr
        logits = self.classifier(joint_repr)
        return logits


class BaseModel14(nn.Module):
    def __init__(self, w_emb, q_emb, v_att, q_net, v_net, rn,classifier):
        super(BaseModel14, self).__init__()
        self.w_emb = w_emb
        self.q_emb = q_emb
        self.rn=rn
        self.v_att = v_att
        self.q_net = q_net
        self.v_net = v_net

        self.classifier = classifier

    def forward(self, v, b, q, labels):
        w_emb = self.w_emb(q)
        q_emb = self.q_emb(w_emb)  # [batch, q_dim]

        rn_out = self.rn(v, q_emb)

        att = self.v_att(rn_out, q_emb)
        v_emb = (att * rn_out).sum(1)  # [batch, v_dim]

        q_repr = self.q_net(q_emb)
        v_repr = self.v_net(v_emb)
        joint_repr = q_repr * v_repr
        logits = self.classifier(joint_repr)
        return logits
class selfattention(nn.Module):
    def __init__(self, w_emb, q_emb, v_att, q_net, v_net, rn,classifier):
        super(selfattention, self).__init__()
        self.w_emb = w_emb
        self.q_emb = q_emb
        self.rn=rn
        self.v_att = v_att
        self.q_net = q_net
        self.v_net = v_net

        self.classifier = classifier

    def forward(self, v, b, q, labels):
        w_emb = self.w_emb(q)
        q_emb = self.q_emb(w_emb)  # [batch, q_dim]

        rn_out = self.rn(v, q_emb)

        att = self.v_att(rn_out, q_emb)
        v_emb = (att * rn_out).sum(1)  # [batch, v_dim]

        q_repr = self.q_net(q_emb)
        v_repr = self.v_net(v_emb)
        joint_repr = q_repr * v_repr
        logits = self.classifier(joint_repr)
        return logits
def build_baseline(dataset, num_hid):
    w_emb = WordEmbedding(dataset.dictionary.ntoken, 300, 0.0)
    q_emb = QuestionEmbedding(300, num_hid, 1, False, 0.0)
    v_att1=Att_3(dataset.v_dim,num_hid,num_hid,"weight","ReLU")
    v_att2 = Att_3(dataset.v_dim, num_hid, num_hid, "weight", "ReLU")
    q_net = FCNet([num_hid, num_hid],0.0,"weight","ReLU")
    v_net = FCNet([dataset.v_dim, num_hid],0.0,"weight","ReLU")
    rn=RN(dataset.v_dim,"weight","ReLU",0.0)
    sfu=SFU(dataset.v_dim,dataset.v_dim)
    classifier = SimpleClassifier(
        num_hid, 5000, dataset.num_ans_candidates, 0.5)
    return BaseModel(w_emb, q_emb, v_att1, v_att2,q_net, v_net,rn,sfu,classifier)
def build_baseline1(dataset, num_hid):
    w_emb = WordEmbedding(dataset.dictionary.ntoken, 300, 0.0)
    q_emb = QuestionEmbedding(300, num_hid, 1, False, 0.0)
    v_att=Att_3(dataset.v_dim,num_hid,num_hid,"weight","ReLU")
    q_net = FCNet([num_hid, num_hid],0.0,"weight","ReLU")
    v_net = FCNet([dataset.v_dim, num_hid],0.0,"weight","ReLU")
    rn=RN(dataset.v_dim,"weight","ReLU",0.0)
    sfu=SFU(dataset.v_dim,dataset.v_dim)
    classifier = SimpleClassifier(
        num_hid, 5000, dataset.num_ans_candidates, 0.5)
    return BaseModel1(w_emb, q_emb, v_att,q_net, v_net,rn,sfu,classifier)
def build_baseline2(dataset, num_hid):
    w_emb = WordEmbedding(dataset.dictionary.ntoken, 300, 0.0)
    q_emb = QuestionEmbedding(300, num_hid, 1, False, 0.0)
    v_att=doubel_project_attention(dataset.v_dim,q_emb.num_hid,q_emb.num_hid,0.2)
    q_net = FCNet([num_hid, num_hid],0.0,"weight","ReLU")
    v_net = FCNet([dataset.v_dim, num_hid],0.0,"weight","ReLU")
    rn=RN(dataset.v_dim,"weight","ReLU",0.0)
    sfu=SFU(dataset.v_dim,dataset.v_dim)
    classifier = SimpleClassifier(
        num_hid, 5000, dataset.num_ans_candidates, 0.5)
    return BaseModel1(w_emb, q_emb, v_att,q_net, v_net,rn,sfu,classifier)
def build_baseline3(dataset, num_hid):
    w_emb = WordEmbedding(dataset.dictionary.ntoken, 300, 0.0)
    q_emb = AttQuestionEmbedding(300,1024,1,0,512,1,2,0)
    v_att = doubel_project_attention(dataset.v_dim, q_emb.num_hid, q_emb.num_hid, 0.2)
    q_net = FCNet([num_hid, num_hid],0.0,"weight","ReLU")
    v_net = FCNet([dataset.v_dim, num_hid],0.0,"weight","ReLU")
    rn=RN(dataset.v_dim,"weight","ReLU",0.0)
    sfu=SFU(dataset.v_dim,dataset.v_dim)
    classifier = SimpleClassifier(
        num_hid, 5000, dataset.num_ans_candidates, 0.5)
    return BaseModel1(w_emb, q_emb, v_att,q_net, v_net,rn,sfu,classifier)

def build_baseline4(dataset, num_hid):
    w_emb = WordEmbedding(dataset.dictionary.ntoken, 300, 0.0)
    q_emb = AttQuestionEmbedding(300,1024,1,0,512,1,2,0)
    v_att = doubel_project_attention(dataset.v_dim, q_emb.num_hid, q_emb.num_hid, 0.2)
    q_net = FCNet([num_hid, num_hid],0.0,"weight","ReLU")
    v_net = FCNet([dataset.v_dim, num_hid],0.0,"weight","ReLU")
    rn=RN4(dataset.v_dim,num_hid,"weight","ReLU",0.0)
    sfu=SFU(dataset.v_dim,dataset.v_dim)
    classifier = SimpleClassifier(
        num_hid, 5000, dataset.num_ans_candidates, 0.5)
    return BaseModel1(w_emb, q_emb, v_att,q_net, v_net,rn,sfu,classifier)

def build_baseline5(dataset, num_hid):
    w_emb = WordEmbedding(dataset.dictionary.ntoken, 300, 0.0)
    q_emb = QuestionEmbedding(300, num_hid, 1, False, 0.0)
    v_att = doubel_project_attention(dataset.v_dim, num_hid,num_hid, 0.2)
    q_net = FCNet([num_hid, num_hid],0.0,"weight","ReLU")
    v_net = FCNet([dataset.v_dim, num_hid],0.0,"weight","ReLU")
    rn=RN4(dataset.v_dim,num_hid,"weight","ReLU",0.2)
    sfu=SFU(dataset.v_dim,dataset.v_dim)
    classifier = SimpleClassifier(
        num_hid, 2*num_hid, dataset.num_ans_candidates, 0.5)
    return BaseModel2(w_emb, q_emb, v_att,q_net, v_net,rn,sfu,classifier)
def build_baseline6(dataset, num_hid):
    w_emb = WordEmbedding(dataset.dictionary.ntoken, 300, 0.0)
    q_emb = QuestionEmbedding(300, num_hid, 1, False, 0.0)
    v_att = doubel_project_attention(dataset.v_dim, num_hid,num_hid, 0.2)
    q_net = FCNet([num_hid, num_hid],0.0,"weight","ReLU")
    v_net = FCNet([dataset.v_dim, num_hid],0.0,"weight","ReLU")
    rn=RN4(dataset.v_dim,num_hid,"weight","ReLU",0.2)
    sfu=Gate(dataset.v_dim)
    classifier = SimpleClassifier(
        num_hid, 2*num_hid, dataset.num_ans_candidates, 0.5)
    return BaseModel3(w_emb, q_emb, v_att,q_net, v_net,rn,sfu,classifier)
def build_baseline7(dataset, num_hid):
    w_emb = WordEmbedding(dataset.dictionary.ntoken, 300, 0.0)
    q_emb = QuestionEmbedding(300, num_hid, 1, False, 0.0)
    v_att = project_attention(dataset.v_dim, num_hid,num_hid, 0.2)
    q_net = FCNet([num_hid, num_hid],0.0,"weight","ReLU")
    v_net = FCNet([dataset.v_dim*2, num_hid],0.0,"weight","ReLU")
    rn=RN4(dataset.v_dim,num_hid,"weight","ReLU",0.0)
    classifier = SimpleClassifier(
        num_hid, 2*num_hid, dataset.num_ans_candidates, 0.5)
    return BaseModel4(w_emb, q_emb, v_att,q_net, v_net,rn,classifier)
def build_baseline8(dataset, num_hid):
    w_emb = WordEmbedding(dataset.dictionary.ntoken, 300, 0.0)
    q_emb = QuestionEmbedding(300, num_hid, 1, False, 0.0)
    v_att = project_attention(dataset.v_dim, num_hid,num_hid, 0.2)
    q_net = FCNet([num_hid, num_hid],0.0,"weight","ReLU")
    v_net = FCNet([dataset.v_dim, num_hid],0.0,"weight","ReLU")
    rn=RN4(dataset.v_dim,num_hid,"weight","ReLU",0.2)
    sfu=SFU(dataset.v_dim,dataset.v_dim)
    classifier = SimpleClassifier(
        num_hid, 2*num_hid, dataset.num_ans_candidates, 0.5)
    return BaseModel2(w_emb, q_emb, v_att, q_net, v_net, rn, sfu, classifier)
def build_baseline9(dataset, num_hid):
    w_emb = WordEmbedding(dataset.dictionary.ntoken, 300, 0.0)
    q_emb = QuestionEmbedding(300, num_hid, 1, False, 0.0)
    v_att = BiAttention2(dataset.v_dim, num_hid,num_hid)
    q_net = FCNet([num_hid, num_hid],0.0,"weight","ReLU")
    v_net = FCNet([dataset.v_dim, num_hid],0.0,"weight","ReLU")
    rn=RN4(dataset.v_dim,num_hid,"weight","ReLU",0.2)
    sfu=SFU(dataset.v_dim,dataset.v_dim)
    classifier = SimpleClassifier(
        num_hid, 2*num_hid, dataset.num_ans_candidates, 0.5)
    return BaseModel5(w_emb, q_emb, v_att, q_net, v_net, rn, classifier)
def build_baseline10(dataset, num_hid):
    w_emb = WordEmbedding(dataset.dictionary.ntoken, 300, 0.0)
    q_emb = QuestionEmbedding(300, num_hid, 1, False, 0.0)
    v_att = project_attention(dataset.v_dim, num_hid,num_hid)
    q_net = FCNet([num_hid, num_hid],0.0,"weight","ReLU")
    v_net = FCNet([dataset.v_dim, num_hid],0.0,"weight","ReLU")
    rn=RN4(dataset.v_dim,num_hid,"weight","ReLU",0.2)
    sfu=SFU(dataset.v_dim,dataset.v_dim)
    classifier = SimpleClassifier(
        num_hid, 2*num_hid, dataset.num_ans_candidates, 0.5)
    return BaseModel8(w_emb, q_emb, v_att, q_net, v_net, rn,classifier)
def build_baseline11(dataset, num_hid):
    w_emb = WordEmbedding(dataset.dictionary.ntoken, 300, 0.0)
    q_emb = QuestionEmbedding(300, num_hid, 1, False, 0.0)
    v_att = project_attention(dataset.v_dim, num_hid,num_hid)
    q_net = FCNet([num_hid, num_hid],0.0,"weight","ReLU")
    v_net = FCNet([dataset.v_dim, num_hid],0.0,"weight","ReLU")
    rn=RN4(dataset.v_dim,num_hid,"weight","ReLU",0.2)
    sfu=SFU3(dataset.v_dim,dataset.v_dim)
    classifier = SimpleClassifier(
        num_hid, 2*num_hid, dataset.num_ans_candidates, 0.5)
    return BaseModel9(w_emb, q_emb, v_att, q_net, v_net, rn, sfu, classifier)
def build_baseline12(dataset, num_hid):
    w_emb = WordEmbedding(dataset.dictionary.ntoken, 300, 0.0)
    q_emb = QuestionEmbedding(300, num_hid, 1, False, 0.0)
    v_att = doubel_project_attention(dataset.v_dim, num_hid,num_hid, 0.2)
    q_net = FCNet([num_hid, num_hid],0.0,"weight","ReLU")
    v_net = FCNet([dataset.v_dim*2, num_hid],0.0,"weight","ReLU")
    rn=RN4(dataset.v_dim,num_hid,"weight","ReLU",0.2)
    classifier = SimpleClassifier(
        num_hid, 2*num_hid, dataset.num_ans_candidates, 0.5)
    return BaseModel8(w_emb, q_emb, v_att,q_net, v_net,rn,classifier)
def build_baseline13(dataset, num_hid):
    w_emb = WordEmbedding(dataset.dictionary.ntoken, 300, 0.0)
    q_emb = QuestionEmbedding(300, num_hid, 1, False, 0.0)
    v_att = doubel_project_attention(dataset.v_dim*2, num_hid,num_hid, 0.2)
    q_net = FCNet([num_hid, num_hid],0.0,"weight","ReLU")
    v_net = FCNet([dataset.v_dim*2, num_hid],0.0,"weight","ReLU")
    rn=RN4(dataset.v_dim,num_hid,"weight","ReLU",0.2)
    sfu = SFU3(dataset.v_dim, dataset.v_dim)
    classifier = SimpleClassifier(
        num_hid, 2*num_hid, dataset.num_ans_candidates, 0.5)
    return BaseModel10(w_emb, q_emb, v_att,q_net, v_net,rn,sfu,classifier)
def build_baseline14(dataset, num_hid):
    w_emb = WordEmbedding(dataset.dictionary.ntoken, 300, 0.0)
    q_emb = QuestionEmbedding(300, num_hid, 1, False, 0.0)
    v_att = project_attention(dataset.v_dim*2, num_hid,num_hid, 0.2)
    q_net = FCNet([num_hid, num_hid],0.0,"weight","LeakyReLU")
    v_net = FCNet([dataset.v_dim*2, num_hid],0.0,"weight","LeakyReLU")
    rn=RN4(dataset.v_dim,num_hid,"weight","LeakyReLU",0.0)
    classifier = SimpleClassifier(
        num_hid, 2*num_hid, dataset.num_ans_candidates, 0.5)
    return BaseModel11(w_emb, q_emb, v_att,q_net, v_net,rn,classifier)
def build_baseline15(dataset, num_hid):
    w_emb = WordEmbedding(dataset.dictionary.ntoken, 300, 0.0)
    q_emb = QuestionEmbedding(300, num_hid, 1, False, 0.0)
    v_att = doubel_project_attention(dataset.v_dim*2, num_hid,num_hid, 0.2)
    q_net = FCNet([num_hid, num_hid],0.0,"weight","ReLU")
    v_net = FCNet([dataset.v_dim*2, num_hid],0.0,"weight","ReLU")
    rn=RN4(dataset.v_dim,num_hid,"weight","ReLU",0.2)
    sfu = SFU4(dataset.v_dim, dataset.v_dim)
    classifier = SimpleClassifier(
        num_hid, 2*num_hid, dataset.num_ans_candidates, 0.5)
    return BaseModel10(w_emb, q_emb, v_att,q_net, v_net,rn,sfu,classifier)
def build_baseline16(dataset, num_hid):
    w_emb = WordEmbedding(dataset.dictionary.ntoken, 300, 0.0)
    q_emb = QuestionEmbedding(300, num_hid, 1, False, 0.0)
    v_att = project_attention(dataset.v_dim, num_hid,num_hid, 0.2)
    q_net = FCNet([num_hid, num_hid],0.2,"weight","ReLU")
    v_net = FCNet([dataset.v_dim, num_hid],0.2,"weight","ReLU")
    rn=RN4(dataset.v_dim,num_hid,"weight","ReLU",0.0)
    sfu = SFU(dataset.v_dim, dataset.v_dim)
    classifier = SimpleClassifier(
        num_hid, 2*num_hid, dataset.num_ans_candidates, 0.5)
    return BaseModel12(w_emb, q_emb, v_att,q_net, v_net,rn,sfu,classifier)

#final model最好的模型结构
def build_baseline17(dataset, num_hid):
    w_emb = WordEmbedding(dataset.dictionary.ntoken, 300, 0.0)
    q_emb = QuestionEmbedding(300, num_hid, 1, False, 0.0)
    v_att = project_attention(dataset.v_dim, num_hid,num_hid, 0.2)
    q_net = FCNet([num_hid, num_hid],0.0,"weight","LeakyReLU")
    v_net = FCNet([dataset.v_dim*2, num_hid],0.0,"weight","LeakyReLU")
    rn=RN4(dataset.v_dim,num_hid,"weight","LeakyReLU",0.0)
    classifier = SimpleClassifier(
        num_hid, 2*num_hid, dataset.num_ans_candidates, 0.5)
    return BaseModel4(w_emb, q_emb, v_att,q_net, v_net,rn,classifier)

def build_baseline18(dataset, num_hid):
    w_emb = WordEmbedding(dataset.dictionary.ntoken, 300, 0.0)
    q_emb = QuestionEmbedding(300, num_hid, 1, False, 0.0)
    v_att = project_attention(dataset.v_dim, num_hid,num_hid, 0.2)
    q_net = FCNet([num_hid, num_hid],0.0,"weight","ReLU")
    v_net = FCNet([dataset.v_dim*2, num_hid],0.0,"weight","ReLU")
    rn=RN4(dataset.v_dim,num_hid,"weight","ReLU",0.0)
    classifier = SimpleClassifier(
        num_hid, 2*num_hid, dataset.num_ans_candidates, 0.5)
    return BaseModel4(w_emb, q_emb, v_att,q_net, v_net,rn,classifier)
#xiaorongshiyan
#zhiyou yuanlaide
def build_baseline19(dataset, num_hid):
    w_emb = WordEmbedding(dataset.dictionary.ntoken, 300, 0.0)
    q_emb = QuestionEmbedding(300, num_hid, 1, False, 0.0)
    v_att = project_attention(dataset.v_dim, num_hid,num_hid, 0.2)
    q_net = FCNet([num_hid, num_hid],0.0,"weight","LeakyReLU")
    v_net = FCNet([dataset.v_dim, num_hid],0.0,"weight","LeakyReLU")
    classifier = SimpleClassifier(
        num_hid, 2*num_hid, dataset.num_ans_candidates, 0.5)
    return BaseModel13(w_emb, q_emb, v_att,q_net, v_net,classifier)
#zhiyou houlaide
def build_baseline20(dataset, num_hid):
    w_emb = WordEmbedding(dataset.dictionary.ntoken, 300, 0.0)
    q_emb = QuestionEmbedding(300, num_hid, 1, False, 0.0)
    v_att = project_attention(dataset.v_dim, num_hid,num_hid, 0.2)
    q_net = FCNet([num_hid, num_hid],0.0,"weight","LeakyReLU")
    v_net = FCNet([dataset.v_dim, num_hid],0.0,"weight","LeakyReLU")
    rn=RN4(dataset.v_dim,num_hid,"weight","ReLU",0.0)
    classifier = SimpleClassifier(
        num_hid, 2*num_hid, dataset.num_ans_candidates, 0.5)
    return BaseModel14(w_emb, q_emb, v_att,q_net, v_net,rn,classifier)


#concat attention对比
def build_baseline21(dataset, num_hid):
    w_emb = WordEmbedding(dataset.dictionary.ntoken, 300, 0.0)
    q_emb = QuestionEmbedding(300, num_hid, 1, False, 0.0)
    v_att = concatenate_attention(dataset.v_dim, num_hid,num_hid)
    q_net = FCNet([num_hid, num_hid],0.0,"weight","LeakyReLU")
    v_net = FCNet([dataset.v_dim*2, num_hid],0.0,"weight","LeakyReLU")
    rn=RN4(dataset.v_dim,num_hid,"weight","LeakyReLU",0.0)
    classifier = SimpleClassifier(
        num_hid, 2*num_hid, dataset.num_ans_candidates, 0.5)
    return BaseModel4(w_emb, q_emb, v_att,q_net, v_net,rn,classifier)

def build_baseline22(dataset, num_hid):
    w_emb = WordEmbedding(dataset.dictionary.ntoken, 300, 0.0)
    q_emb = QuestionEmbedding(300, num_hid, 1, False, 0.0)
    v_att = project_attention(dataset.v_dim, num_hid,num_hid, 0.2)
    q_net = FCNet([num_hid, num_hid],0.0,"weight","LeakyReLU")
    v_net = FCNet([dataset.v_dim*2, num_hid],0.0,"weight","LeakyReLU")
    rn=RN6(dataset.v_dim,num_hid,"weight","LeakyReLU",0.0)
    classifier = SimpleClassifier(
        num_hid, 2*num_hid, dataset.num_ans_candidates, 0.5)
    return BaseModel4(w_emb, q_emb, v_att,q_net, v_net,rn,classifier)

def build_baseline23(dataset, num_hid):
    w_emb = WordEmbedding(dataset.dictionary.ntoken, 300, 0.0)
    q_emb = QuestionEmbedding(300, num_hid, 1, False, 0.0)
    v_att = project_attention(dataset.v_dim, num_hid,num_hid, 0.2)
    q_net = FCNet([num_hid, num_hid],0.0,"weight","LeakyReLU")
    v_net = FCNet([dataset.v_dim*2, num_hid],0.0,"weight","LeakyReLU")
    rn=RN8(dataset.v_dim,num_hid,"weight","LeakyReLU",0.0)
    classifier = SimpleClassifier(
        num_hid, 2*num_hid, dataset.num_ans_candidates, 0.5)
    return BaseModel4(w_emb, q_emb, v_att,q_net, v_net,rn,classifier)

def build_baseline24(dataset, num_hid):
    w_emb = WordEmbedding(dataset.dictionary.ntoken, 300, 0.0)
    q_emb = QuestionEmbedding(300, num_hid, 1, False, 0.0)
    v_att = project_attention(dataset.v_dim, num_hid,num_hid, 0.2)
    q_net = FCNet([num_hid, num_hid],0.0,"weight","LeakyReLU")
    v_net = FCNet([dataset.v_dim*2, num_hid],0.0,"weight","LeakyReLU")
    rn=RN10(dataset.v_dim,num_hid,"weight","LeakyReLU",0.0)
    classifier = SimpleClassifier(
        num_hid, 2*num_hid, dataset.num_ans_candidates, 0.5)
    return BaseModel4(w_emb, q_emb, v_att,q_net, v_net,rn,classifier)

class BaseModel0(nn.Module):
    def __init__(self, w_emb, q_emb, v_att, q_net, v_net, classifier):
        super(BaseModel0, self).__init__()
        self.w_emb = w_emb
        self.q_emb = q_emb
        self.v_att = v_att
        self.q_net = q_net
        self.v_net = v_net
        self.classifier = classifier

    def forward(self, v, b, q, labels):
        """Forward
        v: [batch, num_objs, obj_dim]
        b: [batch, num_objs, b_dim]
        q: [batch_size, seq_length]
        return: logits, not probs
        """
        w_emb = self.w_emb(q)
        q_emb = self.q_emb(w_emb) # [batch, q_dim]

        att = self.v_att(v, q_emb)
        v_emb = (att * v).sum(1) # [batch, v_dim]

        q_repr = self.q_net(q_emb)
        v_repr = self.v_net(v_emb)
        joint_repr = q_repr * v_repr
        logits = self.classifier(joint_repr)
        return logits,att


def build_baseline0(dataset, num_hid):
    w_emb = WordEmbedding(dataset.dictionary.ntoken, 300, 0.0)
    q_emb = QuestionEmbedding(300, num_hid, 1, False, 0.0)
    v_att = Attention(dataset.v_dim, q_emb.num_hid, num_hid)
    q_net = FCNet([num_hid, num_hid],0.0,"weight","ReLU")
    v_net = FCNet([dataset.v_dim, num_hid],0.0,"weight","ReLU")
    classifier = SimpleClassifier(
        num_hid, 2 * num_hid, dataset.num_ans_candidates, 0.5)
    return BaseModel0(w_emb, q_emb, v_att, q_net, v_net, classifier)
