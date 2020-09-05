import torch
import torch.nn as nn
from fc import *

class Gate(nn.Module):
    def __init__(self, v_dim):
        super(Gate, self).__init__()
        self.linear = FCNet([v_dim*2, v_dim],0.2,"weight","ReLU")

    def forward(self, v1,v2):
        v_cat=torch.cat((v1,v2),dim=1)
        v_proj = self.linear(v_cat)
        gate = torch.sigmoid(v_proj)
        return v1 * gate+v2*(1-gate)

class SFU(nn.Module):
    def __init__(self, input_size, output_size):
        super(SFU, self).__init__()
        self.linear_r = FCNet([input_size*2, output_size],0.0,"weight","ReLU")
        self.linear_g = FCNet([input_size*2, output_size],0.0,"weight","ReLU")

    def forward(self, x, fusions):
        r_f = torch.cat([x, fusions], 1)
        r = torch.tanh(self.linear_r(r_f))
        g = torch.sigmoid(self.linear_g(r_f))
        o = x*r+fusions*g
        return o
class SFU2(nn.Module):
    def __init__(self, input_size, output_size):
        super(SFU2, self).__init__()
        self.linear_r = FCNet([input_size*2, output_size],0.0,"weight","ReLU")
        self.linear_g = FCNet([input_size*2, output_size],0.0,"weight","ReLU")

    def forward(self, x, fusions):
        r_f = torch.cat([x, fusions], 2)
        r = torch.tanh(self.linear_r(r_f))
        g = torch.sigmoid(self.linear_g(r_f))
        o = g*r+(1-g)*x
        return o
class SFU3(nn.Module):
    def __init__(self, input_size, output_size):
        super(SFU3, self).__init__()
        self.linear_r = FCNet([input_size*2, output_size],0.0,"weight","ReLU")
        self.linear_g = FCNet([input_size*2, output_size],0.0,"weight","ReLU")

    def forward(self, x, fusions):
        r_f = torch.cat([x, fusions], 2)
        r = torch.tanh(self.linear_r(r_f))
        g = torch.sigmoid(self.linear_g(r_f))
        o = x*r+fusions*g
        return o
class SFU4(nn.Module):
    def __init__(self, input_size, output_size):
        super(SFU4, self).__init__()
        self.linear= FCNet([input_size*2, output_size],0.0,"weight","ReLU")

    def forward(self, x, fusions):
        r_f = torch.cat([x, fusions], 2)
        r = torch.tanh(self.linear(r_f))
        o = x+r*fusions
        return o