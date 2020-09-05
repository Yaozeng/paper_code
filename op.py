import torch.nn as nn
import torch
import torch.nn.functional as F
# import spacy


class Linear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, seed=None):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        if seed:
            torch.manual_seed(seed)
        self.linear = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x):
        if x.size()[-1] != self.in_features:
            raise ValueError(
                '[error] putils.Linear(%s, %s): last dimension of input(%s) should equal to in_features(%s)' %
                (self.in_features, self.out_features, x.size(-1), self.in_features))
        return self.linear(x)

    def __call__(self, *input, **kwargs):
        return super().__call__(*input, **kwargs)


class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, seed=None):
        super(Conv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        if seed:
            torch.manual_seed(seed)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                              padding=padding, dilation=1, groups=1, bias=True)

    def forward(self, x):
        if x.dim() != 4:
            raise ValueError('[error] putils.Conv2d(%s, %s, %s, %s): input_dim (%s) should equal to 4' %
                             (self.in_channels, self.out_channels, self.kernel_size, self.stride, x.dim()))

        # x: b*7*7*512
        x = x.transpose(2, 3).transpose(1, 2)  # b*512*7*7
        x = self.conv(x)  # b*450*7*7
        x = x.transpose(1, 2).transpose(2, 3)  # b*7*7*450
        return x

    def __call__(self, *input, **kwargs):
        return super().__call__(*input, **kwargs)


class Conv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, seed=None):
        super(Conv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        if seed:
            torch.manual_seed(seed)
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride,
                              padding=padding, dilation=1, groups=1, bias=True)

    def forward(self, x):
        if x.dim() != 3:
            raise ValueError('[error] putils.Conv1d(%s, %s, %s, %s): input_dim (%s) should equal to 3' %
                             (self.in_channels, self.out_channels, self.kernel_size, self.stride, x.dim()))
        # x: b*49*512
        x = x.transpose(1, 2)  # b*512*49
        x = self.conv(x)  # b*450*49
        x = x.transpose(1, 2)  # b*49*450
        return x

    def __call__(self, *input, **kwargs):
        return super().__call__(*input, **kwargs)


def bmatmul(inputs1, inputs2):
    b = inputs1.size()[0]
    m = []
    for i in range(b):
        m.append(torch.matmul(inputs1[i], inputs2[i]))
    outputs = torch.stack(m, dim=0)
    return outputs


def bmul(inputs1, inputs2):
    b = inputs1.size()[0]
    m = []
    for i in range(b):
        m.append(inputs1[i] * inputs2[i])
    outputs = torch.stack(m, dim=0)
    return outputs


def bmul3(inputs1, inputs2, inputs3):
    b = inputs1.size()[0]
    m = []
    for i in range(b):
        m.append(inputs1[i] * inputs2[i] * inputs3[i])
    outputs = torch.stack(m, dim=0)
    return outputs


def badd(inputs, inputs2):
    b = inputs.size()[0]
    m = []
    for i in range(b):
        m.append(inputs[i] + inputs2[i])
    outputs = torch.stack(m, dim=0)
    return outputs

class MyConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, seed=None, p=None, af=None,
                 dim=None):
        super(MyConv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.p = p
        self.af = af
        self.dim = dim
        if seed:
            torch.manual_seed(seed)
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride,
                              padding=padding, dilation=1, groups=1, bias=True)

    def forward(self, x):
        if x.dim() != 3:
            raise ValueError('[error] putils.Conv1d(%s, %s, %s, %s): input_dim (%s) should equal to 3' %
                             (self.in_channels, self.out_channels, self.kernel_size, self.stride, x.dim()))
        # x: b*49*512
        if self.p:
            x = F.dropout(x, p=self.p, training=self.training)
        x = x.transpose(1, 2)  # b*512*49
        x = self.conv(x)  # b*450*49
        x = x.transpose(1, 2)  # b*49*450

        if self.af:
            if self.af == 'softmax':
                x = getattr(F, self.af)(x, dim=self.dim)
            else:
                x = getattr(F, self.af)(x)
        return x

    def __call__(self, *input, **kwargs):
        return super().__call__(*input, **kwargs)


class MyLinear(nn.Module):
    def __init__(self, in_features, out_features, seed=None, p=None, af=None, dim=None):
        super(MyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.p = p
        self.af = af
        self.dim = dim
        if seed:
            torch.manual_seed(seed)
        self.linear = nn.Linear(in_features, out_features, bias=True)

    def forward(self, x):
        if x.size()[-1] != self.in_features:
            raise ValueError(
                '[error] putils.Linear(%s, %s): last dimension of input(%s) should equal to in_features(%s)' %
                (self.in_features, self.out_features, x.size(-1), self.in_features))
        if self.p:
            x = F.dropout(x, p=self.p, training=self.training)
        x = self.linear(x)
        if self.af:
            if self.af == 'softmax':
                x = getattr(F, self.af)(x, dim=self.dim)
            else:
                x = getattr(F, self.af)(x)
        return x

    def __call__(self, *input, **kwargs):
        return super().__call__(*input, **kwargs)