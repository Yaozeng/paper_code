import torch
from op import *
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import pickle as cPickl
a=torch.from_numpy(np.array([4]))
logits=np.array([0.1,0.2,0.3,0.1,0.5])
target = torch.zeros(5)
print(target)
target.scatter_(0, a, 1)
print(target)
loss = nn.functional.binary_cross_entropy_with_logits(logits, target)
print(loss)
