import torch
import numpy as np
a=[0,0,0,0,0.0]
a=torch.tensor(a)
softmax0=torch.nn.Softmax()

print(softmax0(a))
