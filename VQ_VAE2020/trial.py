import numpy as np
import librosa.display
import matplotlib.pyplot as plt
import torch
import json
from torch.distributions import Categorical
from torch.nn import Module, Conv1d, BatchNorm1d, ReLU, Sequential
import random
import os
import csv
from csv import reader

# with open(r'C:\Users\Dell\Desktop\New_code\vq_vae_2020\ZeroSpeech\datasets\2019\english\speakers.json') as f:
#   data = json.load(f)
# # speakers = sorted(json.load(r))
# print(data.index(2))

# a=torch.Tensor([0.25,0.5,0.25])
# b=Categorical(probs=a)
# val=torch.randn(3)
# index=b.sample(sample_shape=torch.Size([4,4]))
# print(index)

# a=np.load(r'C:\Users\Dell\Desktop\New_code\input_data1wav.npy')
# print(a.shape)

# a=torch.zeros(16,256,5120).float()
# b=torch.zeros(16,5120).long()
# c=torch.nn.functional.cross_entropy(a,b)

# a= random.randint(1,10)
# print(a)
z=torch.zeros(3).fill_(4 // 2).long()
print(z)




