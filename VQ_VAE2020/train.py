from model import Encoder, Decoder, VQEmbedding, Jitter
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.nn import Module, Conv1d, BatchNorm1d, ReLU, Sequential
from tqdm import tqdm
import numpy as np
from pre_process import mulaw_decode
import hparam
import os


root_path = hparam.root_path
mels = []
wavs = []
with os.scandir(r'C:\Users\Dell\Desktop\New_code' ) as entries:
    count = 1
    for file in entries:
        if file.is_file():
            if(count > 16):
                break
            
            str1=r'C:\Users\Dell\Desktop\New_code\input_data'  + rf'{count}' + r'mel.npy'
            print(str1)
            str2=r'C:\Users\Dell\Desktop\New_code\input_data'  + rf'{count}' + r'wav.npy'
            mels.append(
                np.load(str1))
            wavs.append(
                np.load(str2))
            count += 1

mels_dummy = np.array(mels)
print(mels_dummy.shape)

mels = torch.tensor(mels,dtype=torch.float32)
print(mels.dtype)


if __name__ == "__main__":
    encoder = Encoder(hparam.in_channels, hparam.channels,
                      hparam.n_embeddings, hparam.embedding_dim, hparam.jitter)
    mels=mels[:,:,0:34]
    z = encoder(mels)
    vq = VQEmbedding(hparam.n_embeddings, hparam.embedding_dim)
    vq(z)
    j=Jitter(0.5)
    z=j(z.transpose(1,2))
    z=z.transpose(1,2)
    speakers=torch.zeros(16,dtype=torch.long)
    decoder=Decoder(hparam.in_channels,hparam.n_speakers,hparam.speaker_embedding_dim,hparam.conditioning_channels,
                    hparam.mu_embedding_dim,hparam.rnn_channels,hparam.fc_channels,hparam.bits,hparam.hop_length)
    x=torch.zeros((16,5120),dtype=torch.long)
    z=decoder(z,speakers,x)


