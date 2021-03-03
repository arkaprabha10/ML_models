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


def get_gru_cell(gru):
    gru_cell = nn.GRUCell(gru.input_size, gru.hidden_size)
    gru_cell.weight_hh.data = gru.weight_hh_l0.data
    gru_cell.weight_ih.data = gru.weight_ih_l0.data
    gru_cell.bias_hh.data = gru.bias_hh_l0.data
    gru_cell.bias_ih.data = gru.bias_ih_l0.data
    return gru_cell



class Encoder(nn.Module):
    def __init__(self, in_channels, channels, n_embeddings, embedding_dim, jitter=0):
        super(Encoder, self).__init__()

        self.enc1 = Sequential(
            Conv1d(in_channels=in_channels, out_channels=channels,
                   kernel_size=3, stride=1, padding=0),
            BatchNorm1d(channels),
            ReLU(True),
        )

        self.enc2 = Sequential(
            Conv1d(channels, channels, kernel_size=3, stride=1, padding=1),
            BatchNorm1d(channels),
            ReLU(True),
        )
        self.enc3 = Sequential(
            Conv1d(channels, channels, kernel_size=4, stride=2, padding=1),
            BatchNorm1d(channels),
            ReLU(True),
        )
        self.enc4 = Sequential(
            Conv1d(channels, channels, kernel_size=3, stride=1, padding=1),
            BatchNorm1d(channels),
            ReLU(True),
        )
        self.enc4 = Sequential(
            Conv1d(channels, channels, kernel_size=3, stride=1, padding=1),
            BatchNorm1d(channels),
            ReLU(True),
        )
        self.enc5 = Sequential(
            Conv1d(channels, channels, kernel_size=3, stride=1, padding=1),
            BatchNorm1d(channels),
            ReLU(True),
        )
        self.enc_out = nn.Conv1d(channels, embedding_dim, 1)

    def forward(self, mels):

        
        # conv1
        z = self.enc1(mels)
        print(z.shape)
        # conv2
        z = self.enc2(z)
        print(z.shape)
        # conv3
        z = self.enc3(z)
        print(z.shape)
        #conv 4
        z = self.enc4(z)
        print(z.shape)
        # # conv 5
        z = self.enc5(z)
        print(z.shape)

        # encoder output
        z = self.enc_out(z)
        print(z.shape)
        return z


class VQEmbedding(nn.Module):
    def __init__(self, n_embeddings, embedding_dim):
        super(VQEmbedding, self).__init__()

        range_embedding = 1/hparam.n_embeddings
        self.embedding = torch.Tensor(n_embeddings, embedding_dim)
        self.embedding.uniform_(-range_embedding, range_embedding)
        self.ema_count = torch.zeros(hparam.n_embeddings)
        self.ema_weight = self.embedding
        # self.register_buffer("embedding", self.embedding)
        # # ema_count is a 1*512 vector
        # self.register_buffer("ema_count", torch.zeros(n_embeddings))
        # # ema_weight is a 512*64 vector
        # self.register_buffer("ema_weight", self.embedding.clone())

    def quantize(self, x):
        M, D = self.embedding.size()
        print("M: ", M, "D: ", D)
        x_flat = x.detach().reshape(-1, D)
        print("x_flat:", x_flat.shape)
        distances = torch.addmm(torch.sum(self.embedding ** 2, dim=1) +
                                torch.sum(x_flat ** 2, dim=1, keepdim=True),
                                x_flat, self.embedding.t(),
                                alpha=-2.0, beta=1.0)
        print('distances')
        print(distances.shape)
        indices = torch.argmin(distances.float(), dim=-1)
        print("indices ", indices.shape)
        quantized = F.embedding(indices, self.embedding)
        
        print("value of quantised matrix is :",quantized)
        print("quantized-embedded", quantized.shape)
        quantized = quantized.view_as(x)
        print("quant_final", quantized.shape)
        return quantized, indices, x_flat

    def forward(self, x):
        M, D = self.embedding.size()
        quantized, indices, x_flat = self.quantize(x)
        encodings = F.one_hot(indices, M).float()
        print("encodings", encodings.shape)
        if self.training:
            # ema_count is a 1*512 vector
            self.ema_count = hparam.decay * self.ema_count + \
                (1 - hparam.decay) * torch.sum(encodings, dim=0)
            print("ema_count", self.ema_count.shape)
            n = torch.sum(self.ema_count)
            # ema_count is also a 1*512 vector
            self.ema_count = (self.ema_count + hparam.epsilon) / \
                (n + M * hparam.epsilon) * n
            #ema_weight is 512*64
            dw = torch.matmul(encodings.t(), x_flat)  # 512x64
            print("dw", dw.shape)
            self.ema_weight = hparam.decay * \
                self.ema_weight + (1 - hparam.decay) * dw

            self.embedding = self.ema_weight / self.ema_count.unsqueeze(-1)
            print('embedding: ', self.embedding.shape)

        e_latent_loss = F.mse_loss(x, quantized.detach())
        print('latent_loss: ', e_latent_loss)
        loss = hparam.commitment_cost * e_latent_loss
        print('loss: ', loss)
        # avg_probs = torch.mean(encodings, dim=0)
        # perplexity = torch.exp(-torch.sum(avg_probs *
        #                                   torch.log(avg_probs + 1e-10)))

        return quantized, loss, indices


class Jitter(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
        prob = torch.Tensor([p / 2, 1 - p, p / 2])
        self.register_buffer("prob", prob)

    def forward(self, x):
        if not self.training or self.p == 0:
            return x
        else:
            batch_size, sample_size, channels = x.size()
            #Categorical is multinomial distribution
            dist = Categorical(self.prob)
            #sample produces tensor based on input size and prob.
            index = dist.sample(torch.Size([batch_size, sample_size])) - 1
            index[:, 0].clamp_(0, 1)
            index[:, -1].clamp_(-1, 0)
            index += torch.arange(sample_size, device=x.device)

            x = torch.gather(
                x, 1, index.unsqueeze(-1).expand(-1, -1, channels))
        print('Jitter Output: ',x.shape)
        return x

class Decoder(nn.Module):
    def __init__(self, in_channels, n_speakers, speaker_embedding_dim,
                 conditioning_channels, mu_embedding_dim, rnn_channels,
                 fc_channels, bits, hop_length):
                 super().__init__()

                 self.rnn_channels = rnn_channels
                 self.quantization_channels = 2**bits
                 self.hop_length = hop_length
                 self.speaker_embedding = nn.Embedding(n_speakers, speaker_embedding_dim)
                #  self.rnn1 =  Sequential(nn.GRU(in_channels + speaker_embedding_dim, conditioning_channels,
                #            num_layers=2, batch_first=True, bidirectional=True))
                 self.rnn1 =  Sequential(nn.GRU(128, conditioning_channels,
                           num_layers=2, batch_first=True, bidirectional=True))
                 self.mu_embedding = nn.Embedding(self.quantization_channels, mu_embedding_dim)
                 self.rnn2 = Sequential(nn.GRU(mu_embedding_dim + 2*conditioning_channels, rnn_channels, batch_first=True))
                 self.fc1 = Sequential(nn.Linear(rnn_channels, fc_channels))
                 self.fc2 = Sequential(nn.Linear(fc_channels, self.quantization_channels))
    
    def forward(self,  z, speakers,x):
        print("Decoder Output size: ")
        print(z.shape)
        z = F.interpolate(z, scale_factor=2)
        print(z.shape)
        speakers = self.speaker_embedding(speakers)
        print(speakers.shape)
        z=z.transpose(1,2)
        print(z.shape)
        speakers = speakers.unsqueeze(1).expand(-1, z.size(1), -1)
        print(speakers.shape)
        z = torch.cat((z, speakers), dim=-1)
        print(z.shape)
        z, _ = self.rnn1(z)
        print(z.shape)
        z = F.interpolate(z.transpose(1, 2), scale_factor=self.hop_length)
        print("Error")
        print(z.shape)
        z = z.transpose(1, 2)
        print("check here")
        print(z.shape)
        x = self.mu_embedding(x)
        print(x.shape)
        x, _ = self.rnn2(torch.cat((x, z), dim=2))
        print(x.shape)
        x = F.relu(self.fc1(x))
        print(x.shape)
        x = self.fc2(x)
        print(x.shape)
        return x
    
    def generate(self,z,speaker):
        output = []
        cell = get_gru_cell(self.rnn2)
        z = F.interpolate(z, scale_factor=2)
        speakers = self.speaker_embedding(speakers)
        z=z.transpose(1,2)
        speakers = speakers.unsqueeze(1).expand(-1, z.size(1), -1)
        z = torch.cat((z, speakers), dim=-1)
        z, _ = self.rnn1(z)
        z = F.interpolate(z.transpose(1, 2), scale_factor=self.hop_length)
        z = z.transpose(1, 2)
        batch_size, sample_size, _ = z.size()
        








