import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch import optim
import numpy as np
from copy import deepcopy
import os
from math import log2, floor

os.chdir(os.path.dirname(os.path.realpath(__file__)))
from models import ResBlock, MeanShift, PoolBlock, ConvBlock

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class FineTuner(nn.Module):
    def __init__(self, network, fine_tuning_depth):
        super(FineTuner, self).__init__()
        self.config = network.getconfig()
        self.network = network
        self.depth = fine_tuning_depth
        target_config = self.config[-fine_tuning_depth:]
        self.param_nb = 0
        for conf in self.config:
            if conf[0] == 'conv2d' or conf[0] == 'convt2d' or conf[0] == 'linear':
                self.param_nb += 2
            elif conf[0] == 'prelu':
                self.param_nb += 1
            elif conf[0] == 'bn':
                raise NotImplementedError
            elif conf[0] == 'resblock_leakyrelu' or conf[0] == 'resblock_relu':
                self.param_nb += 4
            else:
                pass

        self.parameters_size = 0
        for type, conf in target_config:
            if type == 'conv2d':
                self.parameters_size += conf[0]*conf[1]*conf[2]*conf[3] + conf[0]
            elif type == 'convt2d':
                self.parameters_size += conf[0]*conf[1]*conf[2]*conf[3] + conf[1]
            elif type == 'linear':
                self.parameters_size += conf[0]*conf[1]
            elif type == 'bn': # No Idea wether a meta learner could be able to figure out how to give the right parameters for a batch normalization
                raise NotImplementedError
            elif type == 'prelu':
                raise NotImplementedError # ToDo
            elif type == 'reblock_leakyrelu' or type == 'resblock_relu':
                self.parameters_size += (conf[0]*conf[1]*conf[2]*conf[3] + conf[0])*2
            else:
                pass
        print("Setuping meta fine-tuner network for " + str(self.parameters_size) + " parameters.", flush=True)
        self.block_number = int(floor(log2(self.parameters_size) - 3)) # -3 because we start with RGB(3) to 8 channels
        print("Making an architecture of depth " + str(self.block_number + 1) + ".", flush=True)

        m = [nn.Conv2d(3, 8, kernel_size=3, padding=0)]
        for i in range(self.block_number):
            if i%2 == 1:
                m.append(PoolBlock(2**(3+i), 2**(3+i+1), act=nn.LeakyReLU(inplace=True)))
            else:
                m.append(ConvBlock(2**(3+i), 2**(3+i+1), act=nn.LeakyReLU(inplace=True)))
        m.append(nn.AdaptiveAvgPool2d(output_size=(1, 1)))
        self.body = nn.Sequential(*m).to(device)
        self.tail = nn.Sequential(nn.Linear(self.block_number, self.parameters_size)).to(device)

    def forward(self, x_spt, x_qry):
        x_spt = self.body(x_spt)
        x_spt = x_spt.view(-1, 2**(3 + self.block_number))
        x_spt = self.tail(x_spt) # x_spt is now a vector representing the parameters that needs to be loaded into the SR network.
        x_spt = x_spt.view(-1)
        self.finetune(x_spt)
        x_qry = self.network(x_qry)
        return x_qry

    def finetune(self, x):
        assert x.size()[0] == self.parameters_size
        k = 0
        for i, param in enumerate(self.network.parameters()):
            if self.param_nb - i < self.depth:
                pass
            else:
                n = param.data.view(-1).size()[0]
                param.data = x.data[k:k+n]
                k = n

