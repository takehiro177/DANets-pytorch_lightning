import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional  as F 
import torch.nn.init as init

from utils.config_loader import load_config
CFG = load_config('../config/experiment/danets_baseline.yaml')

# Mish activation function for fast convergence and better results
class Mish_func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.tanh(F.softplus(i))
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
  
        v = 1. + i.exp()
        h = v.log() 
        grad_gh = 1./h.cosh().pow_(2) 

        # Note that grad_hv * grad_vx = sigmoid(x)
        #grad_hv = 1./v  
        #grad_vx = i.exp()
        
        grad_hx = i.sigmoid()

        grad_gx = grad_gh *  grad_hx #grad_hv * grad_vx 
        
        grad_f =  torch.tanh(F.softplus(i)) + i * grad_gx 
        
        return grad_output * grad_f 

# Mish activation function wrapper
class Mish(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        pass
    def forward(self, input_tensor):
        return Mish_func.apply(input_tensor)

# kaiiming initialization for better convergence for conv1d and relu in DANETs
def kaiming_init(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d):
        init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            init.constant_(m.bias, 0)

# DANETs implementation starts here
# cited from original implementation: https://github.com/WhatAShot/DANet
#@inproceedings{danets, 
#   title={DANets: Deep Abstract Networks for Tabular Data Classification and Regression}, 
#   author={Chen, Jintai and Liao, Kuanlun and Wan, Yao and Chen, Danny Z and Wu, Jian}, 
#   booktitle={AAAI}, 
#   year={2022}
# }
import model.sparsemax as sparsemax
def initialize_glu(module, input_dim, output_dim):
    gain_value = np.sqrt((input_dim + output_dim) / np.sqrt(input_dim))
    torch.nn.init.xavier_normal_(module.weight, gain=gain_value)
    return

class GBN(torch.nn.Module):
    """
    Ghost Batch Normalization
    https://arxiv.org/abs/1705.08741
    """
    def __init__(self, input_dim, virtual_batch_size=512):
        super(GBN, self).__init__()
        self.input_dim = input_dim
        self.virtual_batch_size = virtual_batch_size
        self.bn = nn.BatchNorm1d(self.input_dim)

    def forward(self, x):
        if self.training == True:
            chunks = x.chunk(int(np.ceil(x.shape[0] / self.virtual_batch_size)), 0)
            res = [self.bn(x_) for x_ in chunks]
            return torch.cat(res, dim=0)
        else:
            return self.bn(x)

class LearnableLocality(nn.Module):

    def __init__(self, input_dim, k):
        super(LearnableLocality, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.rand(k, input_dim)))
        self.smax = sparsemax.Entmax15(dim=-1)

    def forward(self, x):
        mask = self.smax(self.weight)
        masked_x = torch.einsum('nd,bd->bnd', mask, x)  # [B, k, D]
        return masked_x

class AbstractLayer(nn.Module):
    def __init__(self, base_input_dim, base_output_dim, k, virtual_batch_size, bias=True):
        super(AbstractLayer, self).__init__()
        self.masker = LearnableLocality(input_dim=base_input_dim, k=k)
        self.fc = nn.Conv1d(base_input_dim * k, 2 * k * base_output_dim, kernel_size=1, groups=k, bias=bias)
        initialize_glu(self.fc, input_dim=base_input_dim * k, output_dim=2 * k * base_output_dim)
        self.bn = GBN(2 * base_output_dim * k, virtual_batch_size)
        self.k = k
        self.base_output_dim = base_output_dim

    def forward(self, x):
        b = x.size(0)
        x = self.masker(x)  # [B, D] -> [B, k, D]
        x = self.fc(x.view(b, -1, 1))  # [B, k, D] -> [B, k * D, 1] -> [B, k * (2 * D'), 1]
        x = self.bn(x)
        chunks = x.chunk(self.k, 1)  # k * [B, 2 * D', 1]
        x = sum([F.relu(torch.sigmoid(x_[:, :self.base_output_dim, :]) * x_[:, self.base_output_dim:, :]) for x_ in chunks])  # k * [B, D', 1] -> [B, D', 1]
        return x.squeeze(-1)

class BasicBlock(nn.Module):
    def __init__(self, input_dim, base_outdim, k, virtual_batch_size, fix_input_dim, drop_rate):
        super(BasicBlock, self).__init__()
        self.conv1 = AbstractLayer(input_dim, base_outdim // 2, k, virtual_batch_size)
        self.conv2 = AbstractLayer(base_outdim // 2, base_outdim, k, virtual_batch_size)

        self.downsample = nn.Sequential(
            nn.Dropout(drop_rate),
            AbstractLayer(fix_input_dim, base_outdim, k, virtual_batch_size)
        )

    def forward(self, x, pre_out=None):
        if pre_out == None:
            pre_out = x
        out = self.conv1(pre_out)
        out = self.conv2(out)
        identity = self.downsample(x)
        out += identity
        return F.leaky_relu(out, 0.01)

#Batch size for Ghost Batch Normalization (virtual_batch_size < batch_size)
class DANet(nn.Module):
    def __init__(self, input_dim, num_classes=CFG['num_classes'], layer_num=32, base_outdim=64, k=5, virtual_batch_size=256, drop_rate=0.1):
        super(DANet, self).__init__()
        params = {'base_outdim': base_outdim, 'k': k, 'virtual_batch_size': virtual_batch_size,
                  'fix_input_dim': input_dim, 'drop_rate': drop_rate}
        self.init_layer = BasicBlock(input_dim, **params)
        self.lay_num = layer_num
        self.layer = nn.ModuleList()
        for i in range((layer_num // 2) - 1):
            self.layer.append(BasicBlock(base_outdim, **params))
        self.drop = nn.Dropout(0.1)

        self.fc = nn.Sequential(nn.Linear(base_outdim, 256),
                                nn.ReLU(inplace=True),
                                nn.Linear(256, 512),
                                nn.ReLU(inplace=True),
                                nn.Linear(512, num_classes))

    def forward(self, x):
        out = self.init_layer(x)
        for i in range(len(self.layer)):
            out = self.layer[i](x, out)
        out = self.drop(out)
        out = self.fc(out)
        return out

class CustomHead(nn.Module):
    def __init__(self, num_features, num_classes):
        super(CustomHead, self).__init__()
        # Add your custom layers here
        self.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        # Assuming x is the output features from the base model
        x = torch.flatten(x, 1)  # Flatten the features
        x = self.fc(x)  # Pass through the custom linear layer
        return x

class ISIC2024KANNet(torch.nn.Module):
    def __init__(self, meta_feats):
        super().__init__()

        self.n_meta_features = len(meta_feats)
        self.base_model = DANet(input_dim=self.n_meta_features)
        
        self.apply(kaiming_init)

    def forward(self, x):
        x = self.base_model(x)
        to_logit = x

        return to_logit
    