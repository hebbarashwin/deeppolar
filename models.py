import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time


def get_activation_fn(activation):
    if activation == 'tanh':
        return  F.tanh
    elif activation == 'elu':
        return F.elu
    elif activation == 'relu':
        return F.relu
    elif activation == 'selu':
        return F.selu
    elif activation == 'sigmoid':
        return F.sigmoid
    elif activation == 'gelu':
        return F.gelu
    elif activation == 'silu':
        return F.silu
    elif activation == 'mish':
        return F.mish
    elif activation == 'linear':
        return nn.Identity()
    else:
        raise NotImplementedError(f'Activation function {activation} not implemented')

class g_Full(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, depth=3, skip_depth = 1, skip_layer = 1, ell = 2, activation = 'selu', use_skip = False, augment = False):
        super(g_Full, self).__init__()

        self.input_size  = input_size


        self.hidden_size = hidden_size
        self.output_size = output_size

        self.depth = depth
        self.ell = ell
        self.ell_input_size = input_size//self.ell

        self.augment = augment
        self.activation_fn = get_activation_fn(activation)
        self.skip_depth = skip_depth
        self.skip_layer = skip_layer

        self.use_skip = use_skip 
        if self.use_skip:
            self.skip = nn.ModuleList([nn.Linear(self.input_size + self.output_size, self.hidden_size, bias=True)])
            self.skip.extend([nn.Linear(self.hidden_size, self.hidden_size, bias=True) for ii in range(1, self.skip_depth)])

        self.linears = nn.ModuleList([nn.Linear(self.input_size, self.hidden_size, bias=True)])
        self.linears.extend([nn.Linear(self.hidden_size, self.hidden_size, bias=True) for ii in range(1, self.depth)])
        self.linears.append(nn.Linear(self.hidden_size, self.output_size, bias=True))


    @staticmethod
    def get_augment(msg, ell):
        u = msg.clone()
        n = int(np.log2(ell))
        for d in range(0, n):
            num_bits = 2**d
            for i in np.arange(0, ell, 2*num_bits):
                # [u v] encoded to [u xor(u,v)]
                if len(u.shape) == 2:
                    u = torch.cat((u[:, :i], u[:, i:i+num_bits].clone() * u[:, i+num_bits: i+2*num_bits], u[:, i+num_bits:]), dim=1)
                elif len(u.shape) == 3:
                    u = torch.cat((u[:, :, :i], u[:, :, i:i+num_bits].clone() * u[:, :, i+num_bits: i+2*num_bits], u[:, :, i+num_bits:]), dim=2)

                # u[:, i:i+num_bits] = u[:, i:i+num_bits].clone() * u[:, i+num_bits: i+2*num_bits].clone
        if len(u.shape) == 3:
            return u[:, :, :-1]
        elif len(u.shape) == 2:
            return u[:, :-1]

    def forward(self, y):
                
        x = y.clone()
        for ii, layer in enumerate(self.linears):
            if ii != self.depth:
                x = self.activation_fn(layer(x))
                if self.use_skip and  ii == self.skip_layer:
                    if len(x.shape) == 3:
                        skip_input = torch.cat([y, g_Full.get_augment(y, self.ell)], dim = 2)
                    elif len(x.shape) == 2:
                        skip_input = torch.cat([y, g_Full.get_augment(y, self.ell)], dim = 1)
                    for jj, skip_layer in enumerate(self.skip):
                        skip_input = self.activation_fn(skip_layer(skip_input))
                    x = x + skip_input
            else:
                x = layer(x)
                if self.augment:
                    x = x + g_Full.get_augment(y, self.ell)
        return x

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.01)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(0.0, 0.01)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.01)
        try:
            m.bias.data.fill_(0.)
        except:
            pass

class f_Full(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_p = 0., activation = 'selu', depth=3, use_norm = False):
        super(f_Full, self).__init__()
        self.input_size  = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.depth = depth
        self.use_norm = use_norm

        self.activation_fn = get_activation_fn(activation)

        self.linears = nn.ModuleList([nn.Linear(self.input_size, self.hidden_size, bias=True)])
        if self.use_norm:
            self.norms = nn.ModuleList([nn.LayerNorm(self.hidden_size)])
        for ii in range(1, self.depth):
            self.linears.append(nn.Linear(self.hidden_size, self.hidden_size, bias=True))
            if self.use_norm:
                self.norms.append(nn.LayerNorm(self.hidden_size))
        self.linears.append(nn.Linear(self.hidden_size, self.output_size, bias=True))

    def forward(self, y, aug = None):

        x = y.clone()
        for ii, layer in enumerate(self.linears):
            if ii != self.depth:
                x = layer(x)
                if not hasattr(self, 'use_norm') or not self.use_norm:
                    pass
                else:
                    x = self.norms[ii](x)
                x = self.activation_fn(x)
            else:
                x = layer(x)
        return x

def get_onehot(actions):
    inds = (0.5 + 0.5*actions).long()
    return torch.eye(2, device = inds.device)[inds].reshape(actions.shape[0], -1)
