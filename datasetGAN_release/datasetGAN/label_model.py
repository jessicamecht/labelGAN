import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152

import sys
sys.path.append('..')
import torch
import torch.nn as nn
torch.manual_seed(0)
import json
from collections import OrderedDict
import numpy as np
import os
from models.stylegan1 import G_mapping,Truncation,G_synthesis
from utils.utils import  Interpolate

def prepare_stylegan(args, device):
    if args['stylegan_ver'] == "1":
        if args['category'] == "car":
            resolution = 512
            max_layer = 8
        elif  args['category'] == "face":
            resolution = 1024
            max_layer = 8
        elif args['category'] == "bedroom":
            resolution = 256
            max_layer = 7
        elif args['category'] == "cat":
            resolution = 256
            max_layer = 7
        elif args['category'] == "xray":
            resolution = 1024
            max_layer = 7
        else:
            assert "Not implementated!"

        avg_latent = np.load(args['average_latent'])
        avg_latent = torch.from_numpy(avg_latent).type(torch.FloatTensor).to(device)
        

        g_all = nn.Sequential(OrderedDict([
            ('g_mapping', G_mapping()),
            ('truncation', Truncation(avg_latent,max_layer=max_layer, device=device, threshold=0.7)),
            ('g_synthesis', G_synthesis( resolution=resolution))
        ]))
        
        ckpt = torch.load(args['stylegan_checkpoint'], map_location=device)
        
        for key in list(ckpt.keys()):
            new_key = key.replace('init_block', 'blocks.4x4').replace('blocks.0.', 'blocks.8x8.')
            new_key = new_key.replace('blocks.1.', 'blocks.16x16.').replace('blocks.5.', 'blocks.256x256.')
            new_key = new_key.replace('blocks.3.', 'blocks.64x64.').replace('blocks.2.', 'blocks.32x32.')
            new_key = new_key.replace('blocks.4.', 'blocks.128x128.').replace('blocks.6.', 'blocks.512x512.')
            new_key = new_key.replace("g_mapping.map.dense", "g_mapping.dense")
            new_key = new_key.replace("g_synthesis.to_rgb.", "g_synthesis.torgb.")
            ckpt[new_key] = ckpt.pop(key)

        g_all.load_state_dict(ckpt, strict=False)
        g_all.eval()
        g_all = nn.DataParallel(g_all).to(device)
        

    else:
        assert "Not implementated error"

    res  = args['dim'][1]
    mode = args['upsample_mode']
    upsamplers = [nn.Upsample(scale_factor=res / 4, mode=mode),
                  nn.Upsample(scale_factor=res / 4, mode=mode),
                  nn.Upsample(scale_factor=res / 8, mode=mode),
                  nn.Upsample(scale_factor=res / 8, mode=mode),
                  nn.Upsample(scale_factor=res / 16, mode=mode),
                  nn.Upsample(scale_factor=res / 16, mode=mode),
                  nn.Upsample(scale_factor=res / 32, mode=mode),
                  nn.Upsample(scale_factor=res / 32, mode=mode),
                  nn.Upsample(scale_factor=res / 64, mode=mode),
                  nn.Upsample(scale_factor=res / 64, mode=mode),
                  nn.Upsample(scale_factor=res / 128, mode=mode),
                  nn.Upsample(scale_factor=res / 128, mode=mode),
                  nn.Upsample(scale_factor=res / 256, mode=mode),
                  nn.Upsample(scale_factor=res / 256, mode=mode)
                  ]

    if resolution > 256:
        upsamplers.append(nn.Upsample(scale_factor=res / 512, mode=mode))
        upsamplers.append(nn.Upsample(scale_factor=res / 512, mode=mode))

    if resolution > 512:

        upsamplers.append(Interpolate(res, 'bilinear'))
        upsamplers.append(Interpolate(res, 'bilinear'))

    return g_all, avg_latent, upsamplers

class label_classifier(nn.Module):
    def __init__(self, label_class, label_dim):
        super(label_classifier, self).__init__()
        self.label_layers = nn.Sequential(
                nn.Linear(label_dim, 128),
                nn.ReLU(),
                nn.BatchNorm1d(num_features=128),
                nn.Linear(128, 32),
                nn.ReLU(),
                nn.BatchNorm1d(num_features=32),
                nn.Linear(32, label_class),
            )

    def init_weights(self, init_type='normal', gain=0.02):
        self.apply(lambda x: init_func(x, init_type, gain))

    def forward(self, x):
        return self.label_layers(x)
    

class segm_classifier(nn.Module):
    def __init__(self, numpy_class, dim):
        super(segm_classifier, self).__init__()
        self.layers = nn.Sequential(
                nn.Linear(dim, 128),
                nn.ReLU(),
                nn.BatchNorm1d(num_features=128),
                nn.Linear(128, 32),
                nn.ReLU(),
                nn.BatchNorm1d(num_features=32),
                nn.Linear(32, numpy_class),
            )

    def init_weights(self, init_type='normal', gain=0.02):
        self.apply(lambda x: init_func(x, init_type, gain))

    def forward(self, x_segm):
        return self.layers(x_segm)
    
def init_func(m, init_type, gain):
    classname = m.__class__.__name__
    if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
        if init_type == 'normal':
            nn.init.normal_(m.weight.data, 0.0, gain)
        elif init_type == 'xavier':
            nn.init.xavier_normal_(m.weight.data, gain=gain)
        elif init_type == 'kaiming':
            nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        elif init_type == 'orthogonal':
            nn.init.orthogonal_(m.weight.data, gain=gain)

        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)

    elif classname.find('BatchNorm2d') != -1:
        nn.init.normal_(m.weight.data, 1.0, gain)
        nn.init.constant_(m.bias.data, 0.0)