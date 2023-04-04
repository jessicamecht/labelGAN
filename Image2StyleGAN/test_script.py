import os 
import sys
#sys.path.append('/home/jessica/labelGAN/StyleGAN.pytorch/')
#from models.GAN import * 
#from stylegan_layers import  G_mapping,G_synthesis
import torch
import matplotlib.patches as patches
from collections import OrderedDict
import torch.nn.functional as F
from sklearn.neighbors import KernelDensity
from torchvision.utils import save_image
from tqdm import tqdm
from PIL import Image, ImageOps
from torchvision import transforms
import torchvision
from torchvision import models
import cv2
import torch.nn as nn
from torchvision.utils import save_image
import pydicom as dicom
import numpy as np
from math import log10
import pandas as pd
import matplotlib.pyplot as plt
from model import * 
from utils import * 
import imageio
sys.path.append("/home/jessica/labelGAN/datasetGAN_release/models")
from stylegan1 import G_mapping,Truncation,G_synthesis


p = '/home/jessica/labelGAN/Image2StyleGAN/images/generated_latents_from_class_distr/'
files = os.listdir(p)
mask = ["tophat" in elem for elem in files]
files = np.array(files)[mask]

latent = np.load('/home/jessica/labelGAN/Image2StyleGAN/latent.npy')
device = 'cuda:3' if torch.cuda.is_available() else 'cpu'
g_all = nn.Sequential(OrderedDict([('g_mapping', G_mapping()),
    #('truncation', Truncation(avg_latent)),
    ('g_synthesis', G_synthesis(resolution=1024))    
    ])).to(device)

#opts = {'mapping_layers': 8, 'truncation_psi': -1.}
'''g_all = Generator(resolution=1024,
                    num_channels=3,
                    structure='linear',
                    **opts)'''
'''g_all = nn.Sequential(OrderedDict([('g_mapping', GMapping()),
    #('truncation', Truncation(avg_latent)),
    ('g_synthesis', GSynthesis(resolution=1024, depth=8))    
    ]))'''
#Load the pre-trained model
ckpt = torch.load('/data1/jessica/data/labelGAN/checkpoints/styleGAN/GAN_GEN_8.pth', map_location=device)
print(ckpt.keys())
for key in list(ckpt.keys()):
            new_key = key.replace('init_block', 'blocks.4x4').replace('blocks.0.', 'blocks.8x8.')
            new_key = new_key.replace('blocks.1.', 'blocks.16x16.').replace('blocks.5.', 'blocks.256x256.')
            new_key = new_key.replace('blocks.3.', 'blocks.64x64.').replace('blocks.2.', 'blocks.32x32.')
            new_key = new_key.replace('blocks.4.', 'blocks.128x128.').replace('blocks.6.', 'blocks.512x512.')
            new_key = new_key.replace("g_mapping.map.dense", "g_mapping.dense")
            new_key = new_key.replace("g_synthesis.to_rgb.", "g_synthesis.torgb.")
            ckpt[new_key] = ckpt.pop(key)
print(ckpt.keys())

g_all.load_state_dict(ckpt, strict=False)
g_all.eval()
g_all.to(device)
g_mapping, g_synthesis = g_all[0], g_all[1]
perceptual = VGG16_perceptual().to(device)
print(torch.round(torch.tensor(latent), decimals=4))

img, _ = g_all[1](torch.round(torch.tensor(latent), decimals=4).to(device))
result_path = '/data3/jessica/data/labelGAN/results_dir_multitask_generation_segm/vis_2000'
save_image(img.detach(),os.path.join(result_path, "vis_" + str(0) + '_image.jpg'))
plt.imshow(img[0].permute(1,2,0).cpu().detach().numpy())
plt.savefig('/home/jessica/labelGAN/Image2StyleGAN/img.png')