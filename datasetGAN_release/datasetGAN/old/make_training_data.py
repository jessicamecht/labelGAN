"""
Copyright (C) 2021 NVIDIA Corporation.  All rights reserved.
Licensed under The MIT License (MIT)

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

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
device_ids = [0]
from PIL import Image
#from models.stylegan import GMapping, GSynthesis, Truncation, Generator

from models.stylegan1 import G_mapping,G_synthesis, Truncation
import copy
from numpy.random import choice
from utils.utils import Interpolate
import argparse

import models as n
#from models.CustomLayer import Truncation
device = 'cuda:3' if torch.cuda.is_available() else 'cpu'

def latent_to_image(g_all, upsamplers, latents, return_upsampled_layers=False, use_style_latents=False,
                    style_latents=None, process_out=True, return_stylegan_latent=False, dim=512, return_only_im=False):
    '''Given a input latent code, generate corresponding image and concatenated feature maps'''

    # assert (len(latents) == 1)  # for GPU memory constraints
    print(latents.shape)
    if not use_style_latents:
        # generate style_latents from latents
        style_latents = g_all.module.truncation(g_all.module.g_mapping(latents))
        style_latents = style_latents.clone()  # make different layers non-alias

    else:
        style_latents = latents

        # style_latents = latents
    if return_stylegan_latent:

        return  style_latents
    if len(style_latents.shape) == 2:
        style_latents = style_latents.unsqueeze(0)
    img_list, affine_layers = g_all.module.g_synthesis(style_latents)

    if return_only_im:
        if process_out:
            if img_list.shape[-2] > 512:
                img_list = upsamplers[-1](img_list)

            img_list = img_list.cpu().detach().numpy()
            img_list = process_image(img_list)
            img_list = np.transpose(img_list, (0, 2, 3, 1)).astype(np.uint8)
        return img_list, style_latents

    number_feautre = 0

    for item in affine_layers:
        number_feautre += item.shape[1]


    affine_layers_upsamples = torch.FloatTensor(1, number_feautre, dim, dim)
    if return_upsampled_layers:

        start_channel_index = 0
        for i in range(len(affine_layers)):
            len_channel = affine_layers[i].shape[1]
            affine_layers_upsamples[:, start_channel_index:start_channel_index + len_channel] = upsamplers[i](
                affine_layers[i]).cpu().detach()
            start_channel_index += len_channel

    if img_list.shape[-2] != 512:
        img_list = upsamplers[-1](img_list)

    if process_out:
        img_list = img_list.cpu().detach().numpy()
        img_list = process_image(img_list)
        img_list = np.transpose(img_list, (0, 2, 3, 1)).astype(np.uint8)
        # print('start_channel_index',start_channel_index)


    return img_list, affine_layers_upsamples


def process_image(images):
    drange = [-1, 1]
    scale = 255 / (drange[1] - drange[0])
    images = images * scale + (0.5 - drange[0] * scale)

    images = images.astype(int)
    images[images > 255] = 255
    images[images < 0] = 0

    return images.astype(int)



def prepare_stylegan(args):

    if args['stylegan_ver'] == "1":
        if args['category'] == "xray":
            resolution = 1024
            max_layer = 8
        else:
            assert "Not implementated!"

        if args['average_latent'] != "":
            avg_latent = np.load(args['average_latent'])
            avg_latent = torch.from_numpy(avg_latent).type(torch.FloatTensor).to(device)
        else:
            avg_latent = None
        '''g_all = nn.Sequential(OrderedDict([
            ('g_mapping', GMapping()),
            ('truncation', Truncation(avg_latent,max_layer=max_layer, threshold=0.7)),
            ('g_synthesis', GSynthesis(resolution=256))
        ]))'''
        g_all = nn.Sequential(OrderedDict([('g_mapping', G_mapping()),
        ('truncation', Truncation(avg_latent, device)),
        ('g_synthesis', G_synthesis(resolution=1024))    
            ]))
        
        #print('hhh', GSynthesis())
        #print('kkk', G_synthesis())
        
        ckpt = torch.load(args['stylegan_checkpoint'], map_location=device)
    
        for key in list(ckpt.keys()):
            new_key = key.replace('init_block', 'blocks.4x4').replace('blocks.0.', 'blocks.8x8.')
            new_key = new_key.replace('blocks.1.', 'blocks.16x16.').replace('blocks.5.', 'blocks.256x256.')
            new_key = new_key.replace('blocks.3.', 'blocks.64x64.').replace('blocks.2.', 'blocks.32x32.')
            new_key = new_key.replace('blocks.4.', 'blocks.128x128.').replace('blocks.6.', 'blocks.512x512.')

            new_key = new_key.replace('blocks.7.', 'blocks.1024x1024.')
            
            new_key = new_key.replace("g_mapping.map.dense", "g_mapping.dense")
            new_key = new_key.replace("g_synthesis.to_rgb.8.", "g_synthesis.torgb.")
            
            ckpt[new_key] = ckpt.pop(key)

        g_all.load_state_dict(ckpt, strict=False)
        g_all.eval()
        g_all = nn.DataParallel(g_all, device_ids=device_ids).to(device)



        if args['average_latent'] == '':
            avg_latent = g_all.module.g_mapping.make_mean_latent(8000)
            g_all.module.truncation.avg_latent = avg_latent



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


def generate_data(args, num_sample, sv_path):
    # use face_palette because it has most classes
    from utils.data_util import face_palette as palette



    if os.path.exists(sv_path):
        pass
    else:
        os.system('mkdir -p %s' % (sv_path))
        print('Experiment folder created at: %s' % (sv_path))


    g_all, avg_latent, upsamplers = prepare_stylegan(args)

    # dump avg_latent for reproducibility
    mean_latent_sv_path = os.path.join(sv_path, "avg_latent_stylegan1.npy")
    np.save(mean_latent_sv_path, avg_latent[0].detach().cpu().numpy())


    with torch.no_grad():
        latent_cache = []

        results = []
        np.random.seed(1111)


        print( "num_sample: ", num_sample)
        latent_path = '/data3/jessica/data/labelGAN/train_images/latents/'#args['annotation_image_latent_path_classification']
        files = os.listdir(latent_path)
        for i in range(num_sample):
            if i % 10 == 0:
                print("Generate", i, "Out of:", num_sample)
            '''if i == 0:

                latent = avg_latent.to(device)
                img, _ = latent_to_image(g_all, upsamplers, latent, dim=args['dim'][1],
                                         return_upsampled_layers=False, use_style_latents=True)
            else:
                latent = np.random.randn(1, 512)
                latent_cache.append(copy.deepcopy(latent))


                latent = torch.from_numpy(latent).type(torch.FloatTensor).to(device)

                img, _ = latent_to_image(g_all, upsamplers, latent, dim=args['dim'][1],
                                                         return_upsampled_layers=False)
            #latent = np.random.randn(1, 512)'''
            
            p = latent_path + files[i]
            style_latent = np.load(p) 
            latent_cache.append(copy.deepcopy(style_latent))

            style_latent = torch.from_numpy(style_latent).type(torch.FloatTensor).to(device)

            img, _ = latent_to_image(g_all, upsamplers, style_latent, dim=args['dim'][1],
                                                         return_upsampled_layers=False, 
                                                         use_style_latents=True)

            #if args['dim'][0] != args['dim'][1]:
            #    img = img[:, 64:448][0]
            #else:
            img = img[0]
            img = Image.fromarray(img)

            image_name =  os.path.join(sv_path, "image_%d.jpg" % i)
            img.save(image_name)

        latent_cache = np.concatenate(latent_cache, 0)
        latent_sv_path = os.path.join(sv_path, "latent_stylegan1.npy")
        np.save(latent_sv_path, latent_cache)

        '''for style_latent in latent_cache:

            style_latent = torch.from_numpy(style_latent).type(torch.FloatTensor).to(device).unsqueeze(0)

            img, _ = latent_to_image(g_all, upsamplers, style_latent, dim=args['dim'][1],
                                                         return_upsampled_layers=False, 
                                                         use_style_latents=True,
                                                        style_latents=style_latent)

            #if args['dim'][0] != args['dim'][1]:
            #    img = img[:, 64:448][0]
            #else:
            img = img[0]
            img = Image.fromarray(img)

            image_name =  os.path.join(sv_path, 'reconstruct', "image_%d.jpg" % i)
            img.save(image_name)'''

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--exp', type=str)
    parser.add_argument('--num_sample', type=int,  default=100)
    parser.add_argument('--sv_path', type=str)

    args = parser.parse_args()

    opts = json.load(open(args.exp, 'r'))
    print("Opt", opts)


    path =opts['exp_dir']
    if os.path.exists(path):
        pass
    else:
        os.system('mkdir -p %s' % (path))
        print('Experiment folder created at: %s' % (path))

    os.system('cp %s %s' % (args.exp, opts['exp_dir']))



    generate_data(opts, args.num_sample, args.sv_path)
