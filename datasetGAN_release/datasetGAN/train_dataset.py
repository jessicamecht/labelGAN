import torch 
import torch.nn as nn 
import numpy as np 
from utils.utils import colorize_mask, latent_to_image
import os
import sys
sys.path.append('..')
from tqdm import tqdm
import pickle
import imageio
torch.manual_seed(0)
from PIL import Image
import gc
from torch.utils.data import Dataset
from train_dataset import *
from label_model import *
import cv2

few_shot_classes = {"Nofinding": 0,
    'NoduleMass': 9,
    'Infiltration': 7,
    'LungOpacity': 3,
    'Consolidation': 4,
    'Pleuralthickening': 5,
    'ILD': 6,
    'Cardiomegaly': 2,
    'Pulmonaryfibrosis': 8,
    'Aorticenlargement': 1,
    'Otherlesion': 10,
    'Pleuraleffusion': 11,
    'Calcification': 12,
    'Atelectasis': 13,
    'Pneumothorax': 14}


class trainData(Dataset):

    def __init__(self, X_data, y_data, args):
        self.X_data = X_data
        self.y_data = y_data
        self.args = args

    def __getitem__(self, index):
        x = torch.tensor(self.X_data[index])
        x = x.reshape(-1, self.args['dim'][2])
        return x.type(torch.FloatTensor).squeeze(), torch.tensor(self.y_data[index]).type(torch.FloatTensor)

    def __len__(self):
        return len(self.X_data)

class labelDataLatent(Dataset):
    def __init__(self, files, path, device):
        self.files = files
        self.path = path
        self.device = device

    def __getitem__(self, index):
        imagename = self.files[index]
        x = torch.tensor(np.load(self.path + imagename)).to(self.device)
        #im_name = os.path.join(self.args['annotation_image_path_classification'], x)
        #x = torch.tensor(np.load(im_name)).type(torch.FloatTensor)
        y = torch.tensor([few_shot_classes[imagename.split("_")[0]]])
        return x,y

    def __len__(self):
        return len(self.files)
    
class labelData(Dataset):
    def __init__(self, files, path, reshaper, device):
        self.files = files
        self.path = path
        self.reshaper = reshaper
        self.device = device

    def __getitem__(self, index):
        imagename = self.files[index]
        image_id = imagename.split("_")[1]
        feat_size = imagename.split("_")[-1].replace(".npy", "")
        x = torch.tensor(np.load(self.path + imagename)).to(self.device)
        x = self.reshaper(x)
        #im_name = os.path.join(self.args['annotation_image_path_classification'], x)
        #x = torch.tensor(np.load(im_name)).type(torch.FloatTensor)
        y = torch.tensor([few_shot_classes[imagename.split("_")[0]]])
        return x,y, image_id, feat_size

    def __len__(self):
        return len(self.files)
    

def prepare_data(args, palette, device):

    g_all, avg_latent, upsamplers = prepare_stylegan(args, device)
    latent_all = np.load(args['annotation_image_latent_path'])
    
    latent_all = torch.from_numpy(latent_all)
    

    # load annotated mask
    mask_list = []
    im_list = []
    latent_all = latent_all[:args['max_training']]
    
    num_data = len(latent_all)

    annotation_mask_path_files = os.listdir(args['annotation_mask_path'])
    for i in tqdm(range(len(latent_all))):
        if i == 3: continue
        if i >= args['max_training']:
            break
        name = 'image_%0d.png' % i
        mask_name = annotation_mask_path_files[i]
        name = mask_name.replace("_mask.png", ".png")
        
        im_frame = Image.open(os.path.join( args['annotation_mask_path'] , mask_name)).convert('L')
        mask = np.array(im_frame)
        mask = mask.squeeze()
        mask =  cv2.resize(np.float32(mask), dsize=(args['dim'][1], args['dim'][0]), interpolation=cv2.INTER_NEAREST)
        mask_list.append(mask)
        im_name = os.path.join( args['annotation_image_path'], name)
        img = Image.open(im_name)
        img = img.resize((args['dim'][1], args['dim'][0]))

        im_list.append(np.array(img))

    # delete small annotation error
    '''for i in range(len(mask_list)):  # clean up artifacts in the annotation, must do
        if mask_list[i] == None: continue 
        for target in range(1, 50):
            if (mask_list[i] == target).sum() < 30:
                mask_list[i][mask_list[i] == target] = 0'''

    all_mask = np.stack(mask_list)
    all_feature_maps_train_list = []
    vis = []
    latent_path = args['annotation_image_latent_path_classification']
    latent_files = np.array(os.listdir(latent_path))
    all_mask_train_list = []
    for i in tqdm(range(len(latent_all))):
        gc.collect()
        mask_name = annotation_mask_path_files[i]
        name = mask_name.split("_")[1]
        mask = [name in elem for elem in latent_files]
        name = latent_files[mask][0]
        
        latent_input = torch.tensor(np.load(latent_path + name)).to(device)

        #latent_input = latent_all[i].float().unsqueeze(0)

        img, feature_maps, style_latents, affine_layers = latent_to_image(g_all, upsamplers, latent_input, dim=args['dim'][1],
                                            return_upsampled_layers=True, use_style_latents=args['annotation_data_from_w'], device=device)

        #print('test', feature_maps.shape)
        mask = all_mask[i:i + 1]
        feature_maps = feature_maps.permute(0, 2, 3, 1)
        feature_maps = feature_maps.reshape(-1, args['dim'][2])
        new_mask =  np.squeeze(mask)
        mask = mask.reshape(-1)
        #all_feature_maps_train[start:end] = feature_maps.cpu().detach().numpy().astype(np.float16)
        if len(mask) == 0: continue
        all_feature_maps_train_list.append(feature_maps.cpu().detach())
        
        #all_mask_train[start:end] = (mask == 143).astype(np.float16)
        all_mask_train_list.append(torch.tensor(mask.astype(np.float16)))
        img_show =  cv2.resize(np.squeeze(img[0]), dsize=(args['dim'][1], args['dim'][1]), interpolation=cv2.INTER_NEAREST)
        curr_vis = np.concatenate( [im_list[i], img_show, colorize_mask(new_mask, palette)], 0)
        vis.append( curr_vis )
    all_feature_maps_train = torch.concat(all_feature_maps_train_list, axis=0)
    vis = np.concatenate(vis, 1)
    all_mask_train_list = torch.concat(all_mask_train_list, axis=0)
    imageio.imwrite(os.path.join(args['exp_dir'], "train_data.jpg"),
                      vis)
    return all_feature_maps_train, all_mask_train_list, num_data

