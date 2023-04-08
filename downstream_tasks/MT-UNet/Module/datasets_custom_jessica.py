import torch
import numpy as np
import os
import matplotlib.pyplot as plt
import PIL
from PIL import Image,ImageOps
from torchvision import transforms
from torch.utils.data import Dataset
import pickle
import pandas as pd
import csv
from sklearn.preprocessing import MultiLabelBinarizer

mlb = {'Aorticenlargement': np.array([1,0,0,0,0,0,0,0,0,0,0,0,0,0,0]),
       'Atelectasis': np.array([0,1,0,0,0,0,0,0,0,0,0,0,0,0,0]), 
       'Calcification': np.array([0,0,1,0,0,0,0,0,0,0,0,0,0,0,0]),
       'Cardiomegaly': np.array([0,0,0,1,0,0,0,0,0,0,0,0,0,0,0]), 
       'Consolidation': np.array([0,0,0,0,1,0,0,0,0,0,0,0,0,0,0]), 
       'ILD': np.array([0,0,0,0,0,1,0,0,0,0,0,0,0,0,0]), 
       'Infiltration': np.array([0,0,0,0,0,0,1,0,0,0,0,0,0,0,0]),
       'LungOpacity': np.array([0,0,0,0,0,0,0,0,1,0,0,0,0,0,0]), 
       'Nofinding': np.array([0,0,0,0,0,0,0,0,0,1,0,0,0,0,0]), 
       'NoduleMass': np.array([0,0,0,0,0,0,0,0,0,1,0,0,0,0,0]), 
       'Otherlesion': np.array([0,0,0,0,0,0,0,0,0,0,1,0,0,0,0]),
       'Pleuraleffusion': np.array([0,0,0,0,0,0,0,0,0,0,0,1,0,0,0]), 
       'Pleuralthickening': np.array([0,0,0,0,0,0,0,0,0,0,0,0,1,0,0]), 
       'Pneumothorax': np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,1,0]),
       'Pulmonaryfibrosis': np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,1])}


class ChestXrayDataset(Dataset):

    def __init__(self, root_dir, image_and_labels, use_aug=False, resize_px=256):

        self.image_and_labels = image_and_labels
        self.root_dir = root_dir
        self.use_aug = use_aug
        self.resize_px = resize_px
        self.vinbig_path = '/data3/jessica/data/labelGAN/vinbig_test_imgs_and_segm'
        self.mask_path = os.listdir(f"{self.vinbig_path}/masks")
        
        train_csv = pd.read_csv('/data3/jessica/data/labelGAN/vinbig/train.csv')
        self.image_id_to_labels = train_csv.groupby(by="image_id").class_name.apply(list).apply(lambda x: np.unique([elem.replace(" ", "").replace("/", "") for elem in x]))

    def __len__(self):
        return len(os.listdir(f"{self.vinbig_path}/masks"))

    def __getitem__(self, idx):
 
        #Reading images
        imgname = self.mask_path[idx]
        imname = imgname.replace("_mask", "")
        img_name = os.path.join(self.vinbig_path, 'imgs', f"{imname}")
        image = Image.open(img_name)
        
        #Converting to grayscale if RGB
        image = ImageOps.grayscale(image)
        
        #Reading segmentation Masks
        mask_name = os.path.join(self.vinbig_path, 'masks', f"{imgname}")
        mask = Image.open(mask_name)
        mask = ImageOps.grayscale(mask)
        
        #Extracting disease label
        label_list = self.image_id_to_labels[imgname.replace(".png", "").replace("_mask", "")]
        labels = np.zeros(15).astype(int)
        for label in label_list:
            labels = labels | mlb[label]
        
        #Converting to tensors and Resizing images
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((self.resize_px, self.resize_px))])
        self.transform_hflip = transforms.functional.hflip
        
        image = self.transform(image)
        mask = self.transform(mask)
        
        data = {
                "cxr": image,
                "gaze": mask, # named gaze with compatibility with MT-UNet code
                "Y": torch.tensor(labels)
        }
        
        return data
    
class AugmentationDataset(Dataset):

    def __init__(self, root_dir, image_paths, mask_paths, labels=None, resize_px=256):
        
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.root_dir = root_dir
        self.labels = labels
        self.resize_px = resize_px
    
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        
 
        #Reading images
        img_name = os.path.join(self.root_dir, 'imgs', self.image_paths[idx])
        image = Image.open(img_name)
        
        #Converting to grayscale if RGB
        image = ImageOps.grayscale(image)
        
        #Reading segmentation Masks
        mask_name = os.path.join(self.root_dir, 'masks', self.mask_paths[idx])
        mask = Image.open(mask_name)
        mask = np.where(np.min(mask, axis=2) >= 150, 1, 0)
        
        #Extracting disease label
        if self.labels != None:
            label = self.labels[idx]
        else: 
            label = self.image_paths[idx].split("_")[1]
            label = mlb[label]
        
        #Converting to tensors and Resizing images
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((self.resize_px, self.resize_px))])
        self.transform_hflip = transforms.functional.hflip
        
        image = self.transform(image)
        mask = self.transform(mask)
        
        #Flipping images and labels with probability 0.5
        probability = torch.rand(1)
        if probability <= 0.5:
            image = self.transform_hflip(image)
            mask = self.transform_hflip(mask)

        data = {
                "cxr": image,
                "gaze": mask, # named gaze with compatibility with MT-UNet code
                "Y": torch.tensor(label)
        }
        
        return data

def get_data_splits(f):
    return [(image.split()[0], np.array(list(map(int, image.split()[1:])))) for image in f.readlines()]

def get_datasets(aug_size= None, use_augmentation = False, resize_px=256):
    root_dir = '/home/jessica/labelGAN/downstream_tasks/vinbig/'
    
    with open(os.path.join(root_dir, "train_binarized_list.txt")) as f:
        train_file = get_data_splits(f)
    train_dataset = ChestXrayDataset(root_dir,
                                     train_file, resize_px=resize_px)
    root_dir
    with open(os.path.join(root_dir, "train_binarized_list.txt")) as f:
        val_file = get_data_splits(f)
    valid_dataset = ChestXrayDataset(root_dir,
                                     val_file, resize_px=resize_px)   
    
    with open(os.path.join(root_dir, "train_binarized_list.txt")) as f:
        test_file = get_data_splits(f)
    test_dataset = ChestXrayDataset(root_dir,
                                    test_file, resize_px=resize_px)  
    
    if use_augmentation:
        synthetic_labels = "To-do: obtain disease labels for synthetic images"
        synth_root = '/data3/jessica/data/labelGAN/results_dir_multitask_generation_segm_new_4/vis_KDE_all/'
        augmentation_dataset = AugmentationDataset(synth_root, os.listdir(os.path.join(synth_root, 'imgs'))[0:aug_size],
                                                   os.listdir(os.path.join(synth_root, 'masks'))[0:aug_size], resize_px=resize_px) 
        train_dataset = torch.utils.data.ConcatDataset([train_dataset, augmentation_dataset])
    
    return train_dataset, valid_dataset, test_dataset

