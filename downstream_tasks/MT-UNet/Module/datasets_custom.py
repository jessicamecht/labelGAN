import torch
import numpy as np
import os
import matplotlib.pyplot as plt
import PIL
from PIL import Image,ImageOps
from torchvision import transforms
from torch.utils.data import Dataset
import pickle
import csv

class ChestXrayDataset(Dataset):

    def __init__(self, root_dir, image_and_labels, res=(1024, 1024), use_aug=False):

        self.image_and_labels = image_and_labels
        self.root_dir = root_dir
        self.res = res
        self.use_aug = use_aug

    def __len__(self):
        return len(self.image_and_labels)

    def __getitem__(self, idx):
 
        #Reading images
        img_name = os.path.join(self.root_dir, 'originals', f"{self.image_and_labels[idx][0]}_json", "img.png")
        image = Image.open(img_name)
        image = ImageOps.grayscale(image)
        
        #Reading segmentation Masks
        mask_name = os.path.join(self.root_dir, 'originals', f"{self.image_and_labels[idx][0]}_json", "label.png")
        mask = Image.open(mask_name)
        mask = np.array(ImageOps.grayscale(mask))
        mask[mask > 5] = 255
        
        #Extracting disease label
        label = self.image_and_labels[idx][1]
        
        #Converting to tensors and Resizing images
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Resize(self.res)])
        self.transform_hflip = transforms.functional.hflip
        
        image = self.transform(image)
        mask = self.transform(mask)
        
        return image, mask, label
    
class AugmentationDataset(Dataset):

    def __init__(self, image_paths, mask_paths, labels):
        
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.labels = labels
    
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
 
        #Reading images
        img_name = os.path.join(self.root_dir, 'synthetic/images', self.image_paths[idx])
        image = Image.open(img_name)
        image = ImageOps.grayscale(image)
        
        #Reading segmentation Masks
        mask_name = os.path.join(self.root_dir, 'synthetic/masks', self.mask_paths[idx])
        mask = Image.open(mask_name)
        mask = np.array(ImageOps.grayscale(mask))
        mask[mask > 5] = 255
        
        #Extracting disease label
        label = self.labels[idx]
        
        #Converting to tensors and Resizing images
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Resize(self.res)])
        self.transform_hflip = transforms.functional.hflip
        
        image = self.transform(image)
        mask = self.transform(mask)
        
        return image, mask, label

def get_data_splits(f):
    return [(image.split()[0], np.array(list(map(int, image.split()[1:])))) for image in f.readlines()]

def get_datasets(res = (256, 256), aug_size=None, use_augmentation = False):
    root_dir = '/home/rmpatil/multi_task_gen/data/vinbig_we_labeled/'
    
    with open(os.path.join(root_dir, "train_binarized_list.txt")) as f:
        train_file = get_data_splits(f)
    train_dataset = ChestXrayDataset(root_dir,
                                     train_file,
                                     res) 
    
    with open(os.path.join(root_dir, "val_binarized_list.txt")) as f:
        val_file = get_data_splits(f)
    valid_dataset = ChestXrayDataset(root_dir,
                                     val_file,
                                     res)   
    
    with open(os.path.join(root_dir, "test_binarized_list.txt")) as f:
        test_file = get_data_splits(f)
    test_dataset = ChestXrayDataset(root_dir,
                                    test_file,
                                    res)  
    
    if use_augmentation:
        with open(os.path.join(root_dir, "aug_binarized_list.txt")) as f:
            aug_file = get_data_splits(f)
        augmentation_dataset = AugmentationDataset(root_dir,
                                                   aug_file[:aug_size],
                                                   res) 
        train_dataset = torch.utils.data.ConcatDataset([train_dataset, augmentation_dataset])
    
    return train_dataset, valid_dataset, test_dataset

