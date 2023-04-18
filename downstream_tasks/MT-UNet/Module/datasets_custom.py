import torch
import numpy as np
import os
import random
import matplotlib.pyplot as plt
import PIL
from PIL import Image,ImageOps, ImageEnhance
from torchvision import transforms
import pandas as pd
from torch.utils.data import Dataset

mlb = {'Aorticenlargement': np.array([1,0,0,0,0,0,0,0,0,0,0,0,0,0,0]),       
       'Atelectasis': np.array([0,1,0,0,0,0,0,0,0,0,0,0,0,0,0]),         
       'Calcification': np.array([0,0,1,0,0,0,0,0,0,0,0,0,0,0,0]),       
       'Cardiomegaly': np.array([0,0,0,1,0,0,0,0,0,0,0,0,0,0,0]),        
       'Consolidation': np.array([0,0,0,0,1,0,0,0,0,0,0,0,0,0,0]),       
       'ILD': np.array([0,0,0,0,0,1,0,0,0,0,0,0,0,0,0]),                 
       'Infiltration': np.array([0,0,0,0,0,0,1,0,0,0,0,0,0,0,0]),        
       'LungOpacity': np.array([0,0,0,0,0,0,0,1,0,0,0,0,0,0,0]),         
       'Nofinding': np.array([0,0,0,0,0,0,0,0,1,0,0,0,0,0,0]),          
       'NoduleMass': np.array([0,0,0,0,0,0,0,0,0,1,0,0,0,0,0]),         
       'Otherlesion': np.array([0,0,0,0,0,0,0,0,0,0,1,0,0,0,0]),        
       'Pleuraleffusion': np.array([0,0,0,0,0,0,0,0,0,0,0,1,0,0,0]),    
       'Pleuralthickening': np.array([0,0,0,0,0,0,0,0,0,0,0,0,1,0,0]),  
       'Pneumothorax': np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,1,0]),       
       'Pulmonaryfibrosis': np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,1])}  

class ChestXrayDataset(Dataset):

    def __init__(self, root_dir, image_and_labels, res=(1024, 1024), use_aug = False):

        self.image_and_labels = image_and_labels
        self.root_dir = root_dir
        self.res = res
        self.use_aug = use_aug

    def __len__(self):
        return len(self.image_and_labels)

    def __getitem__(self, idx):
 
        #Reading images
        img_name = os.path.join(self.root_dir, 'imgs', f"{self.image_and_labels[idx][0]}.png")
        image = Image.open(img_name)
        image = ImageOps.grayscale(image)
        
        #Reading segmentation Masks
        mask_name = os.path.join(self.root_dir, "masks", f"{self.image_and_labels[idx][0]}_mask.png")
        mask = Image.open(mask_name)
        mask = np.array(ImageOps.grayscale(mask))
        mask[mask > 5] = 255
        
        #Extracting disease label
        label = self.image_and_labels[idx][1]
        
        #Converting to tensors and Resizing images
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Resize(self.res)])
        
        image = self.transform(image)
        mask = self.transform(mask)
        
        if self.use_aug:
            
            if random.random() > 0.5:
                # Random crop
                i, j, h, w = transforms.RandomCrop.get_params(
                    image, output_size=(int(self.res[0]/1.75) , int(self.res[0]/1.75)))
                image = transforms.functional.crop(image, i, j, h, w)
                mask = transforms.functional.crop(mask, i, j, h, w)
                
                self.resize = transforms.Resize(self.res)
                image = self.resize(image)
                mask = self.resize(mask)

            # Random horizontal flipping
            if random.random() > 0.5:
                image = transforms.functional.hflip(image)
                mask = transforms.functional.hflip(mask)

            # Random vertical flipping
            if random.random() > 0.5:
                image = transforms.functional.vflip(image)
                mask = transforms.functional.vflip(mask)
        return torch.tensor(image), torch.tensor(mask), torch.tensor(label)
    
class AugmentationDatasetEmbedded(Dataset):

    def __init__(self, root_dir, aug_size, res=(1024, 1024),):

        self.root_dir = root_dir
        self.res = res
        self.mask_path = os.listdir(f"{self.root_dir}/masks/")
        random.shuffle(self.mask_path)
        self.mask_path = self.mask_path[:aug_size]
        train_csv = pd.read_csv('/home/jessica/labelGAN/downstream_tasks/vinbig/train.csv')
        self.image_id_to_labels = train_csv.groupby(by="image_id").class_name.apply(list).apply(lambda x: np.unique([elem.replace(" ", "").replace("/", "") for elem in x]))

    def __len__(self):
        return len(self.mask_path)

    def __getitem__(self, idx):
 
        #Reading images
        imgname = self.mask_path[idx]
        imname = imgname.replace("_mask", "").replace(".jpg", ".png")
        img_name = os.path.join(self.root_dir, 'imgs', f"{imname}")
        image = Image.open(img_name)
        
        #Converting to grayscale if RGB
        image = ImageOps.grayscale(image)
        
        #Reading segmentation Masks
        mask_name = os.path.join(self.root_dir, 'masks', f"{imgname}")
        mask = Image.open(mask_name)
        mask = np.array(ImageOps.grayscale(mask))
        mask[mask > 5] = 255
        
        #Extracting disease label
        label_list = self.image_id_to_labels[imgname.replace(".png", "").replace(".jpg", "").replace("_mask", "")]
        labels = np.zeros(15).astype(int)
        for label in label_list:
            labels = labels | mlb[label]
        
        #Converting to tensors and Resizing images
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Resize(self.res)])
        self.transform_hflip = transforms.functional.hflip
        
        image = self.transform(image)
        mask = self.transform(mask).type(torch.float)

        probability = torch.rand(1)
        if probability <= 0.5:
            image = self.transform_hflip(image)
            mask = self.transform_hflip(mask)

        return torch.tensor(image), torch.tensor(mask), torch.tensor(labels)
    
class AugmentationDatasetKDE(Dataset):

    def __init__(self, root_dir, aug_size, labels=None, res=(1024, 1024)):
        
        self.root_dir = root_dir
        self.labels = labels
        self.res = res
        self.mask_path = os.listdir(f"{self.root_dir}/masks/")
        fil = [elem.replace("_image", "_mask") for elem in os.listdir(f"{self.root_dir}/imgs/")]
        mask = [elem in fil for elem in self.mask_path]
        self.mask_path = np.array(self.mask_path)[mask]
        random.shuffle(self.mask_path)
        self.mask_path = self.mask_path[:aug_size]

    
    def __len__(self):
        return len(self.mask_path)

    def __getitem__(self, idx):
      
        #Reading segmentation Masks
        mask_name = os.path.join(self.root_dir, 'masks', self.mask_path[idx])
        mask = Image.open(mask_name)
        mask = np.array(ImageOps.grayscale(mask))
        mask[mask > 5] = 255

        img_name_str = self.mask_path[idx].replace("_mask", "_image")
        img_name = os.path.join(self.root_dir, 'imgs', img_name_str)
        image = Image.open(img_name)
        
        #Converting to grayscale if RGB
        image = ImageOps.grayscale(image)
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(2.0)
        
        
        #Extracting disease label
        if self.labels != None:
            label = self.labels[idx]
        else: 
            label = self.mask_path[idx].split("_")[1]
            label = mlb[label]
        
        #Converting to tensors and Resizing images
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Resize(self.res)])
        self.transform_hflip = transforms.functional.hflip
        
        image = self.transform(image)
        mask = self.transform(mask)
        
        #Flipping images and labels with probability 0.5
        probability = torch.rand(1)
        if probability <= 0.5:
            image = self.transform_hflip(image)
            mask = self.transform_hflip(mask)
        return torch.tensor(image), torch.tensor(mask), torch.tensor(label)

def get_data_splits(f):
    return [(image.split()[0], np.array(list(map(int, image.split()[1:-1])))) for image in f.readlines()]

def get_datasets(res = (256, 256), aug_size = None, use_augmentation = False, aug_types=["KDE"]):
    root_dir = '/data1/shared/jessica/data/labelGAN/vinbig_test_imgs_and_segm/'
    data_split_dir = '/home/jessica/labelGAN/downstream_tasks/vinbig/'
    
    with open(os.path.join(data_split_dir, "train_binarized_list.txt")) as f:
        train_file = get_data_splits(f)
    train_dataset = ChestXrayDataset(root_dir,
                                     train_file,
                                     res) 
    
    with open(os.path.join(data_split_dir, "val_binarized_list.txt")) as f:
        val_file = get_data_splits(f)
    val_dataset = ChestXrayDataset(root_dir,
                                   val_file,
                                   res)   
    
    with open(os.path.join(data_split_dir, "test_binarized_list.txt")) as f:
        test_file = get_data_splits(f)
    test_dataset = ChestXrayDataset(root_dir,
                                    test_file[0:45],
                                    res)  
    
    if use_augmentation:
        augmented_dataset = [train_dataset]
        for aug_type in aug_types:
            if aug_type == "baseline":
                print("in baseline")
                augment_set_inc = 50
                train_file = np.array(train_file, dtype = object)
                for i in range(int(aug_size / augment_set_inc)): 
                    augmentation_dataset = ChestXrayDataset(root_dir,
                                                            train_file[random.sample(range(train_file.shape[0]), augment_set_inc)],
                                                            res,
                                                            use_augmentation) 
                    augmented_dataset.append(augmentation_dataset)

            if aug_type == "KDE":
                print("in KDE")
                synth_root = '/data1/shared/jessica/drive_data/vis_KDE_all/'
                augmentation_dataset = AugmentationDatasetKDE(synth_root, aug_size, res=res) 
                augmented_dataset.append(augmentation_dataset)

            if aug_type == "Embedded":
                print("in embedded")
                root_dir = '/data1/shared/jessica/drive_data/train_images_embedded/'
                augmentation_dataset = AugmentationDatasetEmbedded(root_dir, aug_size, res)
                augmented_dataset.append(augmentation_dataset)
            
        train_dataset = torch.utils.data.ConcatDataset(augmented_dataset)
    
    return train_dataset, val_dataset, test_dataset

