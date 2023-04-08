import sys
from tqdm import tqdm

import datasets_custom_jessica
from hzhu_MTL_UNet import *

import torch
from torch import nn as nn
from torchmetrics import JaccardIndex

import os
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader

# torch.backends.cudnn.enabled=False

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
import os
import copy
import argparse


def dice_coeff(pred, target):
    smooth = 1.
    pred = torch.where(pred>=0.5, 1, 0)
    iflat = pred.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()
    
    return 1 - ((2. * intersection + smooth) /
              (iflat.sum() + tflat.sum() + smooth))

# create argument parser object
parser = argparse.ArgumentParser(description='')

# add arguments
parser.add_argument('-gpu_num', '--gpu_num', type=int, required=False, help='')
parser.add_argument('-use_augment', '--use_augment', type=bool, required=False, default=True, help='')
parser.add_argument('-num_epochs', '--num_epochs', type=int, required=False, default=50, help='')
parser.add_argument('-bs', '--bs', type=int, required=False, default=8, help='')
parser.add_argument('-aug_size', '--aug_size', type=int, required=False, default=1000, help='')
parser.add_argument('-resize_px', '--resize_px', type=int, required=False, default=512, help='')


# parse arguments from command line
args = parser.parse_args()

# use arguments
use_augmentation = args.use_augment
gpu_num = args.gpu_num
num_epochs = args.num_epochs
bs = args.bs
resize_px = args.resize_px
aug_size = args.aug_size
save_path = f"/data3/jessica/data/labelGAN/downstream_results/aug_size_{aug_size}_use_augment_{use_augmentation}_num_epochs_{num_epochs}_resize_px_{resize_px}_KDE/"

if not os.path.exists(save_path):
    os.mkdir(save_path)

train_dataset, valid_dataset, test_dataset = datasets_custom_jessica.get_datasets(aug_size=aug_size, use_augmentation = use_augmentation, resize_px=resize_px)
dataAll = {
            "Train": DataLoader(train_dataset, batch_size=bs, shuffle=True),
            "Valid": DataLoader(valid_dataset, batch_size=bs, shuffle=True),
            "Test": DataLoader(test_dataset, batch_size=1, shuffle=True) 
}

device = torch.device(f"cuda:{gpu_num}" if torch.cuda.is_available() else "cpu")
model = MTL_UNet(in_channels = 1, out_dict = {'class': 15, 'image': 1},)
model = model.to(device)

jaccard = JaccardIndex(task = 'binary', num_classes=2).to(device)

classification_loss = nn.BCELoss()
seg_pred_loss = nn.KLDivLoss()

val_check = 5
lr = 0.0002
beta1 = 0.5
optimizer = optim.Adam(model.parameters(), lr=lr, betas=(beta1, 0.999))

#Function for visualization
def visualize(data, pred, disease_pred, epoch, test=False, true=None, pred_ = None, pacc=None, dice=None, iou=None, atleastone=None, pred_perc=None):
    fig, axs = plt.subplots(1, 3, figsize=(10,10))
    img_plot = data["cxr"].detach().cpu()
    pred_plot = pred[0].detach().cpu()
    label_plot = data["gaze"].detach().cpu()
    
    axs[0].imshow(img_plot[0, 0])
    axs[0].set_title('Image')

    axs[1].imshow(pred_plot[0],cmap='gray')
    axs[1].set_title('Prediction')

    axs[2].imshow(label_plot[-1, 0],cmap='gray')
    axs[2].set_title('Label')
        
    plt.savefig(f"{save_path}/{epoch}.png")
    
    np.set_printoptions(precision=3)
    if not test:
        true = data['Y'].detach().cpu().numpy().tolist()
        pred = disease_pred.detach().cpu().numpy().tolist()
    else:
        true = true
        pred = pred_

    with open(f'{save_path}/{epoch}.txt', 'w') as f:
        f.write(f'\t{pacc}\t{dice}\t{iou}\t{atleastone}\t{pred_perc}')
        #for x, y in zip(true[0], pred[0]):
        #    f.write(f'{x}\t{round(y, 3)}')

# train model
loss_list = []
best_dice_epoch = 0
best_pixelacc_epoch = 0
best_classif_epoch = 0
sm = nn.Softmax(dim=0)
for epoch in tqdm(range(num_epochs)):
    model.train()

    for data in dataAll["Train"]:
        if epoch == 0: break
        X = data['cxr'].to(device).type(torch.float)

        Y_class = data['Y'].to(device)
        Y_seg = data['gaze'].to(device).type(torch.float)

        optimizer.zero_grad()
        Y_class_pred, Y_seg_pred = model(X)
        
        loss = classification_loss(Y_class_pred.type(torch.float), Y_class.type(torch.float)) + seg_pred_loss(Y_seg_pred.type(torch.float), Y_seg.type(torch.float))
        loss.backward()

        optimizer.step()
        loss_list.append(loss.detach().clone().cpu())
        del X, Y_class, Y_seg
    #visualize(data, Y_seg_pred, Y_class_pred)
    torch.save(model.state_dict(), f"{save_path}/model_{epoch}.pth")
    with torch.no_grad():
        losses = []
        all_preds = []
        all_true = []
        all_pixel_accs = []
        all_dices = []
        all_jaccard = []
        at_least_one = []
        pred_percs = []
        for data in dataAll["Valid"]:
            X = data['cxr'].to(device).type(torch.float)

            Y_class = data['Y'].to(device)
            Y_seg = data['gaze'].to(device).type(torch.float)

            Y_class_pred, Y_seg_pred = model(X)
            
            loss = classification_loss(Y_class_pred.type(torch.float), Y_class.type(torch.float)) + seg_pred_loss(Y_seg_pred.type(torch.float), Y_seg.type(torch.float))


            Y_seg_pred = sm(Y_seg_pred) > 0.5
            Y_seg = Y_seg > 0.5
            
            batch_size = Y_seg_pred.shape[0]
            num_pixels = batch_size * Y_seg_pred.shape[2] * Y_seg_pred.shape[3]
            num_correct_pixels = (Y_seg_pred == Y_seg).sum()
            pixel_accuracy = num_correct_pixels / num_pixels


            all_pixel_accs.append(pixel_accuracy.cpu().detach().numpy())
            dice = dice_coeff(Y_seg_pred, Y_seg)
            all_dices.append(dice.cpu().detach().numpy())
            losses.append(loss.detach().clone().cpu())
            
            all_jaccard.append(jaccard(Y_seg_pred, Y_seg).cpu().detach().numpy())
            true = data['Y'].detach().cpu().numpy().tolist()
            pred = Y_class_pred.detach().cpu().numpy().tolist()

            for i, elem in enumerate(data['Y'].detach().cpu().numpy()):
                t = (elem > 0.5 ).astype(int) 
                p = (Y_class_pred.detach().cpu().numpy()[i] > 0.5).astype(int)
                atleast = (p & t).sum() > 0
                pred_perc = (p == t).mean()
                at_least_one.append(atleast)
                pred_percs.append(pred_perc)
            all_preds.extend(pred)
            all_true.extend(true)
            del X, Y_class, Y_seg
        
        print("PIXEL ACC:", np.array(all_pixel_accs).mean())
        print("MEAN DICE:", np.array(all_dices).mean())
        print("MEAN IOU:", np.array(all_jaccard).mean())
        print("PRED PERC:", np.array(pred_percs).mean())
        print("at_least_one", np.array(at_least_one).mean())
        print(f"TEST LOSS EPOCH {epoch}:", np.array(losses).mean())
        visualize(data, Y_seg_pred, Y_class_pred, epoch, test=True, true=all_true, pred_= all_preds,
        pacc=np.array(all_pixel_accs).mean(), dice=np.array(all_dices).mean(), iou=np.array(all_jaccard).mean(), atleastone=np.array(at_least_one).mean(), pred_perc=np.array(pred_percs).mean())
    
    
    print("[TRAIN] Epoch : %d, Loss : %2.5f" % (epoch, np.mean(loss_list)))
#     if (epoch + 1) % val_check == 0:
    

#         cur_loss = validation(model, val_dataloader)
#         print("[VALIDATION] Epoch : %d, Loss : %2.5f" % (epoch, cur_loss))
#         if cur_loss < best_loss:
#             best_loss = cur_loss
#             torch.save(model.state_dict(), os.path.join("seg", model_path)) 