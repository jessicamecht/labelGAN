import sys
from tqdm import tqdm

import datasets_custom
from dice_loss import DiceLoss
from hzhu_MTL_UNet import *

import torch
from torch import nn as nn
import os
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
from torchmetrics.classification import BinaryJaccardIndex

# torch.backends.cudnn.enabled=False

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
import os
import copy
import argparse

# create argument parser object
parser = argparse.ArgumentParser(description='')

# add arguments
parser.add_argument('-gpu_num', '--gpu_num', type=int, required=False, help='')
parser.add_argument('-use_augment', '--use_augment', action=argparse.BooleanOptionalAction)
parser.add_argument('-num_epochs', '--num_epochs', type=int, required=False, default=50, help='')
parser.add_argument('-train_bs', '--train_bs', type=int, required=False, default=8, help='')
parser.add_argument('-val_bs', '--val_bs', type=int, required=False, default=2, help='')
parser.add_argument('-test_bs', '--test_bs', type=int, required=False, default=1, help='')
parser.add_argument('-aug_size', '--aug_size', type=int, required=False, default=1000, help='')
parser.add_argument('-aug_type', '--aug_type', type=int, required=False, default=1000, help='')
parser.add_argument('-resize_px', '--resize_px', type=int, required=False, default=512, help='')


# parse arguments from command line
args = parser.parse_args()

# use arguments
use_augmentation = args.use_augment
augmentation_type = args.aug_type
aug_size = args.aug_size
gpu_num = args.gpu_num
num_epochs = args.num_epochs
train_bs = args.train_bs
test_bs = args.test_bs
val_bs = args.val_bs
resize_px = args.resize_px

save_path = f"/home/rmpatil/multi_task_gen/data/downstream_results/aug_size_{aug_size}_use_augment_{use_augmentation}_num_epochs_{num_epochs}_resize_px_{resize_px}/"

if not os.path.exists(save_path):
    os.makedirs(save_path)

train_dataset, valid_dataset, test_dataset = datasets_custom.get_datasets(res=(resize_px, resize_px), aug_size=aug_size, use_augmentation = use_augmentation)
dataAll = {
            "Train": DataLoader(train_dataset, batch_size=train_bs, shuffle=True),
            "Valid": DataLoader(valid_dataset, batch_size=val_bs, shuffle=True),
            "Test": DataLoader(test_dataset, batch_size=test_bs, shuffle=True) 
}

device = torch.device(f"cuda:{gpu_num}" if torch.cuda.is_available() else "cpu")
model = MTL_UNet_Main(in_channels = 1, out_dict = {'class': 15, 'image': 1},)
model = model.to(device)

classification_loss = nn.BCELoss()
seg_pred_loss = DiceLoss()

val_check = 5
lr = 0.0002
beta1 = 0.5
optimizer = optim.Adam(model.parameters(), lr=lr, betas=(beta1, 0.999))

#Function for visualization
def visualize(cxr, mask, Y, seg_pred, disease_pred, epoch):
    batchsize = cxr.size()[0]
    fig, axs = plt.subplots(batchsize, 3, figsize=(10, 10))
    
    for i in range(batchsize): 
        img_plot = cxr[i].permute(1,2,0).detach().cpu()
        seg_pred_plot = seg_pred[i].permute(1,2,0).detach().cpu()
        label_plot = mask[i].permute(1,2,0).detach().cpu()

        axs[i, 0].imshow(img_plot)
        axs[i, 0].set_title('Image')

        axs[i, 1].imshow(seg_pred_plot, cmap='gray')
        axs[i, 1].set_title('Prediction')

        axs[i, 2].imshow(label_plot, cmap='gray')
        axs[i, 2].set_title('Label')


#         np.set_printoptions(precision=3)
#         true = Y[i].detach().cpu().numpy().tolist()
#         pred = disease_pred[i].detach().cpu().numpy().tolist()

#         print(f"True v/s Pred:")
#         for x, y in zip(true, pred):
#             print(f'{x}\t{round(y, 3)}')
    
    plt.savefig(f"{save_path}/{epoch}.png")
    
    return


def get_atleast_one_metric(pred, true):
    pred = pred > 0.5
    return torch.any(torch.logical_and(pred, true)).long().item()

#Validation Loop
def validation(model):
    losses = []
    model.eval()
    for cxr, mask, Y in dataAll["Valid"]:
        X = cxr.to(device)
        seg = mask.type(torch.int8).to(device)
        Y = Y.to(device)
        
        Y_pred, seg_pred = model(X)

        loss = model.compute_loss(
                    y_class_pred = Y_pred.type(torch.float),
                    y_image_pred = seg_pred.type(torch.float),
                    y_class_true = Y.type(torch.float), 
                    y_image_true = seg.type(torch.float),
                    loss_class = classification_loss,
                    loss_image = seg_pred_loss)
        
        losses.append(loss.item())
    return np.mean(losses)

# test model
def test(model):
    # uncomment below lines of code for batch testing later
#     model = MTL_UNet_Main(in_channels = 1, out_dict = {'class': 15, 'image': 1})
#     weights = torch.load(model_path)
#     model.load_state_dict(weights)
#     model = model.to(device)
    dice_scores = []
    atleast_one_score = []
    jaccard = BinaryJaccardIndex()
    
    model.eval()
    with torch.no_grad():
        for cxr, mask, Y in dataAll["Test"]:
            X = cxr.to(device)
            seg = mask.type(torch.int8).to(device)
            Y = Y.to(device)

            Y_pred, seg_pred = model(X)
            
            dice_scores.append(jaccard(seg_pred.cpu(), seg.cpu()))
            atleast_one_score.append(get_atleast_one_metric(Y_pred, Y))
    
    return np.mean(dice_scores), np.mean(atleast_one_score)

# train model
def train():
    loss_list = []
    best_loss = np.inf
    for epoch in tqdm(range(num_epochs)):
        model.train()

        for cxr, mask, Y in dataAll["Train"]:
            X = cxr.to(device).type(torch.float)

            Y = Y.to(device).type(torch.float)
            seg = mask.to(device).type(torch.int8)
            
            optimizer.zero_grad()
            Y_pred, seg_pred = model(X)

            loss = model.compute_loss(
                    y_class_pred = Y_pred.type(torch.float),
                    y_image_pred = seg_pred.type(torch.float),
                    y_class_true = Y.type(torch.float), 
                    y_image_true = seg.type(torch.float),
                    loss_class = classification_loss,
                    loss_image = seg_pred_loss)

    #         loss = seg_pred_loss(Y_seg_pred.type(torch.float), Y_seg.type(torch.float)) + classification_loss(Y_class_pred.type(torch.float), Y_class.type(torch.float))

            loss.backward()

            optimizer.step()
            loss_list.append(loss.detach().clone().cpu())

        visualize(cxr, mask, Y, seg_pred, Y_pred, epoch)

        print("[TRAIN] Epoch : %d, Loss : %2.5f" % (epoch, np.mean(loss_list)))
                
        if (epoch + 1) % val_check == 0:
            cur_loss = validation(model)
            print("[VALIDATION] Epoch : %d, Loss : %2.5f" % (epoch, cur_loss))
            if cur_loss < best_loss:
                best_loss = cur_loss
                torch.save(model.state_dict(), os.path.join(save_path, f"model_{epoch}.pt")) 
                
                dsc, custom_acc = test(model)
                print("[TEST] Epoch : %d, DSC: %2.5f, ACC: %.3f" % (epoch, dsc, custom_acc))


if __name__ == "__main__":
    train()