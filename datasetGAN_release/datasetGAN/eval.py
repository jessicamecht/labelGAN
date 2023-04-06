import torch 
import pandas as pd 
import numpy as np 
import os 
import sys
sys.path.append('/home/jessica/labelGAN/datasetGAN_release/datasetGAN')
from label_model import latent_classifier
from train_dataset import labelDataLatent
from tqdm import tqdm
import torch.nn as nn 
device = "cpu"
from torch.utils.data import DataLoader

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

def multi_acc(y_pred, y_test):
    y_pred_softmax = torch.log_softmax(y_pred, dim=1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim=1)

    correct_pred = (y_pred_tags == y_test).float()
    acc = correct_pred.sum() / len(correct_pred)

    acc = acc * 100

    return acc

def calc_test_eval(train_loader_classif, label_classifier_instance, fil):
    all_preds = []
    all_labels = []
    accs = []
    losses = []
    ids = []
    all_probs = []
    sm = nn.Softmax(dim=1)
    corr_preds = []
    for X_batch, labels, ids in tqdm(train_loader_classif):
        #labels = [fil[id] for id in ids]
        X_batch = X_batch.detach()
        y_pred = label_classifier_instance(X_batch.squeeze())
        class_pred = sm(y_pred).argmax(-1).cpu().detach()
        all_labels.extend(labels)

        for i, pred in enumerate(class_pred):
            corr_preds.append(pred == labels.squeeze()[i])#pred.item() in labels[i])
        del X_batch, y_pred
    print("TEST ACC",  np.array(corr_preds).mean())
    print(np.array(corr_preds)[np.array(all_labels) == 1].mean())