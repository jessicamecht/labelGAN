import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
import sys
sys.path.append('..')
import torch
import torch.nn as nn
import pandas as pd
torch.manual_seed(0)
from generate_data import generate_data
import json
import numpy as np
import os
from tqdm import tqdm
import gc
from utils.utils import multi_acc, get_label_stas
import torch.optim as optim
import argparse
from torch.utils.data import DataLoader
device = 'cuda:3' if torch.cuda.is_available() else 'cpu'
from train_dataset import *
from label_model import *
from collections import Counter
import torchvision.models as models

def train_label_classif(args, checkpoint_path_label=""):
    files = os.listdir(args['annotation_image_latent_path_classification'])
    files = sorted(files)
    mask = ["tophat" not in elem for elem in files]
    files = np.array(files)[mask]
    print(len(files),';lkjhgf')
    label_data = labelDataLatent(files, args['annotation_image_latent_path_classification'], device)
    train_loader_classif = DataLoader(dataset=label_data, batch_size=args['batch_size'], shuffle=True)
    label_classifier_instance = latent_classifier(args['number_class_classification']).to(device)
    if checkpoint_path_label != "":
        checkpoint = torch.load(os.path.join(checkpoint_path_label, 'model_label_classif_number_' + '.pth'))
        label_classifier_instance.load_state_dict(checkpoint['model_state_dict'])
        checkpoint = torch.load(os.path.join(checkpoint_path_label, 'reshaper_number' + str(0) + '.pth'))
        label_classifier_instance.eval()
        reshaper.eval()
    else:
        label_classifier_instance.train()
        criterion = nn.CrossEntropyLoss()
        optimizer_label = optim.Adam(label_classifier_instance.parameters(), lr=0.1, weight_decay=1e-5)

    sm = nn.Softmax(dim=1)
    iteration = 0
    best_loss = 10000000
    for epoch in tqdm(range(10)):
        all_preds = []
        all_labels = []
        accs = []
        losses = []
        ids = []
        all_probs = []
        for X_batch, label in tqdm(train_loader_classif):
            X_batch = X_batch.detach()
            label = label.type(torch.LongTensor).to(device).squeeze()
            optimizer_label.zero_grad()
            y_pred = label_classifier_instance(X_batch.squeeze())
            loss = criterion(y_pred, label)
            acc = multi_acc(y_pred, label)
            
            loss.backward()
            optimizer_label.step()

            all_preds.extend(sm(y_pred).argmax(-1).cpu().detach())
            all_probs.extend(torch.max(sm(y_pred), dim=-1).values.cpu().detach())
            all_labels.extend(label.cpu().detach())
            losses.append(loss.item())
            accs.append(acc.cpu())
            iteration += 1
            if iteration % 5000 == 0:
                print('Epoch classif: ', str(epoch), 'loss', np.array(losses).mean(), 'acc', np.array(accs).mean(), "Acc Majority Vote: ")
                gc.collect()
                model_path = os.path.join(args['exp_dir'],
                                            'model_label_classif_' +  str(iteration) + '_number_' + str(0) + '.pth')   
                model_path_reshaper = os.path.join(args['exp_dir'],
                                            'reshaper_' +  str(iteration) + '_number_' + str(0) + '.pth')                  
                if checkpoint_path_label == "":
                    torch.save({'model_state_dict': label_classifier_instance.state_dict()},
                                model_path)

            if epoch > 3:
                if loss.item() < best_loss:
                    best_loss = loss.item()
                    break_count = 0
                else:
                    break_count += 1

                if break_count > 50:
                    stop_sign = 1
                    print("*************** Break, Total iters,", iteration, ", at epoch", str(epoch), "***************")
                    break
        df = pd.DataFrame()
        df["labels"] = all_labels
        df['preds'] = all_preds
        df['probs'] = all_probs
        df.to_csv("res.csv")
        print('Epoch classif: ', str(epoch), 'loss', np.array(losses).mean(), 'acc', np.array(accs).mean(), "Acc Majority Vote: ")

    model_path = os.path.join(args['exp_dir'], 'model_label_classif' + '_number_'+ '.pth')
    torch.save({'model_state_dict': label_classifier_instance.state_dict()},
                   model_path)

    gc.collect()


def main(args, checkpoint_path_segm=""):

    if args['category'] == 'car':
        from utils.data_util import car_20_palette as palette
    elif args['category'] == 'face':
        from utils.data_util import face_palette as palette
    elif args['category'] == 'bedroom':
        from utils.data_util import bedroom_palette as palette
    elif args['category'] == 'cat':
        from utils.data_util import cat_palette as palette
    elif args['category'] == 'xray':
        from utils.data_util import xray_palette as palette
    g_all, avg_latent, upsamplers = prepare_stylegan(args, device)

    gc.collect()
    classifier = segm_classifier((1+1), args['dim'][-1]).to(device)

    if checkpoint_path_segm != "":
        checkpoint = torch.load(os.path.join(checkpoint_path_segm, 'model_' + str(MODEL_NUMBER) + '.pth'))
        classifier.load_state_dict(checkpoint['model_state_dict'])
        classifier.eval()
    else:
        classifier.init_weights()
        classifier.train()

    class_weights = torch.tensor([0.3, 0.7]).to(device )
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(classifier.parameters() , lr=0.001)

    for i in range(args['max_training']-2, 50):#can only fit 2 images into memoory at a time 
        all_feature_maps_train_list, all_mask_train_all, num_data = prepare_data(args, palette, device, i, g_all, avg_latent, upsamplers)

        train_data = trainData(all_feature_maps_train_list,
                            all_mask_train_all, args)

        batch_size = args['batch_size']

        train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
        print(" *********************** Current dataloader length " +  str(len(train_loader)) + " ***********************")
        for MODEL_NUMBER in range(args['model_num']):

            iteration = 0
            break_count = 0
            best_loss = 10000000
            stop_sign = 0
            accs = []
            for epoch in tqdm(range(2)):
                iteration = 0
                break_count = 0
                best_loss = 10000000
                stop_sign = 0
                for X_batch, y_batch in train_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    y_batch = y_batch.type(torch.long)

                    optimizer.zero_grad()
                    y_pred = classifier(X_batch)
                    loss = criterion(y_pred, y_batch)
                    acc = multi_acc(y_pred, y_batch)
                    
                    if checkpoint_path_segm == "":
                        loss.backward()
                        optimizer.step()

                    iteration += 1
                    if iteration % 1000 == 0:
                        print('Epoch : ', str(epoch), 'iteration', iteration, 'loss', loss.item(), 'acc', acc)
                        gc.collect()


                    if iteration % 5000 == 0:
                        model_path = os.path.join(args['exp_dir'],
                                                'model_' +  str(iteration) + '_number_' + str(MODEL_NUMBER) + '.pth')
                        #print('Save checkpoint, Epoch : ', str(epoch), ' Path: ', model_path)
                        
                        if checkpoint_path_segm == "":
                            torch.save({'model_state_dict': classifier.state_dict()},
                                    model_path)

                    accs.append(acc.item())
                    if epoch > 3:
                        if loss.item() < best_loss:
                            best_loss = loss.item()
                            break_count = 0
                        else:
                            break_count += 1

                        if break_count > 50:
                            stop_sign = 1
                            print("*************** Break, Total iters,", iteration, ", at epoch", str(epoch), "***************")
                            break
                print('Epoch : ', str(epoch), 'loss', 'acc_label', np.array(accs).mean(), 'acc', np.array(accs).mean())


                if stop_sign == 1:
                    break

                model_path = os.path.join(args['exp_dir'],
                                    'model_' + '.pth')
                MODEL_NUMBER += 1
                print('save to:',model_path)
                torch.save({'model_state_dict': classifier.state_dict()},
                    model_path)

            gc.collect()
            torch.cuda.empty_cache()    # clear cache memory on GPU
            break
        del all_feature_maps_train_list, all_mask_train_all, num_data, train_data, train_loader, X_batch, y_batch, accs
        gc.collect()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--exp', type=str)
    parser.add_argument('--exp_dir', type=str,  default="")
    parser.add_argument('--generate_data', type=bool, default=False)
    parser.add_argument('--save_vis', type=bool, default=False)
    parser.add_argument('--start_step', type=int, default=0)
    parser.add_argument('--eval_interp', type=bool, default=False)
    parser.add_argument('--train_pixel_classif', type=bool, default=False)

    parser.add_argument('--resume', type=str,  default="")
    parser.add_argument('--resume_label', type=str,  default="")
    parser.add_argument('--num_sample', type=int,  default=2000)
    parser.add_argument('--train_label_classif', type=bool,  default=False)

    args = parser.parse_args()
    print(args.exp)
    opts = json.load(open(args.exp, 'r'))
    print("Opt", opts)

    if args.exp_dir != "":
        opts['exp_dir'] = args.exp_dir


    path =opts['exp_dir']
    if os.path.exists(path):
        pass
    else:
        os.system('mkdir -p %s' % (path))
        print('Experiment folder created at: %s' % (path))

    os.system('cp %s %s' % (args.exp, opts['exp_dir']))

    if args.eval_interp:
        main(opts, checkpoint_path_segm=args.resume, checkpoint_path_label=args.resume_label)
    if args.generate_data:
        print("GENERATE DATA")
        generate_data(opts, args.resume, args.resume_label, args.num_sample, vis=args.save_vis, start_step=args.start_step, device=device)
    if args.train_label_classif:
        train_label_classif(opts)
    if args.train_pixel_classif:
        main(opts)

