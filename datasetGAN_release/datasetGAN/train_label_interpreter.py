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

def find_majority(votes):
    vote_count = Counter(votes)
    top_two = vote_count.most_common(2)
    if len(top_two)>1 and top_two[0][1] == top_two[1][1]:
        # It is a tie
        return 0
    return top_two[0][0]

def train_label_classif(args, checkpoint_path_label=""):
    '''path = '/data3/jessica/data/labelGAN/pre_processed_training_data1/'
    files = os.listdir(path)
    files = sorted(files)
    mask = ["Cardiomegaly" in elem or "Nofinding" in elem or 'Aorticenlargement' in elem for elem in files]
    files = np.array(files)[mask]
    reshaper = Reshaper(args['classification_map_size'], args['classification_channels']).to(device)'''
    path = '/home/jessica/labelGAN/Image2StyleGAN/images/generated_latents_from_class_distr/'
    files = os.listdir(path)
    files = sorted(files)
    mask = ["tophat" not in elem for elem in files]
    #mask = ["Cardiomegaly" in elem or "Nofinding" in elem or 'Aorticenlargement' in elem for elem in files]
    files = np.array(files)[mask]
    #label_data = labelData(files, path, reshaper, device)
    label_data = labelDataLatent(files, path, device)
    train_loader_classif = DataLoader(dataset=label_data, batch_size=args['batch_size'], shuffle=True)
    '''label_classifier_instance = models.resnet50(pretrained=True)
    num_ftrs = label_classifier_instance.fc.in_features
    label_classifier_instance.fc = nn.Linear(num_ftrs, 15)
    label_classifier_instance = label_classifier_instance.to(device)'''
    #label_classifier_instance = label_classifier(args['number_class_classification'], args['classification_map_size'], args['classification_channels']).to(device)#StyleGANClassifier(16).to(device)
    #label_classifier_instance.init_weights()
    label_classifier_instance = latent_classifier(args['number_class_classification']).to(device)
    if checkpoint_path_label != "":
        checkpoint = torch.load(os.path.join(checkpoint_path_label, 'model_label_classif_number' + str(0) + '.pth'))
        label_classifier_instance.load_state_dict(checkpoint['model_state_dict'])
        checkpoint = torch.load(os.path.join(checkpoint_path_label, 'reshaper_number' + str(0) + '.pth'))
        reshaper.load_state_dict(checkpoint['model_state_dict'])
        label_classifier_instance.eval()
        reshaper.eval()
    else:
        label_classifier_instance.train()
        #reshaper.train()
        criterion = nn.CrossEntropyLoss()
        optimizer_label = optim.Adam(label_classifier_instance.parameters(), lr=0.1, weight_decay=1e-5)

        #optimizer_label = optim.Adam(list(label_classifier_instance.parameters()) + list(reshaper.parameters()), lr=0.1, weight_decay=1e-5)
    sm = nn.Softmax(dim=1)
    iteration = 0
    best_loss = 10000000
    for epoch in tqdm(range(250)):
        all_preds = []
        all_labels = []
        accs = []
        losses = []
        ids = []
        all_probs = []
        feats = []
        acc_major = []
        #for X_batch, label, image_ids, fs in tqdm(train_loader_classif):
        for X_batch, label in tqdm(train_loader_classif):
            X_batch = X_batch.detach()
            #print(X_batch)
            label = label.type(torch.LongTensor).to(device).squeeze()
            optimizer_label.zero_grad()
            y_pred = label_classifier_instance(X_batch.squeeze())
            loss = criterion(y_pred, label)
            #print(loss)
            acc = multi_acc(y_pred, label)
            
            loss.backward()
            optimizer_label.step()

            all_preds.extend(sm(y_pred).argmax(-1).cpu().detach())
            all_probs.extend(torch.max(sm(y_pred), dim=-1).values.cpu().detach())
            all_labels.extend(label.cpu().detach())
            #ids.extend(image_ids)
            losses.append(loss.item())
            accs.append(acc.cpu())
            #feats.extend(fs)

            #majority_label = find_majority(list(y_pred.argmax(-1).cpu().detach()))
            #acc_major.append(majority_label == label[0].cpu().detach())
            iteration += 1
            if iteration % 5000 == 0:
                print('Epoch classif: ', str(epoch), 'loss', np.array(losses).mean(), 'acc', np.array(accs).mean(), "Acc Majority Vote: ", np.array(acc_major).mean())
                gc.collect()
                model_path = os.path.join(args['exp_dir'],
                                            'model_label_classif_' +  str(iteration) + '_number_' + str(0) + '.pth')   
                model_path_reshaper = os.path.join(args['exp_dir'],
                                            'reshaper_' +  str(iteration) + '_number_' + str(0) + '.pth')                  
                if checkpoint_path_label == "":
                    torch.save({'model_state_dict': label_classifier_instance.state_dict()},
                                model_path)
                    #torch.save({'model_state_dict': reshaper.state_dict()},
                    #            model_path_reshaper)

            if epoch > 3:
                if loss.item() < best_loss:
                    best_loss = loss.item()
                    break_count = 0
                #else:
                #    break_count += 1

                #if break_count > 50:
                #    stop_sign = 1
                #    print("*************** Break, Total iters,", iteration, ", at epoch", str(epoch), "***************")
                #    break
        df = pd.DataFrame()
        df["labels"] = all_labels
        df['preds'] = all_preds
        df['probs'] = all_probs
        #df['ids'] = ids
        #df['feats'] = feats
        df.to_csv("res.csv")
        print('Epoch classif: ', str(epoch), 'loss', np.array(losses).mean(), 'acc', np.array(accs).mean(), "Acc Majority Vote: ", np.array(acc_major).mean())
        #np.save(all_preds, f'preds_epoch_{epoch}.npy')
        #np.save(all_labels, f'labels_epoch_{epoch}.npy')
        #np.save(ids, f'image_ids_epoch_{epoch}.npy')
    torch.save({'model_state_dict': label_classifier_instance.state_dict()},
                   model_path)
    #torch.save({'model_state_dict': reshaper.state_dict()},
    #               model_path_reshaper)
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

    all_feature_maps_train_list, all_mask_train_all, num_data = prepare_data(args, palette, device)

    train_data = trainData(all_feature_maps_train_list,
                           all_mask_train_all, args)

    count_dict = get_label_stas(train_data)

    max_label = max([*count_dict])
    print(" *********************** max_label " + str(max_label) + " ***********************")
    print(" *********************** Current number data " + str(num_data) + " ***********************")

    batch_size = args['batch_size']

    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)

    print(" *********************** Current dataloader length " +  str(len(train_loader)) + " ***********************")
    for MODEL_NUMBER in range(args['model_num']):
        gc.collect()
        classifier = segm_classifier((max_label+1), args['dim'][-1]).to(device)

        if checkpoint_path_segm != "":
            checkpoint = torch.load(os.path.join(checkpoint_path_segm, 'model_' + str(MODEL_NUMBER) + '.pth'))
            classifier.load_state_dict(checkpoint['model_state_dict'])
            classifier.eval()
        else:
            classifier.init_weights()
            classifier.train()

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(classifier.parameters() , lr=0.001)

        iteration = 0
        break_count = 0
        best_loss = 10000000
        stop_sign = 0
        accs = []
        for epoch in tqdm(range(50)):
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
                                  'model_' + str(MODEL_NUMBER) + '.pth')
            MODEL_NUMBER += 1
            print('save to:',model_path)
            torch.save({'model_state_dict': classifier.state_dict()},
                   model_path)

        gc.collect()
        torch.cuda.empty_cache()    # clear cache memory on GPU


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--exp', type=str)
    parser.add_argument('--exp_dir', type=str,  default="")
    parser.add_argument('--generate_data', type=bool, default=False)
    parser.add_argument('--save_vis', type=bool, default=False)
    parser.add_argument('--start_step', type=int, default=0)
    parser.add_argument('--eval_interp', type=bool, default=False)

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
        generate_data(opts, args.resume, args.resume_label, args.num_sample, vis=args.save_vis, start_step=args.start_step, device=device)
    if args.train_label_classif:
        train_label_classif(opts)
    else:
        main(opts)

