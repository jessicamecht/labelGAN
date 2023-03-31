import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
import sys
sys.path.append('..')
import torch
import torch.nn as nn
torch.manual_seed(0)
import json
import numpy as np
import os
from tqdm import tqdm
import gc
from utils.utils import multi_acc, get_label_stas
import torch.optim as optim
import argparse
from torch.utils.data import DataLoader
device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
from train_dataset import *
from label_model import *

def main(args, checkpoint_path_segm="", checkpoint_path_label=""):

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

    latent_all = np.load(args['annotation_image_latent_path'])
    
    latent_all = torch.from_numpy(latent_all)

    all_mask_train_all, imagenames_classif, affine_layers, labels = prepare_data(args, palette, device, g_all, upsamplers=upsamplers, latent_all=latent_all)
    #all_feature_maps_train_list, all_mask_train_all, num_data, imagenames_classif, affine_layers, labels = prepare_data(args, palette, device)

    train_data = trainDataOptimized(latent_all, all_mask_train_all, g_all, upsamplers, args, device)
    #train_data = trainData(all_feature_maps_train_list,
    #                       all_mask_train_all, args)
    print("number of classif instances", len(imagenames_classif))
    print('affine_layers_upsamples[1].shape, affine_layers_upsamples[2].shape', len(affine_layers), len(labels))

    label_data = labelData(affine_layers, labels, args)

    '''count_dict = get_label_stas(train_data)

    max_label = max([*count_dict])
    print(" *********************** max_label " + str(max_label) + " ***********************")
    print(" *********************** Current number data " + str(num_data) + " ***********************")'''

    batch_size = args['batch_size']

    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    train_loader_classif = DataLoader(dataset=label_data, batch_size=batch_size)#, drop_last=True)

    print(" *********************** Current dataloader length " +  str(len(train_loader)) + " ***********************")
    for MODEL_NUMBER in range(args['model_num']):
        gc.collect()
        feat_size = args['classification_map_size']
        label_classifier_instance = label_classifier(args['number_class_classification'], feat_size).to(device)#StyleGANClassifier(16).to(device)
        classifier = segm_classifier(2, args['dim'][-1]).to(device)

        if checkpoint_path_segm != "":
            checkpoint = torch.load(os.path.join(checkpoint_path_segm, 'model_' + str(MODEL_NUMBER) + '.pth'))
            classifier.load_state_dict(checkpoint['model_state_dict'])
            classifier.eval()
        else:
            classifier.init_weights()
            classifier.train()
            label_classifier_instance.init_weights()
        if checkpoint_path_label != "":
            checkpoint = torch.load(os.path.join(checkpoint_path_label, 'model_label_classif_' + str(MODEL_NUMBER) + '.pth'))
            label_classifier_instance.load_state_dict(checkpoint['model_state_dict'])
            label_classifier_instance.eval()
        else:
            label_classifier_instance.train()

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(classifier.parameters() , lr=0.001)
        optimizer_label = optim.Adam(label_classifier_instance.parameters() , lr=0.01)

        iteration = 0
        break_count = 0
        best_loss = 10000000
        stop_sign = 0
        for epoch in tqdm(range(250)):
            all_preds = []
            all_labels = []
            accs = []
            losses = []
            for X_batch, label in train_loader_classif:
                X_batch = X_batch.to(device)
                label = label.type(torch.LongTensor).to(device).squeeze(0)
                optimizer_label.zero_grad()
                y_pred = label_classifier_instance(X_batch)

                #print(torch.softmax(y_pred, axis=-1).argmax(axis=1), label)
                loss = criterion(y_pred, label)
                acc = multi_acc(y_pred, label)
                all_preds.append(y_pred.argmax(-1).item())
                all_labels.append(label.item())
                losses.append(loss.item())
                accs.append(acc.cpu())
                loss.backward()
                optimizer_label.step()

                iteration += 1
                if iteration % 50000 == 0:
                    gc.collect()
                    model_path = os.path.join(args['exp_dir'],
                                              'model_label_classif_' +  str(iteration) + '_number_' + str(MODEL_NUMBER) + '.pth')                    
                    if checkpoint_path_label == "":
                        torch.save({'model_state_dict': label_classifier_instance.state_dict()},
                                   model_path)

                if epoch > 3:
                    if loss.item() < best_loss:
                        best_loss = loss.item()
                        break_count = 0
                    '''else:
                        break_count += 1

                    if break_count > 50:
                        stop_sign = 1
                        print("*************** Break, Total iters,", iteration, ", at epoch", str(epoch), "***************")
                        break'''
            print('Epoch classif: ', str(epoch), 'loss', np.array(losses).mean(), 'acc', np.array(accs).mean())
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


                if iteration % 50000 == 0:
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

        gc.collect()
        model_path = os.path.join(args['exp_dir'],
                                  'model_' + str(MODEL_NUMBER) + '.pth')
        MODEL_NUMBER += 1
        print('save to:',model_path)
        torch.save({'model_state_dict': classifier.state_dict()},
                   model_path)
        torch.save({'model_state_dict_label': label_classifier_instance.state_dict()},
                   model_path)
        gc.collect()


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
        main(opts, checkpoint_path_segm=args.resumes, checkpoint_path_label=args.resume_label)
    if args.generate_data:
        generate_data(opts, args.resume, args.resume_label, args.num_sample, vis=args.save_vis, start_step=args.start_step, device=device)
    else:
        main(opts)

