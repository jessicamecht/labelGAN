"""
Copyright (C) 2021 NVIDIA Corporation.  All rights reserved.
Licensed under The MIT License (MIT)

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

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
import gc
from utils.utils import multi_acc, get_label_stas
import torch.optim as optim
import argparse
from torch.utils.data import DataLoader
device = 'cuda:2' if torch.cuda.is_available() else 'cpu'
from train_dataset import *
from label_model import *

def main(args, checkpoint_path=""
         ):

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

    all_feature_maps_train_list, all_mask_train_list, num_data, labels = prepare_data(args, palette)
    print('torch.tensor(all_feature_maps_train_list', torch.stack(all_feature_maps_train_list).shape)
    all_feature_maps_train_all = torch.concat(all_feature_maps_train_list, axis=0)
    all_mask_train_all = torch.concat(all_mask_train_list, axis=0)

    train_data = trainData(all_feature_maps_train_all,
                           all_mask_train_all)
    
    label_loader = labelData(torch.stack(all_feature_maps_train_list), labels)

    count_dict = get_label_stas(train_data)

    max_label = max([*count_dict])
    print(" *********************** max_label " + str(max_label) + " ***********************")


    print(" *********************** Current number data " + str(num_data) + " ***********************")


    batch_size = args['batch_size']

    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)

    print(" *********************** Current dataloader length " +  str(len(train_loader)) + " ***********************")
    for MODEL_NUMBER in range(args['model_num']):

        gc.collect()

        label_classifier_instance = label_classifier(len(np.unique(labels)), 18*512)
        classifier = segm_classifier((max_label+1), args['dim'][-1])

        if checkpoint_path != "":
            checkpoint = torch.load(os.path.join(checkpoint_path, 'model_' + str(MODEL_NUMBER) + '.pth'))
            classifier.load_state_dict(checkpoint['model_state_dict'])
            classifier.eval()
        else:
            classifier.init_weights()
            classifier.train()
            label_classifier_instance.init_weights()
            label_classifier_instance.train()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(classifier.parameters() , lr=0.001)
        optimizer_label = optim.Adam(label_classifier_instance.parameters() , lr=0.001)

        iteration = 0
        break_count = 0
        best_loss = 10000000
        stop_sign = 0
        accs, accs_label = [], []
        for epoch in range(50):
            for X_batch, label in label_loader:
                X_batch, label = X_batch.to(device), label.to(device)
                optimizer_label.zero_grad()
                y_pred = label_classifier_instance(X_batch)
                loss = criterion(y_pred, label)
                acc = multi_acc(y_pred, label)
                
                if checkpoint_path == "":
                    loss.backward()
                    optimizer.step()

                iteration += 1
                if iteration % 1000 == 0:
                    print('Epoch : ', str(epoch), 'iteration', iteration, 'loss', loss.item(), 'acc', acc)
                    gc.collect()
                if iteration % 50000 == 0:
                    model_path = os.path.join(args['exp_dir'],
                                              'model_label_classif_' +  str(iteration) + '_number_' + str(MODEL_NUMBER) + '.pth')                    
                    if checkpoint_path == "":
                        torch.save({'model_state_dict_label': label_classifier_instance.state_dict()},
                                   model_path)

                accs_label.append(acc.item())
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

            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                y_batch = y_batch.type(torch.long)

                optimizer.zero_grad()
                y_pred = classifier(X_batch)
                loss = criterion(y_pred, y_batch)
                acc = multi_acc(y_pred, y_batch)
                
                if checkpoint_path == "":
                    loss.backward()
                    optimizer.step()

                iteration += 1
                if iteration % 1000 == 0:
                    print('Epoch : ', str(epoch), 'iteration', iteration, 'loss', loss.item(), 'acc', acc)
                    gc.collect()


                if iteration % 50000 == 0:
                    model_path = os.path.join(args['exp_dir'],
                                              'model_label_' +  str(iteration) + '_number_' + str(MODEL_NUMBER) + '.pth')
                    #print('Save checkpoint, Epoch : ', str(epoch), ' Path: ', model_path)
                    
                    if checkpoint_path == "":
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
        main(opts, checkpoint_path=args.resume)
    if args.generate_data:
        generate_data(opts, args.resume, args.num_sample, vis=args.save_vis, start_step=args.start_step)
    else:
        main(opts)

