import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from tqdm import tqdm
import os
from src.fmow.fmow_utils import *

def get_selector_accuracy(selector, models_list, data_loader, grouper, device, progress=True, dataset=None):
    selector.eval()
    #mean_correct = 0
    if progress:
        data_loader = tqdm(data_loader)
    for x, y_true, metadata in data_loader:
        #z = grouper.metadata_to_group(metadata)
        #z = set(z.tolist())
        #assert z.issubset(set(meta_indices))

        x = x.to(device)
        y_true = y_true.to(device)
        
        with torch.no_grad():
            features = torch.stack([model(x).detach() for model in models_list], dim=-1)
            features = features.permute((0,2,1))
            out = selector(features)
            
            pred = out.max(1, keepdim=True)[1]
            try:
                pred_all = torch.cat((pred_all, pred.view_as(y_true)))
                y_all = torch.cat((y_all, y_true))
                metadata_all = torch.cat((metadata_all, metadata))
            except NameError:
                pred_all = pred.view_as(y_true)
                y_all = y_true
                metadata_all = metadata
    acc, worst_acc = get_fmow_metrics(pred_all, y_all, metadata_all, dataset)
    return acc, worst_acc

def train_model_selector(selector, model_name, models_list, device, root_dir='data',
                         batch_size=64, lr=3e-6, l2=0,
                         num_epochs=20, decayRate=0.96, save=True, test_way='ood'):
    for model in models_list:
        model.eval()
    
    train_loader, val_loader, _, grouper, dataset = get_data_loader(root_dir=root_dir,
                                                batch_size=batch_size, domain=None, 
                                                test_way=test_way, n_groups_per_batch=0,
                                                return_dataset=True)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(selector.parameters(), lr=lr, weight_decay=l2)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=decayRate)
    
    i = 0
    
    losses = []
    acc_best = 0

    tot = len(train_loader)
    
    for epoch in range(num_epochs):
        
        print(f"Epoch:{epoch}|| Total:{tot}")
        
        for x, y_true, metadata in train_loader:
            selector.train()
            
            z = grouper.metadata_to_group(metadata)
            z = set(z.tolist())
            #assert z.issubset(set(meta_indices))
    
            x = x.to(device)
            y_true = y_true.to(device)
            
            with torch.no_grad():
                features = torch.stack([model(x).detach() for model in models_list], dim=-1)
                features = features.permute((0,2,1))
            out = selector(features)

            loss = criterion(out, y_true)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            losses.append(loss.item()/batch_size)
            
            if i % (tot//2) == 0 and i != 0:
                losses = np.mean(losses)
                acc, wc_acc = get_selector_accuracy(selector, models_list, val_loader, 
                                                grouper, device, progress=False, dataset=dataset)
                
                print("Iter: {}/{} || Loss: {:.4f} || Acc:{:.4f} || WC Acc:{:.4f} ".format(i, tot, losses, 
                                                                                            acc, wc_acc))
                losses = []
                
                if wc_acc > acc_best and save:
                    print("Saving model ...")
                    save_model(selector, model_name+"_selector", 0, test_way=test_way)
                    acc_best = wc_acc
                
            i += 1
        scheduler.step()
