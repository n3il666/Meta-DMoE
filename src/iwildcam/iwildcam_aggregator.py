import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from tqdm import tqdm
from sklearn.metrics import f1_score
import os
from src.iwildcam.iwildcam_utils import *

def get_selector_accuracy(selector, models_list, data_loader, grouper, device, progress=True):
    selector.eval()
    correct = 0
    total = 0
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
            correct += pred.eq(y_true.view_as(pred)).sum().item()
            total += x.shape[0]
            try:
                pred_all = torch.cat((pred_all, pred.view_as(y_true)))
                y_all = torch.cat((y_all, y_true))
            except NameError:
                pred_all = pred.view_as(y_true)
                y_all = y_true
    
    y_all = y_all.detach().cpu()
    pred_all = pred_all.detach().cpu()
    f1 = f1_score(y_all,pred_all,average='macro', labels=torch.unique(y_all))

    return correct/total, f1

def train_model_selector(selector, model_name, models_list, device, root_dir='data',
                         batch_size=32, lr=1e-6, l2=0,
                         num_epochs=12, decayRate=0.96, save=True, test_way='ood'):
    for model in models_list:
        model.eval()
    
    train_loader, val_loader, _, grouper = get_data_loader(root_dir=root_dir,
                                                batch_size=batch_size, domain=None, 
                                                test_way=test_way, n_groups_per_batch=0)
    
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
                acc, f1 = get_selector_accuracy(selector, models_list, val_loader, 
                                                grouper, device, progress=False)
                
                print("Iter: {} || Loss: {:.4f} || Acc:{:.4f} || F1:{:.4f}".format(i, losses, acc, f1))
                losses = []
                
                if f1 > acc_best and save:
                    print("Saving model ...")
                    save_model(selector, model_name+"_selector", 0, test_way=test_way)
                    acc_best = f1
                
            i += 1
        scheduler.step()
