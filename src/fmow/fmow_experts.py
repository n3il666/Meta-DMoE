import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import os
from src.fmow.fmow_utils import *

def get_model_accuracy(model, data_loader, grouper, device, domain=None, dataset=None):
    model.eval()

    for x, y_true, metadata in iter(data_loader):
        
        z = grouper.metadata_to_group(metadata)
        z = set(z.tolist())
        if domain is not None:
            assert z.issubset(set(domain))
        
        x = x.to(device)
        y_true = y_true.to(device)
        
        out = model(x)
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

def train_model(model, model_name, device, domain=None, batch_size=64, lr=1e-4, l2=0, 
                num_epochs=50, decayRate=0.96, save=False, test_way='ood', root_dir='data'):

    train_loader, val_loader, test_loader, grouper, dataset = get_data_loader(root_dir=root_dir,
                                                                batch_size=batch_size, domain=domain, 
                                                                test_way=test_way, n_groups_per_batch=0,
                                                                return_dataset=True)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=l2)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=decayRate)
    
    i = 0
    
    losses = []
    acc_best = 0

    tot = len(train_loader)
    
    for epoch in range(num_epochs):
        
        print(f"Epoch:{epoch} || Total:{tot}")
        
        for x, y_true, metadata in iter(train_loader):
            model.train()
            
            z = grouper.metadata_to_group(metadata)
            z = set(z.tolist())
            if domain is not None:
                assert z.issubset(set(domain))
    
            x = x.to(device)
            y_true = y_true.to(device)
            
            pred = model(x)

            loss = criterion(pred, y_true)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            losses.append(loss.item()/batch_size)
            
            if i % (tot//2) == 0 and i != 0:
                losses = np.mean(losses)
                acc, worst_acc = get_model_accuracy(model, val_loader, grouper, device=device, dataset=dataset)
                
                print("Iter: {} || Loss: {:.4f} || Acc:{:.4f} ||Worst Acc:{:.4f} ".format(i,
                                                                                        losses, acc, worst_acc))
                losses = []
                
                if worst_acc > acc_best and save:
                    print("Saving model ...")
                    save_model(model, model_name+"_exp", 0, test_way=test_way)
                    acc_best = worst_acc
                
            i += 1
        scheduler.step()

def train_exp(models_list, domain_specific_indices, device, batch_size=64, lr=1e-4, l2=0,
          num_epochs=50, decayRate=0.96, save=False, test_way='ood', name="Resnet50_experts",
          root_dir='data'):
    
    assert len(models_list) == len(domain_specific_indices)
    for i in range(len(models_list)):
        print(f"Training model {i} for domain", *domain_specific_indices[i])
        train_model(models_list[i], name+'_'+str(i), device=device, 
                    domain=domain_specific_indices[i], batch_size=batch_size, 
                    lr=lr, l2=l2, num_epochs=num_epochs, 
                    decayRate=decayRate, save=save, test_way=test_way,
                    root_dir=root_dir)

def get_expert_split(num_experts, root_dir='data'):
    all_split = split_domains(num_experts=num_experts, root_dir=root_dir)
    split_to_cluster = {d:i for i in range(len(all_split)) for d in all_split[i]}
    return all_split, split_to_cluster
