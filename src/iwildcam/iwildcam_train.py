import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from tqdm import tqdm
import copy
import learn2learn as l2l
from sklearn.metrics import f1_score
import time
from datetime import datetime
from collections import defaultdict
import os
import wilds
from src.iwildcam.iwildcam_utils import *

def train_epoch(selector, selector_name, models_list, student, student_name, 
                train_loader, grouper, epoch, curr, mask_grouper, split_to_cluster,
                device, acc_best=0, tlr=1e-4, slr=1e-4, ilr=1e-3,
                batch_size=256, sup_size=24, test_way='id', save=False,
                root_dir='data'):
    for model in models_list:
        model.eval()
    
    student_ce = nn.CrossEntropyLoss()
    #teacher_ce = nn.CrossEntropyLoss()
    
    features = student.features
    head = student.classifier
    features = l2l.algorithms.MAML(features, lr=ilr)
    features.to(device)
    head.to(device)
    
    all_params = list(features.parameters()) + list(head.parameters())
    optimizer_s = optim.Adam(all_params, lr=slr)
    optimizer_t = optim.Adam(selector.parameters(), lr=tlr)
    
    i = 0
    
    losses = []
    
    iter_per_epoch = len(train_loader)
        
    for x, y_true, metadata in train_loader:
        selector.eval()
        head.eval()
        features.eval()
        
        z = grouper.metadata_to_group(metadata)
        z = set(z.tolist())
        assert len(z) == 1
        mask = mask_grouper.metadata_to_group(metadata)
        mask.apply_(lambda x: split_to_cluster[x])

        #sup_size = x.shape[0]//2
        x_sup = x[:sup_size]
        y_sup = y_true[:sup_size]
        x_que = x[sup_size:]
        y_que = y_true[sup_size:]
        mask = mask[:sup_size]

        x_sup = x_sup.to(device)
        y_sup = y_sup.to(device)
        x_que = x_que.to(device)
        y_que = y_que.to(device)
        
        with torch.no_grad():
            logits = torch.stack([model(x_sup).detach() for model in models_list], dim=-1)
            #logits[:, :, split_to_cluster[z]] = torch.zeros_like(logits[:, :, split_to_cluster[z]])
            #logits = logits.permute((0,2,1))
            logits = mask_feat(logits, mask, len(models_list), exclude=True)
        
        t_out = selector.get_feat(logits)  

        task_model = features.clone()
        task_model.module.eval()
        feat = task_model(x_que)
        feat = feat.view(feat.shape[0], -1)
        out = head(feat)
        with torch.no_grad():
            loss_pre = student_ce(out, y_que).item()/x_que.shape[0]
        
        feat = task_model(x_sup)
        feat = feat.view_as(t_out)

        inner_loss = l2_loss(feat, t_out)
        task_model.adapt(inner_loss)
        
        x_que = task_model(x_que)
        x_que = x_que.view(x_que.shape[0], -1)
        s_que_out = head(x_que)
        s_que_loss = student_ce(s_que_out, y_que)
        #t_sup_loss = teacher_ce(t_out, y_sup)
        
        s_que_loss.backward()
        
        optimizer_s.step()
        optimizer_t.step()
        optimizer_s.zero_grad()
        optimizer_t.zero_grad()
        
        #print("Step:{}".format(time.time() - t_1))
        #t_1 = time.time()
        with open('log/iwildcam/'+curr+'.txt', 'a+') as f:
            f.write('Iter: {}/{}, Loss Before:{:.4f}, Loss After:{:.4f}\r\n'.format(i,iter_per_epoch,
                                                                    loss_pre,
                                                                    s_que_loss.item()/x_que.shape[0]))
        losses.append(s_que_loss.item()/x_que.shape[0])
        
        if i == iter_per_epoch//2:
            losses = np.mean(losses)
            _, acc, f1 = eval(selector, models_list, student, sup_size, device=device,
                        ilr=ilr, test=False, progress=False, uniform_over_groups=False,
                        root_dir=root_dir)
            
            with open('log/iwildcam/'+curr+'.txt', 'a+') as f:
                f.write(f'Accuracy: {acc} || F1:{f1} \r\n')
            losses = []
            
            if f1 > acc_best and save:
                with open('log/iwildcam/'+curr+'.txt', 'a+') as f:
                    f.write("Saving model ...\r\n")
                save_model(selector, selector_name+"_selector", 0, test_way=test_way)
                save_model(student, student_name+"_student", 0, test_way=test_way)
                acc_best = f1
            
        i += 1
    return acc_best

def train_kd(selector, selector_name, models_list, student, student_name, split_to_cluster, device,
             batch_size=256, sup_size=24, tlr=1e-4, slr=1e-4, ilr=1e-5, num_epochs=30,
             decayRate=0.96, save=False, test_way='ood', root_dir='data'):
    
    train_loader, _, _, grouper = get_data_loader(root_dir=root_dir, batch_size=batch_size, domain=None,
                                                  test_way=test_way, n_groups_per_batch=1)

    mask_grouper = get_mask_grouper(root_dir=root_dir)

    curr = str(datetime.now())
    if not os.path.exists("log/iwildcam"):
        os.makedirs("log/iwildcam")
    #print(curr)
    print("Training log saved to log/iwildcam/"+curr+".txt")

    with open('log/iwildcam/'+curr+'.txt', 'a+') as f:
        f.write(selector_name+' '+student_name+'\r\n')
        f.write(f'tlr={tlr} slr={slr} ilr={ilr}\r\n')

    accu_best = 0
    for epoch in range(num_epochs):
        with open('log/iwildcam/'+curr+'.txt', 'a+') as f:
            f.write(f'Epoch: {epoch}\r\n')
        accu_epoch = train_epoch(selector, selector_name, models_list, student, student_name, 
                                train_loader, grouper, epoch, curr, mask_grouper, split_to_cluster,
                                device, acc_best=accu_best, tlr=tlr, slr=slr, ilr=ilr,
                                batch_size=batch_size, sup_size=sup_size, test_way=test_way, save=save,
                                root_dir=root_dir)
        accu_best = max(accu_best, accu_epoch)
        _, accu, f1 = eval(selector, models_list, student, sup_size, device=device, 
                    ilr=ilr, test=False, progress=False, uniform_over_groups=False,
                    root_dir=root_dir)

        with open('log/iwildcam/'+curr+'.txt', 'a+') as f:
            f.write(f'Accuracy: {accu} || F1:{f1} \r\n')
        
        if f1 > accu_best and save:
            with open('log/iwildcam/'+curr+'.txt', 'a+') as f:
                f.write("Saving model ...\r\n")
            save_model(selector, selector_name+"_selector", 0, test_way=test_way)
            save_model(student, student_name+"_student", 0, test_way=test_way)
            accu_best = f1
        tlr = tlr*decayRate
        slr = slr*decayRate

def eval(selector, models_list, student, batch_size, device, ilr=1e-5,
         test=False, progress=True, uniform_over_groups=False, root_dir='data'):

    if test:
        _, _, loader, grouper = get_data_loader(root_dir=root_dir, batch_size=batch_size, test_way='ood',
                                                n_groups_per_batch=1, uniform_over_groups=uniform_over_groups)
    else:
        _, loader, _, grouper = get_data_loader(root_dir=root_dir, batch_size=batch_size, test_way='ood',
                                                n_groups_per_batch=1, uniform_over_groups=uniform_over_groups)
    '''if test:
        loader, que_set, grouper = get_test_loader(batch_size=batch_size, test_way='test')
    else:
        loader, que_set, grouper = get_test_loader(batch_size=batch_size, test_way='val')'''

    features = student.features
    head = student.classifier
    head.to(device)

    student_maml = l2l.algorithms.MAML(features, lr=ilr)
    student_maml.to(device)

    correct = defaultdict(int)
    total = defaultdict(int)
    nor_correct = 0
    nor_total = 0
    old_domain = {}
    if progress:
        loader = tqdm(loader)

    for x_sup, y_sup, metadata in loader:
        student_maml.module.eval()
        selector.eval()
        head.eval()
        
        z = grouper.metadata_to_group(metadata)
        z = set(z.tolist())
        assert len(z) == 1

        x_sup = x_sup.to(device)
        y_sup = y_sup.to(device)
        task_model = student_maml.clone()
        task_model.eval()
        
        if list(z)[0] not in old_domain:
            with torch.no_grad():
                logits = torch.stack([model(x_sup).detach() for model in models_list], dim=-1)
                logits = logits.permute((0,2,1))
                t_out = selector.get_feat(logits)  
            
            feat = task_model(x_sup)
            feat = feat.view_as(t_out)
        
            kl_loss = l2_loss(feat, t_out)
            torch.cuda.empty_cache()
            task_model.adapt(kl_loss)
            old_domain[list(z)[0]] = task_model.state_dict()
        else:
            task_model.load_state_dict(old_domain[list(z)[0]])
        
        with torch.no_grad():
            task_model.module.eval()
            x_sup = task_model(x_sup)
            x_sup = x_sup.view(x_sup.shape[0], -1)
            s_que_out = head(x_sup)
            pred = s_que_out.max(1, keepdim=True)[1]
            c = pred.eq(y_sup.view_as(pred)).sum().item()
            correct[list(z)[0]] += c
            total[list(z)[0]] += x_sup.shape[0]
            nor_correct += c
            nor_total += x_sup.shape[0]
            try:
                pred_all = torch.cat((pred_all, pred.view_as(y_sup)))
                y_all = torch.cat((y_all, y_sup))
            except NameError:
                pred_all = pred.view_as(y_sup)
                y_all = y_sup
    
    y_all = y_all.detach().cpu()
    pred_all = pred_all.detach().cpu()
    f1 = f1_score(y_all,pred_all,average='macro', labels=torch.unique(y_all))
    mean_acc = []
    for key,value in correct.items():
        mean_acc.append(value/total[key])
    return np.mean(mean_acc).item(), nor_correct/nor_total, f1
