import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import math
import random
import pickle5 as pickle
from src.transformer import Transformer
import copy
import time
import os
import wilds
from wilds import get_dataset
from wilds.common.data_loaders import get_train_loader
from wilds.common.grouper import CombinatorialGrouper
from wilds.common.utils import split_into_groups

# Utils

def split_domains(num_experts, root_dir='data'):
    dataset = get_dataset(dataset='fmow', download=True, root_dir=root_dir)
    train_data = dataset.get_subset('train')
    year = list(set(train_data.metadata_array[:,1].detach().numpy().tolist()))
    random.shuffle(year)
    num_domains_per_super = len(year) / float(num_experts)
    all_split = [[] for _ in range(num_experts)]
    for i in range(len(year)):
        all_split[int(i//num_domains_per_super)].append(year[i])
    return all_split

def get_subset_with_domain(dataset, split=None, domain=None, transform=None, grouper=None):
    if type(dataset) == wilds.datasets.wilds_dataset.WILDSSubset:
        subset = copy.deepcopy(dataset)
    else:
        subset = dataset.get_subset(split, transform=transform)
    if domain is not None:
        if grouper is not None:
            z = grouper.metadata_to_group(subset.dataset.metadata_array[subset.indices])
            idx = np.argwhere(np.isin(z, domain)).ravel()
            subset.indices = subset.indices[idx]
        else:
            idx = np.argwhere(np.isin(subset.dataset.metadata_array[:,1][subset.indices], domain)).ravel()
            subset.indices = subset.indices[idx]
    return subset

def initialize_image_base_transform(dataset):
    transform_steps = []
    if dataset.original_resolution is not None and min(dataset.original_resolution)!=max(dataset.original_resolution):
        crop_size = min(dataset.original_resolution)
        transform_steps.append(transforms.CenterCrop(crop_size))
    transform_steps += [
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]
    transform = transforms.Compose(transform_steps)
    return transform

def get_data_loader(batch_size=16, domain=None, test_way='id',
                    n_groups_per_batch=0, uniform_over_groups=False, root_dir='data',
                    groupby_fields=['year'], return_dataset=False):
    dataset = get_dataset(dataset='fmow', download=True, root_dir=root_dir)
    grouper = CombinatorialGrouper(dataset, groupby_fields)
    
    transform = initialize_image_base_transform(dataset)
    
    train_data = get_subset_with_domain(dataset, 'train', domain=domain, transform=transform)
    if test_way == 'ood':
        val_data = dataset.get_subset('val', transform=transform)
        test_data = dataset.get_subset('test', transform=transform)
    else:
        val_data = get_subset_with_domain(dataset, 'id_val', domain=domain, transform=transform)
        test_data = get_subset_with_domain(dataset, 'id_test', domain=domain, transform=transform)
    
    
    if n_groups_per_batch == 0:
        #0 identify standard loader
        train_loader = get_train_loader('standard', train_data, batch_size=batch_size)
        val_loader = get_train_loader('standard', val_data, batch_size=batch_size)
        test_loader = get_train_loader('standard', test_data, batch_size=batch_size)
        
    else:
        # All use get_train_loader to enable grouper
        train_loader = get_train_loader('group', train_data, grouper=grouper, 
                                        n_groups_per_batch=n_groups_per_batch, batch_size=batch_size)
        val_loader = get_train_loader('group', val_data, grouper=grouper, 
                                        n_groups_per_batch=n_groups_per_batch, batch_size=batch_size,
                                        uniform_over_groups=uniform_over_groups)
        test_loader = get_train_loader('group', test_data, grouper=grouper, 
                                        n_groups_per_batch=n_groups_per_batch, batch_size=batch_size,
                                        uniform_over_groups=uniform_over_groups)
    if return_dataset:
        return train_loader, val_loader, test_loader, grouper, dataset
    return train_loader, val_loader, test_loader, grouper

def get_mask_grouper(root_dir='data'):
    dataset = get_dataset(dataset='fmow', download=True, root_dir=root_dir)
    grouper = CombinatorialGrouper(dataset, ['year'])
    return grouper

def save_model(model, name, epoch, test_way='ood'):
    if not os.path.exists("model/fmow"):
        os.makedirs("model/fmow")
    path = "model/fmow/{0}_best.pth".format(name)
    torch.save(model.state_dict(), path)

def mask_feat(feat, mask_index, num_experts, exclude=True):
    assert feat.shape[0] == mask_index.shape[0]
    if exclude:
        new_idx = [list(range(0, int(m.item()))) + list(range(int(m.item())+1, num_experts)) for m in mask_index]
        return feat[torch.arange(feat.shape[0]).unsqueeze(-1), :, new_idx]
    else:
        feat[list(range(feat.shape[0])), :, mask_index] = torch.zeros_like(feat[list(range(feat.shape[0])), :, mask_index])
        feat = feat.permute((0,2,1))
        return feat

def l2_loss(input, target):
    loss = torch.square(target - input)
    loss = torch.mean(loss)
    return loss

# Models
class ResNetFeature(nn.Module):
    def __init__(self, original_model, layer=-1):
        super(ResNetFeature, self).__init__()
        self.num_ftrs = original_model.classifier.in_features
        self.features = nn.Sequential(*list(original_model.children())[:layer])
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        
    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = x.view(-1, self.num_ftrs)
        return x

class fa_selector(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout = 0., out_dim=62, pool='mean'):
        super(fa_selector, self).__init__()
        self.transformer = Transformer(dim, depth, heads, mlp_dim, dropout=dropout)
        self.pool = pool
        self.mlp = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, out_dim)
        )

    def forward(self, x):
        x = self.transformer(x)
        if self.pool == 'mean':
            x = x.mean(dim=1)
        elif self.pool == 'max':
            x = x.max(dim=1)
        else:
            raise NotImplementedError
        x = self.mlp(x)
        return x
    
    def get_feat(self, x):
        x = self.transformer(x)
        if self.pool == 'mean':
            x = x.mean(dim=1)
        elif self.pool == 'max':
            x = x.max(dim=1)
        else:
            raise NotImplementedError
        return x

class DivideModel(nn.Module):
    def __init__(self, original_model, layer=-1):
        super(DivideModel, self).__init__()
        self.num_ftrs = original_model.classifier.in_features
        self.num_class = original_model.classifier.out_features
        self.features = nn.Sequential(*list(original_model.children())[:layer])
        self.features.add_module("avg_pool", nn.AdaptiveAvgPool2d((1,1)))
        self.classifier = nn.Sequential(*list(original_model.children())[layer:])
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, self.num_ftrs)
        x = self.classifier(x)
        x = x.view(-1, self.num_class)
        return x

def StudentModel(device, num_classes=62, load_path=None):
    model = torchvision.models.densenet121(pretrained=True)
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Linear(num_ftrs, num_classes)
    if load_path:
        model.load_state_dict(torch.load(load_path))
    model = DivideModel(model)
    model = model.to(device)
    return model

def get_feature_list(models_list, device):
    feature_list = []
    for model in models_list:
        feature_list.append(ResNetFeature(model).to(device))
    return feature_list

def get_models_list(device, num_domains=3, num_classes=62, pretrained=False, bb='d121'):
    models_list = []
    for _ in range(num_domains+1):
        model = torchvision.models.densenet121(pretrained=True)
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, num_classes)
        model = model.to(device)
        models_list.append(model)
    return models_list

def get_fmow_metrics(y_pred, y_true, metadata, dataset):
    worst_acc = float('inf')
    grouper = CombinatorialGrouper(dataset, ['region'])
    regions = grouper.metadata_to_group(metadata)
    unique_groups, group_indices, _ = split_into_groups(regions)
    for i_group in group_indices:
        correct = y_pred[i_group].eq(y_true[i_group].view_as(y_pred[i_group])).sum().item()
        total = y_pred[i_group].shape[0]
        worst_acc = min(correct/total, worst_acc)
    correct = y_pred.eq(y_true.view_as(y_pred)).sum().item()
    total = y_pred.shape[0]
    return correct/total, worst_acc
