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
from wilds.common.data_loaders import get_train_loader, get_eval_loader
from wilds.common.grouper import CombinatorialGrouper

# Utils

def split_domains(num_experts, root_dir='data'):
    dataset = get_dataset(dataset='iwildcam', download=True, root_dir=root_dir)
    train_data = dataset.get_subset('train')
    locations = list(set(train_data.metadata_array[:,0].detach().numpy().tolist()))
    random.shuffle(locations)
    num_domains_per_super = len(locations) / float(num_experts)
    all_split = [[] for _ in range(num_experts)]
    for i in range(len(locations)):
        all_split[int(i//num_domains_per_super)].append(locations[i])
    return all_split

def get_subset_with_domain(dataset, split=None, domain=None, transform=None):
    if type(dataset) == wilds.datasets.wilds_dataset.WILDSSubset:
        subset = copy.deepcopy(dataset)
    else:
        subset = dataset.get_subset(split, transform=transform)
    if domain is not None:
        idx = np.argwhere(np.isin(subset.dataset.metadata_array[:,0][subset.indices], domain)).ravel()
        subset.indices = subset.indices[idx]
    return subset

def initialize_image_base_transform(dataset):
    transform_steps = []
    if dataset.original_resolution is not None and min(dataset.original_resolution)!=max(dataset.original_resolution):
        crop_size = min(dataset.original_resolution)
        transform_steps.append(transforms.CenterCrop(crop_size))
    transform_steps.append(transforms.Resize((448,448)))
    transform_steps += [
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]
    transform = transforms.Compose(transform_steps)
    return transform

def get_data_loader(batch_size=16, domain=None, test_way='id',
                    n_groups_per_batch=0, uniform_over_groups=False, root_dir='data'):
    dataset = get_dataset(dataset='iwildcam', download=True, root_dir=root_dir)
    grouper = CombinatorialGrouper(dataset, ['location'])
    
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
    
    return train_loader, val_loader, test_loader, grouper

def get_test_loader(batch_size=16, split='val', root_dir='data'):
    dataset = get_dataset(dataset='iwildcam', download=True, root_dir=root_dir)
    grouper = CombinatorialGrouper(dataset, ['location'])
    
    transform = initialize_image_base_transform(dataset)
    
    eval_data = get_subset_with_domain(dataset, split, domain=None, transform=transform)
    all_domains = list(set(grouper.metadata_to_group(eval_data.dataset.metadata_array[eval_data.indices]).tolist()))
    test_loader = []

    for domain in all_domains:
        domain_data = get_subset_with_domain(eval_data, split, domain=domain, transform=transform)
        domain_loader = get_eval_loader('standard', domain_data, batch_size=batch_size)
        test_loader.append((domain, domain_loader))
    
    return test_loader, grouper

def get_mask_grouper(root_dir='data'):
    dataset = get_dataset(dataset='iwildcam', download=True, root_dir=root_dir)
    grouper = CombinatorialGrouper(dataset, ['location'])
    return grouper

def save_model(model, name, epoch, test_way='ood'):
    if not os.path.exists("model/iwildcam"):
        os.makedirs("model/iwildcam")
    path = "model/iwildcam/{0}_best.pth".format(name)
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
        self.features = nn.Sequential(*list(original_model.children())[:layer])
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.shape[0], -1)
        return x

class fa_selector(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout = 0., out_dim=182, pool='mean'):
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
        self.num_ftrs = original_model.fc.in_features
        self.num_class = original_model.fc.out_features
        self.features = nn.Sequential(*list(original_model.children())[:layer])
        self.classifier = nn.Sequential(*list(original_model.children())[layer:])
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, self.num_ftrs)
        x = self.classifier(x)
        x = x.view(-1, self.num_class)
        return x

def StudentModel(device, num_classes=182, load_path=None):
    model = torchvision.models.resnet50(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
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

def get_models_list(device, num_domains=9, num_classes=182, pretrained=False, bb='res50'):
    models_list = []
    for _ in range(num_domains+1):
        model = torchvision.models.resnet50(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        model = model.to(device)
        models_list.append(model)
    return models_list
