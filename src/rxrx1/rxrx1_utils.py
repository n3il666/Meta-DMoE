import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
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

# Utils

def split_domains(num_experts, root_dir='data'):
    dataset = get_dataset(dataset='rxrx1', download=True, root_dir=root_dir)
    train_data = dataset.get_subset('train')
    experiment = list(set(train_data.metadata_array[:,1].detach().numpy().tolist()))
    random.shuffle(experiment)
    num_domains_per_super = len(experiment) / float(num_experts)
    all_split = [[] for _ in range(num_experts)]
    for i in range(len(experiment)):
        all_split[int(i//num_domains_per_super)].append(experiment[i])
    return all_split

def get_subset_with_domain(dataset, split, domain=None, transform=None):
    subset = dataset.get_subset(split, transform=transform)
    if domain is not None:
        idx = np.argwhere(np.isin(subset.dataset.metadata_array[:,1][subset.indices], domain)).ravel()
        subset.indices = subset.indices[idx]
    return subset

def initialize_image_base_transform(is_training):
    def standardize(x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=(1, 2))
        std = x.std(dim=(1, 2))
        std[std == 0.] = 1.
        return TF.normalize(x, mean, std)
    t_standardize = transforms.Lambda(lambda x: standardize(x))

    angles = [0, 90, 180, 270]
    def random_rotation(x: torch.Tensor) -> torch.Tensor:
        angle = angles[torch.randint(low=0, high=len(angles), size=(1,))]
        if angle > 0:
            x = TF.rotate(x, angle)
        return x
    t_random_rotation = transforms.Lambda(lambda x: random_rotation(x))

    if is_training:
        transforms_ls = [
            t_random_rotation,
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            t_standardize,
        ]
    else:
        transforms_ls = [
            transforms.ToTensor(),
            t_standardize,
        ]
    transform = transforms.Compose(transforms_ls)
    return transform

def get_data_loader(batch_size=16, domain=None, test_way='id',
                    n_groups_per_batch=0, uniform_over_groups=False, root_dir='data'):
    dataset = get_dataset(dataset='rxrx1', download=True, root_dir=root_dir)
    grouper = CombinatorialGrouper(dataset, ['experiment'])
    
    transform_train = initialize_image_base_transform(True)
    transform_test = initialize_image_base_transform(False)
    
    train_data = get_subset_with_domain(dataset, 'train', domain=domain, transform=transform_train)
    if test_way == 'ood':
        val_data = dataset.get_subset('val', transform=transform_test)
        test_data = dataset.get_subset('test', transform=transform_test)
    else:
        val_data = get_subset_with_domain(dataset, 'id_val', domain=domain, transform=transform_test)
        test_data = get_subset_with_domain(dataset, 'id_test', domain=domain, transform=transform_test)
    
    
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

def get_mask_grouper(root_dir='data'):
    dataset = get_dataset(dataset='rxrx1', download=True, root_dir=root_dir)
    grouper = CombinatorialGrouper(dataset, ['experiment'])
    return grouper

def save_model(model, name, epoch, test_way='ood'):
    if not os.path.exists("model/rxrx1"):
        os.makedirs("model/rxrx1")
    path = "model/rxrx1/{0}_best.pth".format(name)
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
    def __init__(self, dim, depth, heads, mlp_dim, dropout = 0., out_dim=1139, pool='mean'):
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

def StudentModel(device, num_classes=1139, load_path=None):
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

def get_models_list(device, num_domains=2, num_classes=1139, pretrained=False, bb='res50'):
    models_list = []
    for _ in range(num_domains+1):
        model = torchvision.models.resnet50(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        model = model.to(device)
        models_list.append(model)
    return models_list
