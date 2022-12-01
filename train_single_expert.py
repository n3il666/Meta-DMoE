import os
import torch
import numpy as np
import random
import argparse
import pickle
from src.configs import default_param

def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--gpu', type=str, default='0') # We only support single gpu training for now
    parser.add_argument('--threads', type=int, default=12)

    parser.add_argument('--dataset', type=str, default='iwildcam', 
                        choices=['iwildcam', 'fmow', 'camelyon', 'rxrx1', 'poverty'])
    parser.add_argument('--data_dir', type=str, default='data')

    parser.add_argument('--expert_idx', type=int)

    args = parser.parse_args()
    args_dict = args.__dict__
    args_dict.update(default_param[args.dataset])
    args = argparse.Namespace(**args_dict)
    return args
    
def set_seed(seed):
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def train(args):

    if args.dataset == 'iwildcam':
        from src.iwildcam.iwildcam_utils import get_models_list
        from src.iwildcam.iwildcam_experts import train_model, get_expert_split
    elif args.dataset == 'camelyon':
        from src.camelyon.camelyon_utils import get_models_list
        from src.camelyon.camelyon_experts import train_model, get_expert_split
    elif args.dataset == 'rxrx1':
        from src.rxrx1.rxrx1_utils import get_models_list
        from src.rxrx1.rxrx1_experts import train_model, get_expert_split
    elif args.dataset == 'fmow':
        from src.fmow.fmow_utils import get_models_list
        from src.fmow.fmow_experts import train_model, get_expert_split
    else:
        raise NotImplementedError

    name = f"{args.dataset}_{str(args.num_experts)}experts_seed{str(args.seed)}"

    models_list = get_models_list(device=device, num_domains=0)

    try:
        with open(f"model/{args.dataset}/domain_split.pkl", "rb") as f:
            all_split, split_to_cluster = pickle.load(f)
    except FileNotFoundError:
        all_split, split_to_cluster = get_expert_split(args.num_experts, root_dir=args.data_dir)
        with open(f"model/{args.dataset}/domain_split.pkl", "wb") as f:
            pickle.dump((all_split, split_to_cluster), f)

    print(f"Training model {args.expert_idx} for domain ", *all_split[args.expert_idx])
    train_model(models_list[0], name+'_'+str(args.expert_idx), device=device, 
                domain=all_split[args.expert_idx], batch_size=args.expert_batch_size, 
                lr=args.expert_lr, l2=args.expert_l2, num_epochs=args.expert_epoch, 
                save=True, root_dir=args.data_dir)

if __name__ == "__main__":
    args = get_parser()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    torch.set_num_threads(args.threads)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)
    train(args)
