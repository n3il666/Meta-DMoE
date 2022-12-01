import os
import torch
import numpy as np
import random
import argparse
import pickle
from src.configs import default_param

OUT_DIM = {'iwildcam':182,
           'camelyon':1,
           'rxrx1':1139,
           'fmow':62,
           'poverty':1}

FEAT_DIM = {'iwildcam':2048,
            'camelyon':1024,
            'rxrx1':2048,
            'fmow':1024,
            'poverty':512}

def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--gpu', type=str, default='0') # We only support single gpu training for now
    parser.add_argument('--threads', type=int, default=12)

    parser.add_argument('--dataset', type=str, default='iwildcam', 
                        choices=['iwildcam', 'fmow', 'camelyon', 'rxrx1', 'poverty'])
    parser.add_argument('--data_dir', type=str, default='data')

    parser.add_argument('--load_trained_experts', action='store_true')
    parser.add_argument('--load_pretrained_aggregator', action='store_true')
    parser.add_argument('--load_pretrained_student', action='store_true')

    parser.add_argument('--test', action='store_true')

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
        from src.iwildcam.iwildcam_utils import fa_selector, StudentModel, get_models_list, get_feature_list
        from src.iwildcam.iwildcam_experts import train_exp, train_model, get_expert_split
        from src.iwildcam.iwildcam_aggregator import train_model_selector
        from src.iwildcam.iwildcam_train import train_kd, eval
    elif args.dataset == 'camelyon':
        from src.camelyon.camelyon_utils import fa_selector, StudentModel, get_models_list, get_feature_list
        from src.camelyon.camelyon_experts import train_exp, train_model, get_expert_split
        from src.camelyon.camelyon_aggregator import train_model_selector
        from src.camelyon.camelyon_train import train_kd, eval
    elif args.dataset == 'rxrx1':
        from src.rxrx1.rxrx1_utils import fa_selector, StudentModel, get_models_list, get_feature_list
        from src.rxrx1.rxrx1_experts import train_exp, train_model, get_expert_split
        from src.rxrx1.rxrx1_aggregator import train_model_selector
        from src.rxrx1.rxrx1_train import train_kd, eval
    elif args.dataset == 'fmow':
        from src.fmow.fmow_utils import fa_selector, StudentModel, get_models_list, get_feature_list
        from src.fmow.fmow_experts import train_exp, train_model, get_expert_split
        from src.fmow.fmow_aggregator import train_model_selector
        from src.fmow.fmow_train import train_kd, eval
    else:
        raise NotImplementedError

    name = f"{args.dataset}_{str(args.num_experts)}experts_seed{str(args.seed)}"

    models_list = get_models_list(device=device, num_domains=args.num_experts-1)

    try:
        with open(f"model/{args.dataset}/domain_split.pkl", "rb") as f:
            all_split, split_to_cluster = pickle.load(f)
    except FileNotFoundError:
        all_split, split_to_cluster = get_expert_split(args.num_experts, root_dir=args.data_dir)
        with open(f"model/{args.dataset}/domain_split.pkl", "wb") as f:
            pickle.dump((all_split, split_to_cluster), f)

    if args.load_trained_experts:
        print("Skip training domain specific experts...")
    else:
        print("Training domain specific experts...")
        train_exp(models_list, all_split, device, batch_size=args.expert_batch_size,
                lr=args.expert_lr, l2=args.expert_l2, num_epochs=args.expert_epoch,
                save=True, name=name, root_dir=args.data_dir)

    for i,model in enumerate(models_list):
        model.load_state_dict(torch.load(f"model/{args.dataset}/{name}_{str(i)}_exp_best.pth"))
    models_list = get_feature_list(models_list, device=device)

    selector = fa_selector(dim=FEAT_DIM[args.dataset], depth=args.aggregator_depth, heads=args.aggregator_heads, 
                        mlp_dim=FEAT_DIM[args.dataset]*2, dropout=args.aggregator_dropout,
                        out_dim=OUT_DIM[args.dataset]).to(device)
    if args.load_pretrained_aggregator:
        print("Skip pretraining knowledge aggregator...")
    else:
        print("Pretraining knowledge aggregator...")
        train_model_selector(selector, name+'_pretrained', models_list, device, root_dir=args.data_dir,
                            num_epochs=args.aggregator_pretrain_epoch, save=True)

    selector.load_state_dict(torch.load(f"model/{args.dataset}/{name}_pretrained_selector_best.pth"))

    student = StudentModel(device=device, num_classes=OUT_DIM[args.dataset])

    if args.load_pretrained_student:
        print("Skip pretraining student...")
    else:
        print("Pretraining student...")
        train_model(student, name+"_pretrained", device=device, 
                    num_epochs=args.student_pretrain_epoch, save=True,
                    root_dir=args.data_dir)

    student.load_state_dict(torch.load(f"model/{args.dataset}/{name}_pretrained_exp_best.pth"))

    print("Start meta-training...")
    train_kd(selector, name+"_meta", models_list, student, name+"_meta", split_to_cluster,
            device=device, batch_size=args.batch_size, sup_size=args.sup_size, 
            tlr=args.tlr, slr=args.slr, ilr=args.ilr, num_epochs=args.epoch, save=True, test_way='ood',
            root_dir=args.data_dir)

def test(args):

    if args.dataset == 'iwildcam':
        from src.iwildcam.iwildcam_utils import fa_selector, StudentModel, get_models_list, get_feature_list
        from src.iwildcam.iwildcam_train import eval
    elif args.dataset == 'camelyon':
        from src.camelyon.camelyon_utils import fa_selector, StudentModel, get_models_list, get_feature_list
        from src.camelyon.camelyon_train import eval
    elif args.dataset == 'rxrx1':
        from src.rxrx1.rxrx1_utils import fa_selector, StudentModel, get_models_list, get_feature_list
        from src.rxrx1.rxrx1_train import eval
    elif args.dataset == 'fmow':
        from src.fmow.fmow_utils import fa_selector, StudentModel, get_models_list, get_feature_list
        from src.fmow.fmow_train import eval

    name = f"{args.dataset}_{str(args.num_experts)}experts_seed{str(args.seed)}"
    models_list = get_models_list(device=device, num_domains=args.num_experts-1)
    for i,model in enumerate(models_list):
        model.load_state_dict(torch.load(f"model/{args.dataset}/{name}_{str(i)}_exp_best.pth"))
    models_list = get_feature_list(models_list, device=device)
    selector = fa_selector(dim=FEAT_DIM[args.dataset], depth=args.aggregator_depth, heads=args.aggregator_heads, 
                        mlp_dim=FEAT_DIM[args.dataset]*2, dropout=args.aggregator_dropout,
                        out_dim=OUT_DIM[args.dataset]).to(device)
    selector.load_state_dict(torch.load(f"model/{args.dataset}/{name}_meta_selector_best.pth"))

    student = StudentModel(device=device, num_classes=OUT_DIM[args.dataset]).to(device)
    student.load_state_dict(torch.load(f"model/{args.dataset}/{name}_meta_student_best.pth"))
    metrics = eval(selector, models_list, student, batch_size=args.sup_size,
                      device=device, ilr=args.ilr, test=True, root_dir=args.data_dir)
    if args.dataset == 'iwildcam':
        print(f"Test Accuracy:{metrics[1]:.4f} Test Macro-F1:{metrics[2]:.4f}")
        with open(f'result/{args.dataset}/result.txt', 'a+') as f:
            f.write(f"Seed: {args.seed} || Test Accuracy:{metrics[1]:.4f} || Test Macro-F1:{metrics[2]:.4f}")
    elif args.dataset in ['camelyon', 'rxrx1']:
        print(f"Test Accuracy:{metrics:.4f}")
        with open(f'result/{args.dataset}/result.txt', 'a+') as f:
            f.write(f"Seed: {args.seed} || Test Accuracy:{metrics:.4f}")
    elif args.dataset == 'fmow':
        print(f"WC Accuracy:{metrics[0]:.4f} Acc:{metrics[1]:.4f}")
        with open(f'result/{args.dataset}/result.txt', 'a+') as f:
            f.write(f"Seed: {args.seed} || WC Accuracy:{metrics[0]:.4f} || Acc:{metrics[1]:.4f}")

if __name__ == "__main__":
    args = get_parser()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    torch.set_num_threads(args.threads)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)
    if not os.path.exists(f"model/{args.dataset}"):
        os.makedirs(f"model/{args.dataset}")
    if not os.path.exists(f"log/{args.dataset}"):
        os.makedirs(f"log/{args.dataset}")
    if args.test:
        if not os.path.exists(f"result/{args.dataset}"):
            os.makedirs(f"result/{args.dataset}")
        test(args)
    else:
        train(args)
