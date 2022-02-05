import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import torchvision
import torchvision.utils as vision_utils
import copy
from matplotlib.ticker import NullFormatter
import json
import math
import argparse
import random

from models import get_model, initialize_weights
from utils import get_pgd_perturbation, get_udp_perturbation, get_trades_perturbation
from utils import get_acc, get_pgd_acc, get_udp_acc
from utils import eval_with_autoattack
from data import get_dataset
from eval_params import get_eval_params


def get_args():
    parser = argparse.ArgumentParser()
    # General training params
    parser.add_argument('--batch_size_train', default=256, type=int)
    parser.add_argument('--batch_size_eval', default=1024, type=int)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--device', default='cuda:0', type=str)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--scheduler', default='none', choices=['triangle', 'multistep', 'none'])
    parser.add_argument('--opt', default='adam', choices=['adam', 'sgd'])
    parser.add_argument('--eval_freq', default=400, type=int) # in iterations
    parser.add_argument('--ckpt_freq', default=1, type=int) # in epochs
    parser.add_argument('--results_base_folder', default="./exps", type=str)
    parser.add_argument('--initialize_preactresnet', action='store_true')
    parser.add_argument('--test-with-autoattack', action='store_true')
    # Perturbation params
    parser.add_argument('--adv_training', default='none', choices=['pgd', 'udp', 'udp-reg', 'trades', 'udp+trades', 'none'])
    parser.add_argument('--mixed_training', action='store_true')
    parser.add_argument('--eps', default=0.1, type=float)
    parser.add_argument('--alpha', default=0.01, type=float)
    parser.add_argument('--reg_scalar', default=1.0, type=float)
    parser.add_argument('--attack_iters', default=-1, type=int)
    parser.add_argument('--attack_rs', action='store_true')
    parser.add_argument('--udp_sample_iters', default='uniform', choices=['uniform', 'not-uniform', 'none'])
    parser.add_argument('--train_w_a_scheduler', action='store_true')
    # Dataset and model
    parser.add_argument('--model', default='lenet', choices=['lenet', 'mlp', 'preactresnet18'])
    parser.add_argument('--dataset', default='fashion-mnist', choices=['mnist', 'fashion-mnist', 'svhn', 'cifar10', 'cifar10-ldr', 'cifar10-daug'])
    return parser.parse_args()


def train(model, opt, train_dl, valid_dl, test_dl, eval_parameters, max_epoch=10, adv_training=None, mixed_training=False, reg_scalar=1.0,
          eps=0.1, alpha=0.02, attack_iters=5, rs=False, clamp_val=None, train_w_a_scheduler=False, scheduler=None, udp_sample_iters='none',
          eval_freq=400, ckpt_freq=1, ckpt_path="", start_epoch=0, best_valid_pgd_acc=-1, early_stopped_model=None, aa_test=False):
  
    itr, best_model = -1, None

    stats = {"valid_acc": [], "valid_pgd_acc": [], "last_iterate_autoattack_linf_acc": [], 
             "test_acc": None, "last_iterate_autoattack_l2_acc": []}

    model.train()

    for epoch in range(start_epoch, max_epoch):
        for x, y in train_dl:
            itr += 1

            if adv_training == 'pgd':
                if (not mixed_training) or (mixed_training and itr % 2 == 0): 
                    delta = get_pgd_perturbation(model, x, y, eps, alpha, attack_iters, 
                                                       rs, clamp_val, use_alpha_scheduler=train_w_a_scheduler)
                    x = x + delta
                loss = F.cross_entropy(model(x), y)
            elif adv_training == 'udp':
                if (not mixed_training) or (mixed_training and itr % 2 == 0): 
                    delta = get_udp_perturbation(model, x, y, eps, alpha, attack_iters, 
                                                 rs, clamp_val, use_alpha_scheduler=train_w_a_scheduler, 
                                                 sample_iters=udp_sample_iters)
                    x = x + delta
                loss = F.cross_entropy(model(x), y)
            elif adv_training == 'trades':
                delta = get_trades_perturbation(model, x, y, eps, alpha, attack_iters, 
                                                rs, clamp_val, use_alpha_scheduler=train_w_a_scheduler)
                logits_1 = model(x)
                logits_2 = model(x + delta)
                p_1 = torch.softmax(logits_1, dim=1)
                p_2 = torch.softmax(logits_2, dim=1)
                loss = F.cross_entropy(logits_1, y) + reg_scalar * (-(p_1 * (p_2+1e-8).log()).sum(dim=1)).mean()
            elif adv_training == 'udp+trades':
                delta = get_udp_perturbation(model, x, y, eps, alpha, attack_iters, 
                                             rs, clamp_val, use_alpha_scheduler=train_w_a_scheduler, 
                                             sample_iters=udp_sample_iters)
                logits_1 = model(x)
                logits_2 = model(x + delta)
                p_1 = torch.softmax(logits_1, dim=1)
                p_2 = torch.softmax(logits_2, dim=1)
                loss = F.cross_entropy(logits_1, y) + reg_scalar * (-(p_1 * (p_2+1e-8).log()).sum(dim=1)).mean()
            elif adv_training == 'udp-reg':
                delta = get_udp_perturbation(model, x, y, eps, alpha, attack_iters, 
                                             rs, clamp_val, use_alpha_scheduler=train_w_a_scheduler, 
                                             sample_iters=udp_sample_iters)
                loss = F.cross_entropy(model(x), y) + reg_scalar * F.cross_entropy(model(x+delta), y)
            elif adv_training == 'none':
                loss = F.cross_entropy(model(x), y)
                
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            if scheduler is not None:
                scheduler.step()

            if itr % eval_freq == 0:
                model.eval()
                valid_acc = get_acc(model, valid_dl)
                valid_pgd_acc = get_pgd_acc(model, valid_dl, eps, alpha, attack_iters, rs=True, clamp_val=clamp_val, use_alpha_scheduler=True)
                p_s = f"{epoch}:{itr} [train] loss: {loss.item():.3f} [valid] acc: {valid_acc:.3f} PGD-acc: {valid_pgd_acc:.3f} "
                stats['valid_acc'].append((itr, valid_acc))
                stats['valid_pgd_acc'].append((itr, valid_pgd_acc))
                
                if valid_pgd_acc > best_valid_pgd_acc:
                    best_valid_pgd_acc = valid_pgd_acc
                    early_stopped_model = copy.deepcopy(model.state_dict())
                    
                if itr != 0 and scheduler is not None:
                    p_s += f"[lr] {scheduler.get_last_lr()[0]:.5f} "

                print(p_s)
                if math.isnan(loss.item()): 
                    raise(ValueError("Loss is NaN."))
                model.train()
                
        if epoch % ckpt_freq == 0:
            torch.save({'last_model': model.state_dict(), 
                        'last_opt': opt.state_dict(),
                        'last_scheduler': scheduler.state_dict() if scheduler is not None else None,
                        'early_stopped_model': early_stopped_model,
                        'last_epoch': epoch,
                        'best_valid_pgd_acc': best_valid_pgd_acc
                       }, ckpt_path)    

    model.eval()
    if aa_test:
        max_data_size = eval_parameters['max_data_size']
        for idx, eps_eval in enumerate(eval_parameters['eps']): # test on multiple eps
            li_rob_acc = eval_with_autoattack(model, test_dl, eps_eval, bs=eval_parameters['batch_size_eval'], max_data_size=max_data_size)
            stats["last_iterate_autoattack_linf_acc"].append((eps_eval, li_rob_acc))
            print(f"[test L-inf] (eps={eps_eval}) last-iterate autoattack-rob-acc: {li_rob_acc:.3f} ") 

        for idx, eps_eval in enumerate(eval_parameters['l2_eps']): # test on multiple eps
            li_rob_acc = eval_with_autoattack(model, test_dl, eps_eval, bs=eval_parameters['batch_size_eval'], norm='L2', max_data_size=max_data_size)
            stats["last_iterate_autoattack_l2_acc"].append((eps_eval, li_rob_acc))
            print(f"[test L2] (eps={eps_eval}) last-iterate autoattack-rob-acc: {li_rob_acc:.3f} ") 
        
    test_acc = get_acc(model, test_dl)
    stats['test_acc'] = test_acc
    print(f"[test] test-acc: {test_acc:.3f} ") 

    return stats


def main(args): 
    
    args.device = torch.device(args.device)
    
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    train_dl, valid_dl, test_dl = get_dataset(args)
    
    start_epoch, best_valid_pgd_acc, early_stopped_model = 0, -1, None
    
    print(f"Train dataset length: {len(train_dl.dataset)}")
    print(f"Valid dataset length: {len(valid_dl.dataset)}")
    print(f"Test dataset length: {len(test_dl.dataset)}")
    
    model = get_model(args.model, args.dataset).to(args.device)
    if args.model == 'preactresnet18' and args.initialize_preactresnet:
        model.apply(initialize_weights)
    eval_parameters = get_eval_params(args.dataset)
    
    if args.opt == 'adam':
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    else:
        opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
    
    if args.scheduler != 'none':
        if args.scheduler == 'triangle':
            scheduler = torch.optim.lr_scheduler.CyclicLR(opt, 0, args.lr, 
                                                          step_size_up=(len(train_dl)*args.epochs)//2, 
                                                          mode='triangular', cycle_momentum=False)
        elif args.scheduler == 'multistep':
            n_iters = len(train_dl)*args.epochs
            milestones = [0.25*n_iters, 0.5*n_iters, 0.75*n_iters] # hard-coded steps for now
            scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=milestones, gamma=0.3)
        else:
            raise NotImplementedError(f"Unknown scheduler type: {args.scheduler}.")
    else:
        scheduler = None

    
    if args.attack_iters == -1:
        args.attack_iters = round(args.eps/args.alpha)
        
    exp_name = f"at={args.adv_training}_att_iters={args.attack_iters}_eps={args.eps}_alpha={args.alpha}_ep={args.epochs}" + \
               f"_lrmax={args.lr}_reg_scalar={args.reg_scalar}_dataset={args.dataset}_mixed_training={args.mixed_training}" + \
               f"_train_w_a_scheduler={args.train_w_a_scheduler}_model={args.model}_scheduler={args.scheduler}_seed={args.seed}" + \
               f"_attack_rs={args.attack_rs}_udp_sample_iters={args.udp_sample_iters}_opt={args.opt}"
    
    ckpt_path = f"{args.results_base_folder}/{args.dataset}/{args.model}/{args.adv_training}/{exp_name}"
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)
    else:
        if os.path.isfile(f"{ckpt_path}/summary.json"): # the experiment was already completed
            sys.exit(0)
        if os.path.isfile(f"{ckpt_path}/ckpt.pt"):
            ckpt_dict = torch.load(f"{ckpt_path}/ckpt.pt")
            model.load_state_dict(ckpt_dict["last_model"])
            opt.load_state_dict(ckpt_dict["last_opt"])
            start_epoch = ckpt_dict["last_epoch"]+1
            best_valid_pgd_acc = ckpt_dict["best_valid_pgd_acc"]
            early_stopped_model = ckpt_dict["early_stopped_model"]
            if scheduler is not None:
                scheduler.load_state_dict(ckpt_dict["last_scheduler"])
            
            
    print(f"\nTraining \n{vars(args)}\n")
    stats = train(model, opt, train_dl, valid_dl, test_dl, max_epoch=args.epochs, 
                  adv_training=args.adv_training, mixed_training=args.mixed_training, eps=args.eps, alpha=args.alpha,
                  attack_iters=args.attack_iters, rs=args.attack_rs, clamp_val=(0,1), train_w_a_scheduler=args.train_w_a_scheduler,
                  eval_parameters=eval_parameters, scheduler=scheduler, reg_scalar=args.reg_scalar, ckpt_path=f"{ckpt_path}/ckpt.pt",
                  udp_sample_iters=args.udp_sample_iters, eval_freq=args.eval_freq, ckpt_freq=args.ckpt_freq, start_epoch=start_epoch,
                  best_valid_pgd_acc=best_valid_pgd_acc, early_stopped_model=early_stopped_model,
                  aa_test=args.test_with_autoattack)
    
    args.device = None
    stats['args'] = vars(args)
    
    with open(f"{ckpt_path}/summary.json", "w") as fs:
        json.dump(stats, fs)
        


if __name__ == "__main__":
    
    args = get_args()
    
    main(args)

    
    