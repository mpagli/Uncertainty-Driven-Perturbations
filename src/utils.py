import torch
import numpy as np
import torch.nn.functional as F
from autoattack import AutoAttack


def get_pgd_perturbation(model, X, y, eps, alpha, attack_iters, rs=False, clamp_val=None, use_alpha_scheduler=False):
  
  delta = torch.zeros_like(X).to(X.device)
    
  if use_alpha_scheduler:
    alpha_scheduler = lambda t: np.interp([t], [0, attack_iters // 2, attack_iters], [alpha, max(eps/5, alpha), alpha])[0]

  if rs:
    delta.uniform_(-eps, eps)

  delta.requires_grad = True
  for itr in range(attack_iters):

    if clamp_val is not None:
      X_ = torch.clamp(X + delta, clamp_val[0], clamp_val[1])
    else:
      X_ = X + delta

    output = model(X_)
    loss = F.cross_entropy(output, y)
    loss.backward()
    grad = delta.grad.detach()
    if use_alpha_scheduler:
      delta.data = delta + alpha_scheduler(itr+1) * grad.sign()
    else:
      delta.data = delta + alpha * grad.sign()
    if clamp_val is not None:
      delta.data = torch.clamp(X + delta.data, clamp_val[0], clamp_val[1]) - X
    delta.data = torch.clamp(delta.data, -eps, eps)
    delta.grad.zero_()

  return delta.detach()


def get_trades_perturbation(model, X, y, eps, alpha, attack_iters, rs=False, clamp_val=None, use_alpha_scheduler=False): 
  # y is not used, just here to keep the interface

  delta = torch.zeros_like(X).to(X.device)
    
  if use_alpha_scheduler:
    alpha_scheduler = lambda t: np.interp([t], [0, attack_iters // 2, attack_iters], [alpha, max(eps/5, alpha), alpha])[0]

  if rs:
    delta.uniform_(-eps, eps)

  with torch.no_grad():
    p = torch.softmax(model(X), dim=1)

  delta.requires_grad = True
  for itr in range(attack_iters):

    if clamp_val is not None:
      X_ = torch.clamp(X + delta, clamp_val[0], clamp_val[1])
    else:
      X_ = X + delta

    p_ = torch.softmax(model(X_), dim=1)
    loss = (-(p * (p_+1e-8).log()).sum(dim=1)).mean()
    loss.backward()
    grad = delta.grad.detach()
    if use_alpha_scheduler:
      delta.data = delta + alpha_scheduler(itr+1) * grad.sign()
    else:
      delta.data = delta + alpha * grad.sign()
    if clamp_val is not None:
      delta.data = torch.clamp(X + delta.data, clamp_val[0], clamp_val[1]) - X
    delta.data = torch.clamp(delta.data, -eps, eps)
    delta.grad.zero_()

  return delta.detach()


def get_udp_perturbation(model, X, y, eps, alpha, attack_iters, rs=False, clamp_val=None, use_alpha_scheduler=False, 
                         sample_iters='none'):
  # y is not used, just here to keep the interface
  
  delta = torch.zeros_like(X).to(X.device)
    
  if use_alpha_scheduler:
    alpha_scheduler = lambda t: np.interp([t], [0, attack_iters // 2, attack_iters], [alpha, max(eps/2, alpha), alpha])[0]

  if rs:
    delta.uniform_(-eps, eps)

  if sample_iters == 'uniform':
    shape = [delta.shape[0]] + [1] * (len(delta.shape)-1)
    sampled_iters = torch.randint(1,attack_iters+1,shape).expand_as(delta).to(X.device)

  delta.requires_grad = True

  for itr in range(attack_iters):

    if clamp_val is not None:
      X_ = torch.clamp(X + delta, clamp_val[0], clamp_val[1])
    else:
      X_ = X + delta

    output = model(X_)
    p = torch.softmax(output, dim=1)
    entropy = - (p * p.log()).sum(dim=1)
    entropy.mean().backward()
    grad = delta.grad.detach().sign()
    if sample_iters != 'none':
      grad[sampled_iters <= itr] = 0.0
    if use_alpha_scheduler:
      delta.data = delta + alpha_scheduler(itr+1) * grad
    else:
        delta.data = delta + alpha * grad
    if clamp_val is not None:
      delta.data = torch.clamp(X + delta.data, clamp_val[0], clamp_val[1]) - X
    delta.data = torch.clamp(delta.data, -eps, eps)
    delta.grad.zero_()

  return delta.detach()


@torch.no_grad()
def get_acc(model, dl):
  model.eval()
  acc = []
  for X, y in dl:
    #acc.append((torch.sigmoid(model(X)) > 0.5) == y)
    acc.append(torch.argmax(model(X), dim=1) == y)
  acc = torch.cat(acc)
  acc = torch.sum(acc)/len(acc)
  model.train()
  return acc.item()


def get_pgd_acc(model, dl, eps, alpha, attack_iters, rs=False, clamp_val=None, use_alpha_scheduler=False):
  model.eval()
  acc = []
  for X, y in dl:
    deltas = get_pgd_perturbation(model, X, y, eps, alpha, attack_iters, rs=rs, clamp_val=clamp_val, use_alpha_scheduler=use_alpha_scheduler) 
    acc.append(torch.argmax(model(X + deltas[0]), dim=1) == y)
  acc = torch.cat(acc)
  acc = torch.sum(acc)/len(acc)
  model.train()
  return acc.item()


def get_udp_acc(model, dl, eps, alpha, attack_iters, rs=False, clamp_val=None, use_alpha_scheduler=False):
  model.eval()
  acc = []
  for X, y in dl:
    deltas = get_udp_perturbation(model, X, y, eps, alpha, attack_iters, rs=rs, clamp_val=clamp_val, use_alpha_scheduler=use_alpha_scheduler) 
    acc.append(torch.argmax(model(X + deltas[0]), dim=1) == y)
  acc = torch.cat(acc)
  acc = torch.sum(acc)/len(acc)
  model.train()
  return acc.item()


def eval_with_autoattack(model, test_dl, eps, norm='Linf', bs=5000, max_data_size=10000):
  l = [x for (x, y) in test_dl]
  x_test = torch.cat(l, 0)[:max_data_size]
  l = [y for (x, y) in test_dl]
  y_test = torch.cat(l, 0)[:max_data_size]
  adversary = AutoAttack(model, norm=norm, eps=eps, version='standard', verbose=False)
      
  with torch.no_grad():
    x_adv, robust_dict, robust_acc = adversary.run_standard_evaluation(x_test, y_test, bs=bs)
    
  return robust_acc

