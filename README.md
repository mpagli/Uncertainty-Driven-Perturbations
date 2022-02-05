# Uncertainty-Driven-Perturbations

Repository for the paper: "Increasing the Classification Margin with Uncertainty Driven Perturbations"

## Usage

```
usage: main.py [-h] [--batch_size_train BATCH_SIZE_TRAIN] [--batch_size_eval BATCH_SIZE_EVAL] [--seed SEED] [--device DEVICE] [--epochs EPOCHS] [--lr LR] [--scheduler {triangle,multistep,none}] [--opt {adam,sgd}] [--eval_freq EVAL_FREQ]
               [--ckpt_freq CKPT_FREQ] [--results_base_folder RESULTS_BASE_FOLDER] [--initialize_preactresnet] [--test-with-autoattack] [--adv_training {pgd,udp,udp-reg,trades,udp+trades,none}] [--mixed_training] [--eps EPS]
               [--alpha ALPHA] [--reg_scalar REG_SCALAR] [--attack_iters ATTACK_ITERS] [--attack_rs] [--udp_sample_iters {uniform,none}] [--train_w_a_scheduler] [--model {lenet,mlp,preactresnet18}]
               [--dataset {mnist,fashion-mnist,svhn,cifar10,cifar10-ldr,cifar10-daug}]

optional arguments:
  -h, --help            show this help message and exit
  --batch_size_train BATCH_SIZE_TRAIN
  --batch_size_eval BATCH_SIZE_EVAL
  --seed SEED
  --device DEVICE
  --epochs EPOCHS
  --lr LR
  --scheduler {triangle,multistep,none}
  --opt {adam,sgd}
  --eval_freq EVAL_FREQ
  --ckpt_freq CKPT_FREQ
  --results_base_folder RESULTS_BASE_FOLDER
  --initialize_preactresnet
  --test-with-autoattack
  --adv_training {pgd,udp,udp-reg,trades,udp+trades,none}
  --mixed_training
  --eps EPS
  --alpha ALPHA
  --reg_scalar REG_SCALAR
  --attack_iters ATTACK_ITERS
  --attack_rs
  --udp_sample_iters {uniform,none}
  --train_w_a_scheduler
  --model {lenet,mlp,preactresnet18}
  --dataset {mnist,fashion-mnist,svhn,cifar10,cifar10-ldr,cifar10-daug}
```

## Requirements

AutoAttack needs to be installed:

```
git clone https://github.com/fra31/auto-attack.git 
```

To facilitate the integration with the AutoAttack package, we modified the `run_standard_evaluation` method of the `AutoAttack` class to return the robust accuracy (line 222 in `autoattack/autoattack.py`):

```python
return x_adv
```

was changed for 

```python
return x_adv, robust_accuracy_dict, robust_accuracy
```

To run the AutoAttack evaluation, the `--test-with-autoattack` flag needs to be used.

## Reproducing our experiments  

The scripts in the `scripts` folder can be used to reproduce our results. 
