# PGD
python ./src/main.py --adv_training pgd --eps 0.03 --attack_iters 10 --alpha 0.006 --seed 0 --epochs 60 --reg_scalar 1.0 --dataset cifar10-daug --lr 0.15 --model preactresnet18 --scheduler triangle --opt sgd --initialize_preactresnet --test-with-autoattack


# UDPR
python ./src/main.py --adv_training udp-reg --eps 0.05 --attack_iters 10 --alpha 0.01 --seed 0 --epochs 60 --reg_scalar 1.0 --dataset cifar10-daug --lr 0.15 --model preactresnet18 --scheduler triangle --opt sgd --udp_sample_iters uniform --initialize_preactresnet --test-with-autoattack


# UDP-PGD
python ./src/main.py --adv_training udp --eps 0.08 --attack_iters 20 --alpha 0.01 --seed 0 --epochs 60 --reg_scalar 1.0 --dataset cifar10-daug --lr 0.15 --model preactresnet18 --scheduler triangle --opt sgd --udp_sample_iters uniform --initialize_preactresnet --test-with-autoattack