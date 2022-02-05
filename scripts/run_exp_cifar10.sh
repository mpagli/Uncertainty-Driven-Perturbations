# UDPR
python ./src/main.py --adv_training udp-reg --eps 0.05 --alpha 0.002 --seed 0 --epochs 60 --reg_scalar 1.0 --dataset cifar10 --lr 0.14 --model preactresnet18 --scheduler triangle --opt sgd --udp_sample_iters uniform --test-with-autoattack
python ./src/main.py --adv_training udp-reg --eps 0.05 --alpha 0.002 --seed 1 --epochs 60 --reg_scalar 1.0 --dataset cifar10 --lr 0.14 --model preactresnet18 --scheduler triangle --opt sgd --udp_sample_iters uniform --test-with-autoattack
python ./src/main.py --adv_training udp-reg --eps 0.05 --alpha 0.002 --seed 2 --epochs 60 --reg_scalar 1.0 --dataset cifar10 --lr 0.14 --model preactresnet18 --scheduler triangle --opt sgd --udp_sample_iters uniform --test-with-autoattack


# UDP-PGD
python ./src/main.py --adv_training udp --eps 0.06 --alpha 0.002 --seed 0 --epochs 60 --reg_scalar 1.0 --dataset cifar10 --lr 0.15 --model preactresnet18 --scheduler triangle --opt sgd --udp_sample_iters uniform --test-with-autoattack
python ./src/main.py --adv_training udp --eps 0.06 --alpha 0.002 --seed 1 --epochs 60 --reg_scalar 1.0 --dataset cifar10 --lr 0.15 --model preactresnet18 --scheduler triangle --opt sgd --udp_sample_iters uniform --test-with-autoattack
python ./src/main.py --adv_training udp --eps 0.06 --alpha 0.002 --seed 2 --epochs 60 --reg_scalar 1.0 --dataset cifar10 --lr 0.15 --model preactresnet18 --scheduler triangle --opt sgd --udp_sample_iters uniform --test-with-autoattack

python ./src/main.py --adv_training udp --eps 0.04 --alpha 0.001 --seed 0 --epochs 60 --reg_scalar 1.0 --dataset cifar10 --lr 0.15 --model preactresnet18 --scheduler triangle --opt sgd --udp_sample_iters uniform --test-with-autoattack
python ./src/main.py --adv_training udp --eps 0.04 --alpha 0.001 --seed 1 --epochs 60 --reg_scalar 1.0 --dataset cifar10 --lr 0.15 --model preactresnet18 --scheduler triangle --opt sgd --udp_sample_iters uniform --test-with-autoattack
python ./src/main.py --adv_training udp --eps 0.04 --alpha 0.001 --seed 2 --epochs 60 --reg_scalar 1.0 --dataset cifar10 --lr 0.15 --model preactresnet18 --scheduler triangle --opt sgd --udp_sample_iters uniform --test-with-autoattack


# PGD
python ./src/main.py --adv_training pgd --eps 0.02 --alpha 0.001 --seed 0 --epochs 60 --reg_scalar 1.0 --dataset cifar10 --lr 0.15 --model preactresnet18 --scheduler triangle --opt sgd --test-with-autoattack
python ./src/main.py --adv_training pgd --eps 0.02 --alpha 0.001 --seed 1 --epochs 60 --reg_scalar 1.0 --dataset cifar10 --lr 0.15 --model preactresnet18 --scheduler triangle --opt sgd --test-with-autoattack
python ./src/main.py --adv_training pgd --eps 0.02 --alpha 0.001 --seed 2 --epochs 60 --reg_scalar 1.0 --dataset cifar10 --lr 0.15 --model preactresnet18 --scheduler triangle --opt sgd --test-with-autoattack

python ./src/main.py --adv_training pgd --eps 0.01 --alpha 0.001 --seed 0 --epochs 60 --reg_scalar 1.0 --dataset cifar10 --lr 0.15 --model preactresnet18 --scheduler triangle --opt sgd --test-with-autoattack
python ./src/main.py --adv_training pgd --eps 0.01 --alpha 0.001 --seed 1 --epochs 60 --reg_scalar 1.0 --dataset cifar10 --lr 0.15 --model preactresnet18 --scheduler triangle --opt sgd --test-with-autoattack
python ./src/main.py --adv_training pgd --eps 0.01 --alpha 0.001 --seed 2 --epochs 60 --reg_scalar 1.0 --dataset cifar10 --lr 0.15 --model preactresnet18 --scheduler triangle --opt sgd --test-with-autoattack


# TRADES
python ./src/main.py --adv_training trades --eps 0.03 --alpha 0.001 --seed 0 --epochs 60 --reg_scalar 1.0 --dataset cifar10 --lr 0.15 --model preactresnet18 --scheduler triangle --opt sgd --test-with-autoattack
python ./src/main.py --adv_training trades --eps 0.03 --alpha 0.001 --seed 1 --epochs 60 --reg_scalar 1.0 --dataset cifar10 --lr 0.15 --model preactresnet18 --scheduler triangle --opt sgd --test-with-autoattack
python ./src/main.py --adv_training trades --eps 0.03 --alpha 0.001 --seed 2 --epochs 60 --reg_scalar 1.0 --dataset cifar10 --lr 0.15 --model preactresnet18 --scheduler triangle --opt sgd --test-with-autoattack

python ./src/main.py --adv_training trades --eps 0.02 --alpha 0.001 --seed 0 --epochs 60 --reg_scalar 1.0 --dataset cifar10 --lr 0.15 --model preactresnet18 --scheduler triangle --opt sgd --test-with-autoattack
python ./src/main.py --adv_training trades --eps 0.02 --alpha 0.001 --seed 1 --epochs 60 --reg_scalar 1.0 --dataset cifar10 --lr 0.15 --model preactresnet18 --scheduler triangle --opt sgd --test-with-autoattack
python ./src/main.py --adv_training trades --eps 0.02 --alpha 0.001 --seed 2 --epochs 60 --reg_scalar 1.0 --dataset cifar10 --lr 0.15 --model preactresnet18 --scheduler triangle --opt sgd --test-with-autoattack
