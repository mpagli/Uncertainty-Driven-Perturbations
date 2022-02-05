# UDPR
python ./src/main.py --adv_training udp-reg --eps 0.05 --alpha 0.002 --seed 0 --epochs 60 --reg_scalar 1.0 --dataset svhn --lr 0.15 --model preactresnet18 --scheduler triangle --opt sgd --udp_sample_iters uniform --initialize_preactresnet --test-with-autoattack
python ./src/main.py --adv_training udp-reg --eps 0.05 --alpha 0.002 --seed 1 --epochs 60 --reg_scalar 1.0 --dataset svhn --lr 0.15 --model preactresnet18 --scheduler triangle --opt sgd --udp_sample_iters uniform --initialize_preactresnet --test-with-autoattack
python ./src/main.py --adv_training udp-reg --eps 0.05 --alpha 0.002 --seed 2 --epochs 60 --reg_scalar 1.0 --dataset svhn --lr 0.15 --model preactresnet18 --scheduler triangle --opt sgd --udp_sample_iters uniform --initialize_preactresnet --test-with-autoattack


# UDP-PGD
python ./src/main.py --adv_training udp --eps 0.04 --alpha 0.002 --seed 0 --epochs 60 --reg_scalar 1.0 --dataset svhn --lr 0.15 --model preactresnet18 --scheduler triangle --opt sgd --udp_sample_iters uniform --initialize_preactresnet --test-with-autoattack
python ./src/main.py --adv_training udp --eps 0.04 --alpha 0.002 --seed 1 --epochs 60 --reg_scalar 1.0 --dataset svhn --lr 0.15 --model preactresnet18 --scheduler triangle --opt sgd --udp_sample_iters uniform --initialize_preactresnet --test-with-autoattack
python ./src/main.py --adv_training udp --eps 0.04 --alpha 0.002 --seed 2 --epochs 60 --reg_scalar 1.0 --dataset svhn --lr 0.15 --model preactresnet18 --scheduler triangle --opt sgd --udp_sample_iters uniform --initialize_preactresnet --test-with-autoattack


# PGD
python ./src/main.py --adv_training pgd --eps 0.02 --alpha 0.002 --seed 0 --epochs 60 --reg_scalar 1.0 --dataset svhn --lr 0.15 --model preactresnet18 --scheduler triangle --opt sgd --initialize_preactresnet --test-with-autoattack
python ./src/main.py --adv_training pgd --eps 0.02 --alpha 0.002 --seed 1 --epochs 60 --reg_scalar 1.0 --dataset svhn --lr 0.15 --model preactresnet18 --scheduler triangle --opt sgd --initialize_preactresnet --test-with-autoattack
python ./src/main.py --adv_training pgd --eps 0.02 --alpha 0.002 --seed 2 --epochs 60 --reg_scalar 1.0 --dataset svhn --lr 0.15 --model preactresnet18 --scheduler triangle --opt sgd --initialize_preactresnet --test-with-autoattack

python ./src/main.py --adv_training pgd --eps 0.04 --alpha 0.002 --seed 0 --epochs 60 --reg_scalar 1.0 --dataset svhn --lr 0.15 --model preactresnet18 --scheduler triangle --opt sgd --initialize_preactresnet --test-with-autoattack
python ./src/main.py --adv_training pgd --eps 0.04 --alpha 0.002 --seed 1 --epochs 60 --reg_scalar 1.0 --dataset svhn --lr 0.15 --model preactresnet18 --scheduler triangle --opt sgd --initialize_preactresnet --test-with-autoattack
python ./src/main.py --adv_training pgd --eps 0.04 --alpha 0.002 --seed 2 --epochs 60 --reg_scalar 1.0 --dataset svhn --lr 0.15 --model preactresnet18 --scheduler triangle --opt sgd --initialize_preactresnet --test-with-autoattack


# TRADES
python ./src/main.py --adv_training trades --eps 0.05 --alpha 0.002 --seed 0 --epochs 60 --reg_scalar 1.0 --dataset svhn --lr 0.15 --model preactresnet18 --scheduler triangle --opt sgd --initialize_preactresnet --test-with-autoattack
python ./src/main.py --adv_training trades --eps 0.05 --alpha 0.002 --seed 1 --epochs 60 --reg_scalar 1.0 --dataset svhn --lr 0.15 --model preactresnet18 --scheduler triangle --opt sgd --initialize_preactresnet --test-with-autoattack
python ./src/main.py --adv_training trades --eps 0.05 --alpha 0.002 --seed 2 --epochs 60 --reg_scalar 1.0 --dataset svhn --lr 0.15 --model preactresnet18 --scheduler triangle --opt sgd --initialize_preactresnet --test-with-autoattack
