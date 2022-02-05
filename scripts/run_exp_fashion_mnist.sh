# UDPR
python ./src/main.py --adv_training udp-reg --eps 0.5 --alpha 0.025 --attack_iters 20 --seed 0 --epochs 100 --reg_scalar 1.0 --dataset fashion-mnist --lr 0.001 --udp_sample_iters uniform --model lenet --opt adam --test-with-autoattack
python ./src/main.py --adv_training udp-reg --eps 0.5 --alpha 0.025 --attack_iters 20 --seed 1 --epochs 100 --reg_scalar 1.0 --dataset fashion-mnist --lr 0.001 --udp_sample_iters uniform --model lenet --opt adam --test-with-autoattack
python ./src/main.py --adv_training udp-reg --eps 0.5 --alpha 0.025 --attack_iters 20 --seed 2 --epochs 100 --reg_scalar 1.0 --dataset fashion-mnist --lr 0.001 --udp_sample_iters uniform --model lenet --opt adam --test-with-autoattack


# UDP-PGD
python ./src/main.py --adv_training udp --eps 0.5 --alpha 0.025 --attack_iters 20 --seed 0 --epochs 100 --reg_scalar 1.0 --dataset fashion-mnist --lr 0.001 --udp_sample_iters uniform --model lenet --opt adam --test-with-autoattack
python ./src/main.py --adv_training udp --eps 0.5 --alpha 0.025 --attack_iters 20 --seed 1 --epochs 100 --reg_scalar 1.0 --dataset fashion-mnist --lr 0.001 --udp_sample_iters uniform --model lenet --opt adam --test-with-autoattack
python ./src/main.py --adv_training udp --eps 0.5 --alpha 0.025 --attack_iters 20 --seed 2 --epochs 100 --reg_scalar 1.0 --dataset fashion-mnist --lr 0.001 --udp_sample_iters uniform --model lenet --opt adam --test-with-autoattack

python ./src/main.py --adv_training udp --eps 0.3 --alpha 0.015 --attack_iters 25 --seed 0 --epochs 100 --reg_scalar 1.0 --dataset fashion-mnist --lr 0.001 --udp_sample_iters uniform --model lenet --opt adam --test-with-autoattack
python ./src/main.py --adv_training udp --eps 0.3 --alpha 0.015 --attack_iters 25 --seed 1 --epochs 100 --reg_scalar 1.0 --dataset fashion-mnist --lr 0.001 --udp_sample_iters uniform --model lenet --opt adam --test-with-autoattack
python ./src/main.py --adv_training udp --eps 0.3 --alpha 0.015 --attack_iters 25 --seed 2 --epochs 100 --reg_scalar 1.0 --dataset fashion-mnist --lr 0.001 --udp_sample_iters uniform --model lenet --opt adam --test-with-autoattack


# PGD
python ./src/main.py --adv_training pgd --eps 0.2 --alpha 0.015 --attack_iters 20 --seed 0 --epochs 100 --reg_scalar 1.0 --dataset fashion-mnist --lr 0.001 --model lenet --opt adam --test-with-autoattack
python ./src/main.py --adv_training pgd --eps 0.2 --alpha 0.015 --attack_iters 20 --seed 1 --epochs 100 --reg_scalar 1.0 --dataset fashion-mnist --lr 0.001 --model lenet --opt adam --test-with-autoattack
python ./src/main.py --adv_training pgd --eps 0.2 --alpha 0.015 --attack_iters 20 --seed 2 --epochs 100 --reg_scalar 1.0 --dataset fashion-mnist --lr 0.001 --model lenet --opt adam --test-with-autoattack


# TRADES
python ./src/main.py --adv_training trades --eps 0.2 --alpha 0.015 --attack_iters 20 --seed 0 --epochs 100 --reg_scalar 1.0 --dataset fashion-mnist --lr 0.001 --model lenet --opt adam --test-with-autoattack
python ./src/main.py --adv_training trades --eps 0.2 --alpha 0.015 --attack_iters 20 --seed 1 --epochs 100 --reg_scalar 1.0 --dataset fashion-mnist --lr 0.001 --model lenet --opt adam --test-with-autoattack
python ./src/main.py --adv_training trades --eps 0.2 --alpha 0.015 --attack_iters 20 --seed 1 --epochs 100 --reg_scalar 1.0 --dataset fashion-mnist --lr 0.001 --model lenet --opt adam --test-with-autoattack
