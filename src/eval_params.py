def get_eval_params(dataset):
    if dataset in ['mnist', 'fashion-mnist']:
        return {
            'eps': [0.05, 0.1, 0.2],
            'max_data_size': 10000,
            'batch_size_eval': 5000,
            'l2_eps': [1.5]
        }
    elif dataset in ['svhn', 'cifar10', 'cifar10-daug']:
        return {
            'eps': [0.01, 0.03],
            'batch_size_eval': 4000,
            'max_data_size': 10000,
            'l2_eps': [0.5, 1]
        }
    elif dataset in ['cifar10-ldr']:
        return {
            'eps': [0.01, 0.03],
            'batch_size_eval': 4000,
            'max_data_size': 10000,
            'l2_eps': [0.5, 1]
        }
    else:
        raise KeyError(f"Unknown/missing evaluation parameters for dataset '{dataset}'.")