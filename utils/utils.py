import torch
import random
import numpy as np
import matplotlib.pyplot as plt

def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
        return True
    raise ValueError(f'{value} is not a valid boolean value')

def setting_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def plot_loss(loss_pool, dataset):
    plt.plot(np.arange(len(loss_pool)).tolist(), loss_pool)
    np.save("loss_curve/train_loss_{}.npy".format(dataset), loss_pool)
    plt.show()

