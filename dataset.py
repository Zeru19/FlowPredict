import os
import math
import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

from utils import next_batch


class Dataset:
    """Generic Flow Dataset"""

    def __init__(self, name, 
                 lag=24 * 7, horizon=24,
                 val_ratio=0.2, test_ratio=0.2):
        self.name = name
        self.lag = lag
        self.horizon = horizon
        
        self.raw_path = '/mnt/nfsData10/ZhouZeyu/datasets/'
        self.src_data = np.load(os.path.join(self.raw_path, name + '.npz'), 
                                allow_pickle=True)['data']
        self.num_nodes = self.src_data.shape[1]

        self.train_x, self.train_y, self.val_x, self.val_y, self.test_x, self.test_y, self.normalizer, self.denormalizer = \
            self.construct_samples(self.src_data[:, :, 0], lag, horizon, val_ratio, test_ratio)
        
    def get_samples(self, type_: str, mask_pre=False, **kwargs):
        assert type_ in ["train", "val", "test"], "type_ must be train, val or test"

        if mask_pre:
            return self._get_maskpre_samples(type_, **kwargs)
        else:
            return self._get_samples(type_, **kwargs)

    def _get_samples(self, type_: str, **kwargs):
        device = kwargs.get('device', 'cpu')
        x = torch.tensor(getattr(self, type_ + '_x'), device=device).float()
        y = torch.tensor(getattr(self, type_ + '_y'), device=device).float()

        return x, y, None
    
    def _get_maskpre_samples(self, type_: str, **kwargs):
        device = kwargs.get('device', 'cpu')

        x = torch.tensor(getattr(self, type_ + '_x'), device=device).float()
        y = torch.tensor(getattr(self, type_ + '_y'), device=device).float()

        mask = torch.concat(
            [torch.zeros(*x.shape[:-1], device=device), torch.ones(*y.shape[:-1], device=device)],
            dim=1
        ).int()
        mask_tensor = torch.zeros(*y.shape, device=device).float()
        x = torch.concat([x, mask_tensor], dim=1)

        return x, y, mask
    
    def get_normalizer(self, **kwargs):
        device = kwargs.get('device', 'cpu')

        return self.normalizer, self.denormalizer

    def __str__(self):
        return "Dataset({})".format(self.name)
    
    @staticmethod
    def construct_samples(data, lag, horizon, val_ratio, test_ratio):
        flow_train, flow_val, flow_test = split_data_by_ratio(data, val_ratio, test_ratio)

        train_x, train_y = add_window_horizon(flow_train, lag, horizon)
        val_x, val_y = add_window_horizon(flow_val, lag, horizon)
        test_x, test_y = add_window_horizon(flow_test, lag, horizon)

        mean_ = flow_train.mean()
        std_ = flow_train.std()

        return train_x, train_y, val_x, val_y, test_x, test_y, Normalizer(mean_, std_), Denormalizer(mean_, std_)
        

def split_data_by_ratio(data, val_ratio, test_ratio):
    data_len = data.shape[0]
    test_data = data[-int(data_len * test_ratio):]
    val_data = data[-int(data_len * (test_ratio + val_ratio)):-int(data_len * test_ratio)]
    train_data = data[:-int(data_len * (test_ratio + val_ratio))]
    return train_data, val_data, test_data


class Normalizer(nn.Module):
    """"Standard normalization"""

    def __init__(self, mean, std):
        super().__init__()
        self.mean = mean
        self.std = std

    def forward(self, data):
        return (data - self.mean) / self.std


class Denormalizer(nn.Module):
    """Inverse Standard normalization"""

    def __init__(self, mean, std):
        super().__init__()
        self.mean = mean
        self.std = std

    def forward(self, data):
        return (data * self.std) + self.mean


def add_window_horizon(data, window=3, horizon=1, single=False):
    '''
    :param data: shape [B, ...]
    :param window:
    :param horizon:
    :return: X is [B, W, ...], Y is [B, H, ...]
    '''
    length = len(data)
    end_index = length - horizon - window + 1
    X = []      #windows
    Y = []      #horizon
    index = 0
    if single:
        while index < end_index:
            # X.append(np.concatenate(data[index - 24], data[index:index+window]))  # index = 24
            X.append(data[index: index + window])
            Y.append(data[index+window+horizon-1:index+window+horizon])
            index = index + 1
    else:
        while index < end_index:
            # X.append(np.concatenate([data[index - 24: index - 23], data[index:index+window]], axis=0))
            X.append(data[index:index+window])
            Y.append(data[index+window:index+window+horizon])
            index = index + 1
    X = np.array(X)
    Y = np.array(Y)
    return X, Y


if __name__ == '__main__':
    data = Dataset('pems08')
    batchiter = data.get_batchiter('train', batch_size=64, mask_pre=True)
    for x, y, m in batchiter:
        print(x.shape, y.shape)
        break
