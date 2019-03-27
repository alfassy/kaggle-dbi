import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np

use_cuda = torch.cuda.is_available()


def get_feature_vecs(data_loader, model, test_mode=False):
    x_features_batch = []
    targets_batch = []
    for x, y in tqdm(iter(data_loader)):
        if use_cuda:
            x = x.cuda()
            y = y.cuda()
        x_features = model.features(x)
        x_features = model.avg_pool(x_features)
        x_features = x_features.view(x_features.size(0), -1)
        x_features_batch.append(x_features.clone().cpu().data.numpy())
        targets_batch.append(y.clone().cpu().data.numpy())
    feature_vecs_inputs = np.concatenate(x_features_batch)
    targets = np.concatenate(targets_batch)
    return feature_vecs_inputs, targets


class DropClassifier(nn.Module):
    def __init__(self):
        super(DropClassifier, self).__init__()
        # self.last_linear = nn.Linear(4032, 120, bias=True)
        self.last_linear = nn.Linear(4032, 120)

    def forward(self,x):
        x = nn.Dropout(p=0.5)(x)
        x = self.last_linear(x)
        return x


