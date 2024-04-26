#!/usr/bin/env python
# coding: utf-8

# In[1]:


""" Change it to the project root path """
PROJECT_PATH = './FACE/'


# # Model Parameter

# In[2]:


GPU_ID = 0
dataset_name = 'power'
ID = 4

ckpts_PATH = PROJECT_PATH + 'train/models/{}/'.format(dataset_name)
data_PATH  = PROJECT_PATH + 'data/'

""" network parameters"""
hidden_features = 108
num_flow_steps = 6
train_batch_size = 512
learning_rate = 0.0005
monitor_interval = 5000


anneal_learning_rate = True
base_transform_type = 'rq-coupling'

dropout_probability = 0
grad_norm_clip_value = 5.
linear_transform_type='lu'

num_bins = 8
num_training_steps = 400000
num_transform_blocks = 2
seed = 1638128
tail_bound = 3
use_batch_norm = False

val_batch_size = 262144


# # Load Data

# In[3]:


import prefetcher as pf

import argparse
import json
import numpy as np
import torch
import os
import time
import datetime
# from tensorboardX import SummaryWriter
from time import sleep
from torch import optim
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils import data
from tqdm import tqdm

from matplotlib import pyplot as plt
from torch.utils.data import Dataset

from nflows import transforms
from nflows import distributions
from nflows import utils
from nflows import flows
import nflows.nn as nn_


os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(GPU_ID)
assert torch.cuda.is_available()
device = torch.device('cuda')



torch.manual_seed(seed)
np.random.seed(seed)
torch.set_default_tensor_type('torch.cuda.FloatTensor')




class  PowerDataset(Dataset):
    def __init__(self, split='train', frac=None):
        path = os.path.join(data_PATH, '{}.npy'.format(dataset_name))
        self.data = np.load(path).astype(np.float32)
        print('data shape:', self.data.shape)

        self.n, self.dim = self.data.shape
        if frac is not None:
            self.n = int(frac * self.n)

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return self.n


# In[4]:


st_time = time.time()
train_dataset = PowerDataset()
train_loader = data.DataLoader(
    train_dataset,
    batch_size = train_batch_size,
    shuffle=False,
    drop_last=False
)

val_dataset = PowerDataset()
val_loader = data.DataLoader(
    dataset=val_dataset,
    batch_size=val_batch_size,
    shuffle=False,
    drop_last=False
)

train_loader = list(train_loader)
val_loader = list(val_loader)
TRAIN_LOADER_LEN = len(train_loader)


features = train_dataset.dim

print('Load data took [{}] s'.format(time.time() - st_time))
print('train loader length is [{}]'.format(TRAIN_LOADER_LEN))
print('val loader length is [{}]'.format(len(val_loader)))


# # Build Model

# In[5]:


def get_timestamp():
    formatted_time = time.strftime('%d-%b-%y||%H:%M:%S')
    return formatted_time
timestamp = get_timestamp()
print('Timestamp is ', timestamp)


# In[6]:


def create_linear_transform():
    if linear_transform_type == 'permutation':
        return transforms.RandomPermutation(features=features)
    elif linear_transform_type == 'lu':
        return transforms.CompositeTransform([
            transforms.RandomPermutation(features=features),
            transforms.LULinear(features, identity_init=True)
        ])
    elif linear_transform_type == 'svd':
        return transforms.CompositeTransform([
            transforms.RandomPermutation(features=features),
            transforms.SVDLinear(features, num_householder=10, identity_init=True)
        ])
    else:
        raise ValueError




def create_base_transform(i):
    # tmp_mask = utils.create_alternating_binary_mask(features, even=(i % 2 == 0))
    return transforms.coupling.PiecewiseRationalQuadraticCouplingTransform(
        mask=utils.create_alternating_binary_mask(features, even=(i % 2 == 0)),
        transform_net_create_fn=lambda in_features, out_features: nn_.nets.ResidualNet(
            in_features=in_features,
            out_features=out_features,
            hidden_features=hidden_features,
            context_features=None,
            num_blocks=num_transform_blocks,
            activation=F.relu,
            dropout_probability=dropout_probability,
            use_batch_norm=use_batch_norm
        ),
        num_bins=num_bins,
        tails='linear',
        tail_bound=tail_bound,
        apply_unconditional_transform=True
    )


# torch.masked_select()
def create_transform():
    transform = transforms.CompositeTransform([
        transforms.CompositeTransform([
            create_linear_transform(),
            create_base_transform(i)
        ]) for i in range(num_flow_steps)
    ] + [
        create_linear_transform()
    ])
    return transform


# In[7]:


distribution = distributions.StandardNormal((features,))
transform = create_transform()
flow = flows.Flow(transform, distribution).to(device)


# In[8]:


n_params = utils.get_num_parameters(flow)
print('There are {} trainable parameters in this model.'.format(n_params))
print('Parameters total size is {} MB'.format(n_params * 4 / 1024 / 1024))

optimizer = optim.Adam(flow.parameters(), lr=learning_rate)
if anneal_learning_rate:
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, np.ceil(num_training_steps / TRAIN_LOADER_LEN) , 0)


# # Model Training

# In[9]:


best_val_score = -1e10
prefetcher = pf.data_prefetcher(train_loader)

num_training_steps = int(np.ceil(num_training_steps/TRAIN_LOADER_LEN) * TRAIN_LOADER_LEN)

print('num training steps is ', num_training_steps)
for step in range(num_training_steps):
    if step % 10000 == 0:
        print('[{}] {}/400000  {}% has finished!'.format(datetime.datetime.now(), step, 100.*step/400000))

    flow.train()
    optimizer.zero_grad()

    batch = prefetcher.next()
    if batch is None:
        prefetcher = pf.data_prefetcher(train_loader)
        batch = prefetcher.next()

    log_density = flow.log_prob(batch)
    loss = - torch.mean(log_density)
    loss.backward()
    if grad_norm_clip_value is not None:
        clip_grad_norm_(flow.parameters(), grad_norm_clip_value)
    optimizer.step()

    if (step + 1) % monitor_interval == 0:
        flow.eval()
        val_prefetcher = pf.data_prefetcher(val_loader)
        with torch.no_grad():
            running_val_log_density = 0
            while True:
                val_batch = val_prefetcher.next()
                if val_batch is None:
                    break

                log_density_val = flow.log_prob(val_batch.to(device).detach())
                mean_log_density_val = torch.mean(log_density_val).detach()
                running_val_log_density += mean_log_density_val
            running_val_log_density /= len(val_loader)
            print('[{}] step now is [{:6d}] running_val_log_density is {:.4f}'.format(datetime.datetime.now(), step, running_val_log_density), end='')

        if running_val_log_density > best_val_score:
            best_val_score = running_val_log_density
            print('  ## New best! ##')
            path = os.path.join(ckpts_PATH,
                                '{}-id{}-best-val.t'.format(dataset_name, ID))
            torch.save(flow.state_dict(), path)
        else:
            print('')
    

    if (step + 1) % 20000 == 0 :
        flow.eval()
        print('[{}] save once. Step is {} best val score is {}'.format(datetime.datetime.now(), step, best_val_score))


        path = os.path.join(ckpts_PATH,
                            '{}-id{}-step-{}.t'.format(dataset_name, ID, step + 1))
        torch.save(flow.state_dict(), path)
        
    if anneal_learning_rate and (step + 1) % TRAIN_LOADER_LEN == 0:
        scheduler.step()


# In[ ]:





# In[ ]:




