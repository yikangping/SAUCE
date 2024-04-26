import torch
import json
import numpy as np
import torch
import os, sys
import time
import datetime
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import csv
import pickle
import nflows.nn as nn_

from torch.nn.utils import clip_grad_norm_
from torch.utils import data
from tqdm import tqdm
from torchquad import enable_cuda, VEGAS
from time import sleep
from torch import optim
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
from nflows import transforms
from nflows import distributions
from nflows import utils
from nflows import flows
from torchquad import BatchMulVEGAS

sys.path.append("./")
from update_utils.end2end_utils import communicator
import FACE.FACE_utils.dataUtils as ut

""" Change it to the project root path """
PROJECT_PATH = "./FACE/"
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

GPU_ID = 1
dataset_name = "BJAQ"

""" network parameters """
hidden_features = 56
num_flow_steps = 6
flow_id = 1
features = 5
REUSE_FROM_FILE = False
REUSE_FILE_PATH = PROJECT_PATH + "train/"

""" query settings"""
query_seed = 45
QUERY_CNT = 2000

""" detailed network parameters"""
anneal_learning_rate = True
base_transform_type = "rq-coupling"
dropout_probability = 0
grad_norm_clip_value = 5.0
linear_transform_type = "lu"
num_bins = 8
num_training_steps = 400000
num_transform_blocks = 2
seed = 1638128
tail_bound = 3
use_batch_norm = False


class HiddenPrints:
    def __init__(self, activated=True):
        self.activated = activated
        self.original_stdout = None

    def open(self):
        """no output"""
        sys.stdout.close()
        sys.stdout = self.original_stdout

    def close(self):
        """output"""
        self.original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")

    def __enter__(self):
        if self.activated:
            self.close()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.activated:
            self.open()


def create_linear_transform():
    if linear_transform_type == "permutation":
        return transforms.RandomPermutation(features=features)
    elif linear_transform_type == "lu":
        return transforms.CompositeTransform(
            [
                transforms.RandomPermutation(features=features),
                transforms.LULinear(features, identity_init=True),
            ]
        )
    elif linear_transform_type == "svd":
        return transforms.CompositeTransform(
            [
                transforms.RandomPermutation(features=features),
                transforms.SVDLinear(features, num_householder=10, identity_init=True),
            ]
        )
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
            use_batch_norm=use_batch_norm,
        ),
        num_bins=num_bins,
        tails="linear",
        tail_bound=tail_bound,
        apply_unconditional_transform=True,
    )


# torch.masked_select()
def create_transform():
    transform = transforms.CompositeTransform(
        [
            transforms.CompositeTransform(
                [create_linear_transform(), create_base_transform(i)]
            )
            for i in range(num_flow_steps)
        ]
        + [create_linear_transform()]
    )
    return transform


""" q-error """


def ErrorMetric(est_card, card):
    if isinstance(est_card, torch.FloatTensor) or isinstance(est_card, torch.IntTensor):
        est_card = est_card.cpu().detach().numpy()
    if isinstance(est_card, torch.Tensor):
        est_card = est_card.cpu().detach().numpy()
    est_card = np.float32(est_card)
    card = np.float32(card)
    if card == 0 and est_card != 0:
        return est_card
    if card != 0 and est_card == 0:
        return card
    if card == 0 and est_card == 0:
        return 1.0
    return max(est_card / card, card / est_card)


def BatchErrorMetrix(est_list, oracle_list):
    ret = np.zeros(len(est_list))
    ID = 0
    for est, real in zip(est_list, oracle_list):
        ret[ID] = ErrorMetric(est, real)
        ID = ID + 1
    return ret


f_batch_time = 0


def f_batch(inp):
    global f_batch_time
    with torch.no_grad():
        inp = inp.cuda()

        print("【Example input】", inp[0, :])
        print("inp shape ", inp.shape)
        st = time.time()
        prob_list = flow.log_prob(inp)
        prob_list = torch.exp(prob_list)
        print("【max_prob】 ", prob_list.max())
        print("【median_prob】 ", prob_list.median())
        en = time.time()
        f_batch_time += en - st

        return prob_list


""" set GPU first """
os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(GPU_ID)
assert torch.cuda.is_available()
device = torch.device("cuda")
# device = torch.device('cpu')
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True
torch.manual_seed(seed)
np.random.seed(seed)

torch.set_default_tensor_type("torch.cuda.FloatTensor")

DEVICENAME = torch.cuda.get_device_name(0)
print("DEVICE NAME\n", DEVICENAME)


distribution = distributions.StandardNormal((features,))
transform = create_transform()
flow = flows.Flow(transform, distribution).to(device)
model_name = "BJAQ"

# if 'Ti' in DEVICENAME:
#     path = os.path.join(PROJECT_PATH+'train/models/{}'.format(dataset_name),
#                         '{}-id{}-best-val.t'.format(dataset_name, flow_id))

# else:
#     assert False

path = os.path.join(
    PROJECT_PATH + "train/models/{}".format(model_name),
    "{}-id{}-best-val.t".format(model_name, flow_id),
)

flow.load_state_dict(torch.load(path))

flow.cuda()
flow.eval()
n_params = utils.get_num_parameters(flow)
print("There are {} trainable parameters in this model.".format(n_params))
print("Parameters total size is {} MB".format(n_params * 4 / 1024 / 1024))

data, n, dim = ut.LoadTable(dataset_name)
DW = ut.DataWrapper(data, dataset_name)
rng = np.random.RandomState(query_seed)

rng = np.random.RandomState(query_seed)
queries = DW.generateNQuery(2000, rng)
DW.getAndSaveOracle(queries, query_seed)

""" Load oracle_cards"""
oracle_cards = ut.LoadOracleCardinalities(dataset_name, query_seed)

legal_lists = DW.getLegalRangeNQuery(queries)
legal_tensors = torch.Tensor(legal_lists).to("cuda")

from torchquad import set_up_backend

set_up_backend("torch", data_type="float32")


# f=open('/home/jiayi/disk/FACE/map.pickle','wb')
# pickle.dump(target_map, f)
if REUSE_FROM_FILE == True:
    f = open(REUSE_FILE_PATH + "{}.pickle".format(dataset_name), "rb")
    target_map = pickle.load(f)

z = DW.getLegalRangeQuery([[], [], []])
z = torch.Tensor(z)
print(z.shape)
full_integration_domain = torch.Tensor(z)

domain_starts = full_integration_domain[:, 0]
domain_sizes = full_integration_domain[:, 1] - domain_starts
domain_volume = torch.prod(domain_sizes)

if REUSE_FROM_FILE == False:
    vegas = VEGAS()
    bigN = 1000000 * 40

    st = time.time()
    result = vegas.integrate(
        f_batch,
        dim=features,
        N=bigN,
        integration_domain=full_integration_domain,
        use_warmup=True,
        use_grid_improve=True,
        max_iterations=40,
    )

    en = time.time()
    print("Took ", en - st)
    print(result)
    result = result * DW.n

    print("result is ", result)

if REUSE_FROM_FILE == False:
    target_map = vegas.map
    import pickle

    f = open(REUSE_FILE_PATH + "{}.pickle".format(dataset_name), "wb")
    pickle.dump(target_map, f)
    f.close()


def getResult(n, N, num_iterations=3, alpha=0.5, beta=0.5):
    global f_batch_time
    """ n: batch size """
    z = BatchMulVEGAS()
    DIM = features
    full_integration_domain = torch.Tensor(DIM * [[0, 1]])

    start_id = 0
    end_id = 0

    f_batch_time = 0
    st = time.time()
    results = []
    with torch.no_grad():
        while start_id < 2000:
            end_id = end_id + n
            if end_id > 2000:
                end_id = 2000
            z.setValues(
                f_batch,
                dim=DIM,
                alpha=alpha,
                beta=beta,
                N=N,
                n=end_id - start_id,
                iterations=num_iterations,
                integration_domains=legal_tensors[start_id:end_id],
                rng=None,
                seed=1234,
                reuse_sample_points=True,
                target_map=target_map,
                target_domain_starts=domain_starts,
                target_domain_sizes=domain_sizes,
            )
            start_id = start_id + n
            results.append(z.integrate())

    en = time.time()
    total_time = en - st
    return total_time, results


def testHyper(n, N, num_iterations, alpha, beta):
    with HiddenPrints():
        total_time, result = getResult(
            n=n, N=N, num_iterations=num_iterations, alpha=alpha, beta=beta
        )

        result = torch.cat(tuple(result))
        FULL_SIZE = torch.Tensor([DW.n])
        result = result * FULL_SIZE
        result = result.to("cpu")

        n_ = 2000
        oracle_list = oracle_cards.copy()

        err_list = BatchErrorMetrix(result.int(), oracle_list)

        total_query_time = total_time
        avg_per_query_time = 1000.0 * (total_query_time / n_)
        avg_f_batch_time = 1000.0 * f_batch_time / n_
        avg_vegas_time = avg_per_query_time - avg_f_batch_time

    print(
        "********** total_n=[{}] batchn=[{}]  N=[{}]  nitr=[{}]  alpha=[{}]  beta=[{}] ******".format(
            n_, n, N, num_iterations, alpha, beta
        )
    )
    print("@ Average per query          [{}] ms".format(avg_per_query_time))
    print(" --  Average per query NF    [{}] ms".format(avg_f_batch_time))
    print(" --  Average per query vegas [{}] ms".format(avg_vegas_time))
    p50 = np.percentile(err_list, 50)
    mean = np.mean(err_list)
    p95 = np.percentile(err_list, 95)
    p99 = np.percentile(err_list, 99)
    pmax = np.max(err_list)
    print(
        "Mean [{:.3f}]  Median [{:.3f}]  95th [{:.3f}]  99th [{:.3f}]  max [{:.3f}]".format(
            mean, p50, p95, p99, pmax
        )
    )

    return p50, p95, p99, pmax


alpha_list = [0.4]
beta_list = [0.2]

p50s = []
p95s = []
p99s = []
pmaxs = []

for alpha in alpha_list:
    for beta in beta_list:
        p50, p95, p99, pmax = testHyper(1000, 16000, 4, alpha, beta)
        p50s.append(p50)
        p95s.append(p95)
        p99s.append(p99s)
        pmaxs.append(pmax)
