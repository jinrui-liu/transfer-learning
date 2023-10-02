import torch
import pandas as pd

import pyro
import pyro.distributions as dist
import pyro.distributions.constraints as constraints

import helpers

ALPHA = .01
BETA = .01


# Read in model inputs
def get_model_inputs(train_fn, sample_fn, drug_fn):
    df = pd.read_pickle(train_fn)
    sample_dict = helpers.read_pickle(sample_fn)
    drug_dict = helpers.read_pickle(drug_fn)
    n_samp = len(sample_dict.keys())
    n_drug = len(drug_dict.keys())
    s_idx = df['s_idx'].to_numpy()
    d_idx = df['d_idx'].to_numpy()
    obs = torch.Tensor(df['log(V_V0)'])
    return n_samp, n_drug, s_idx, d_idx, obs


# A note on how models read in data: Our mental model of the dataset is often as a matrix: rows correspond to
# samples, columns correspond to drugs, and the entry (i, j) corresponds to sample i and drug j. However,
# we don't pass in the data as a matrix. Instead, we pass in a list of sample_ids "s_idx", a list of drug_ids
# "d_idx", and a list of observations "obs". Then obs[i] corresponds to sample s_idx[i] and drug d_idx[i].

# The transfer model trains on both source and target data. We call the target data used for training "target_train"
# to distinguish from the held-out target data.

# n_samp: number of samples total 
# n_drug: number of drugs total
# s_idx1: list of sample_ids for source data
# d_idx1: list of drug_ids for source date
# obs1: observations for source date
# s_idx2: list of sample_ids for target_train 
# d_idx2: list of drug_ids for target_train
# obs2: observations for target_train
# k: embedding dimension
# r: rank r of the embeddings, used to construct the W matrix, r should be <= k
def transfer_model(n_samp, n_drug, s_idx1, d_idx1, s_idx2, d_idx2, obs1=None, n_obs1=None, obs2=None, n_obs2=None, k=1,
                   r=1):
    if obs1 is None and n_obs1 is None:
        print('Error!: both obs1 and n_obs1 are None.')
    if obs1 is not None:
        n_obs1 = obs1.shape[0]
    if obs2 is None and n_obs2 is None:
        print('Error: both obs2 and n_obs2 are None')
    if obs2 is not None:
        n_obs2 = obs2.shape[0]
    # create global offset
    a1_sigma = pyro.sample('a1_sigma', dist.Gamma(ALPHA, BETA))
    a1 = pyro.sample('a1', dist.Normal(0, a1_sigma))
    # create s
    s_sigma = pyro.sample('s_sigma', dist.Gamma(ALPHA, BETA))
    with pyro.plate('s_plate', n_samp):
        with pyro.plate('k_s', k):
            s = pyro.sample('s', dist.Normal(0, s_sigma))
    s = torch.transpose(s, 0, 1)
    # create d
    d_sigma = pyro.sample('d_sigma', dist.Gamma(ALPHA, BETA))
    with pyro.plate('d_plate', n_drug):
        with pyro.plate('k_d', k):
            d = pyro.sample('d', dist.Normal(0, d_sigma))
    # given s and d, compute the matrix of means used to make source estimates
    # mat1 = sd
    mat1 = torch.matmul(s, d)
    assert (mat1.shape[0] == n_samp) and (mat1.shape[1] == n_drug)
    # index into the matrix to get the estimates for each source sample, drug pairs
    mean1 = mat1[s_idx1, d_idx1] + a1
    sigma1 = pyro.sample('sigma1', dist.Gamma(ALPHA, BETA))
    with pyro.plate('data1_plate', n_obs1):
        pyro.sample('data1', dist.Normal(mean1, sigma1 * torch.ones(n_obs1)), obs=obs1)
    # create global offset
    a2_sigma = pyro.sample('a2_sigma', dist.Gamma(ALPHA, BETA))
    a2 = pyro.sample('a2', dist.Normal(0, a2_sigma))
    # create s' (s prime)
    # create w matrix: w = d + j, where d is diagonal and j is a product to two low rank matrices
    w_sigma = pyro.sample('w_sigma', dist.Gamma(ALPHA, BETA))
    # create a vector for diagonal matrix
    with pyro.plate('c_row', k):
        c = pyro.sample('c', dist.Normal(0, w_sigma))
    # create j matrix k x k
    j = torch.zeros(k, k)
    us = []
    vs = []
    ranks = [i + 1 for i in list(range(r))]
    for i in ranks:
        with pyro.plate(f'u_col_{i}', i):
            with pyro.plate(f'u_row_{i}', k):
                us.append(pyro.sample(f'u_{i}', dist.Normal(0, w_sigma)))
        with pyro.plate(f'v_col_{i}', k):
            with pyro.plate(f'v_row_{i}', i):
                vs.append(pyro.sample(f'v_{i}', dist.Normal(0, w_sigma)))
    for u, v in zip(us, vs):
        j += torch.matmul(u, v)
    w = c * torch.diag(torch.ones(k)) + j
    assert (w.shape[0] == k) and (w.shape[1] == k)
    # compute s': s'^T = ws^T
    spr_transpose = torch.matmul(w, torch.transpose(s, 0, 1))
    spr = torch.transpose(spr_transpose, 0, 1)
    assert (spr.shape[0] == n_samp) and (spr.shape[1] == k)
    # given s' and d, compute the matrix of means used to make target estimates
    mat2 = torch.matmul(spr, d)
    assert (mat2.shape[0] == n_samp) and (mat2.shape[1] == n_drug)
    # index into the matrix to get the estimates for target sample, drug pairs
    mean2 = mat2[s_idx2, d_idx2] + a2
    sigma2 = pyro.sample('sigma2', dist.Gamma(ALPHA, BETA))
    with pyro.plate('data2_plate', n_obs2):
        pyro.sample('data2', dist.Normal(mean2, sigma2 * torch.ones(n_obs2)), obs=obs2)


def target_only_model(n_samp, n_drug, s_idx, d_idx, obs=None, n_obs=None, k=1):
    if obs is None and n_obs is None:
        print('Error!: both obs and n_obs are None.')
    if obs is not None:
        n_obs = obs.shape[0]
    # create global offset
    a_sigma = pyro.sample('a_sigma', dist.Gamma(ALPHA, BETA))
    a = pyro.sample('a', dist.Normal(0, a_sigma))
    # create s
    s_sigma = pyro.sample('s_sigma', dist.Gamma(ALPHA, BETA))
    with pyro.plate('s_plate', n_samp):
        with pyro.plate('k_s', k):
            s = pyro.sample('s', dist.Normal(0, s_sigma))
    # create d
    d_sigma = pyro.sample('d_sigma', dist.Gamma(ALPHA, BETA))
    with pyro.plate('d_plate', n_drug):
        with pyro.plate('k_d', k):
            d = pyro.sample('d', dist.Normal(0, d_sigma))
    # multiply s and d to create matrix
    s = torch.transpose(s, 0, 1)
    mat = torch.matmul(s, d)  # should be: n-samp x n-drug
    assert (mat.shape[0] == n_samp) and (mat.shape[1] == n_drug)
    mean = mat[s_idx, d_idx] + a
    sigma = pyro.sample('sigma', dist.Gamma(ALPHA, BETA))
    with pyro.plate('data_plate', n_obs):
        data = pyro.sample('data', dist.Normal(mean, sigma * torch.ones(n_obs)), obs=obs)
    return data


# n_samp: number of samples
# n_drug: number of drugs
# obs: torch.Tensor of observations
# s_idx: numpy array where s_idx[i] is the index of the sample for the i-th observation
# d_idx: numpy array where d_idx[i] is the index of the drug for the i-th observation
def model(n_samp, n_drug, s_idx, d_idx, params, obs=None, n_obs=None, k=1):
    if k != 1:
        print('need k = 1!')
    print('NORMAL MODEL!')
    if obs is None and n_obs is None:
        print('Error!: both obs and n_obs are None.')
    if obs is not None:
        n_obs = obs.shape[0]
    # create global offset
    alpha = torch.Tensor([params['alpha']])
    beta = torch.Tensor([params['beta']])
    a_sigma = pyro.param('a_sigma', dist.Gamma(alpha, beta), constraint=constraints.positive)
    a = pyro.sample('a', dist.Normal(torch.zeros(()), a_sigma * torch.ones(())))
    # create s
    s_sigma = pyro.param('s_sigma', dist.Gamma(alpha, beta), constraint=constraints.positive)
    a_s_sigma = pyro.param('a_s_sigma', dist.Gamma(alpha, beta), constraint=constraints.positive)
    with pyro.plate('s_plate', n_samp):
        a_s = pyro.sample('a_s', dist.Normal(torch.zeros(n_samp), a_s_sigma * torch.ones(n_samp)))
        s = pyro.sample('s', dist.Normal(torch.zeros(n_samp), s_sigma * torch.ones(n_samp)))
    # create d
    d_sigma = pyro.param('d_sigma', dist.Gamma(alpha, beta), constraint=constraints.positive)
    a_d_sigma = pyro.param('a_d_sigma', dist.Gamma(alpha, beta), constraint=constraints.positive)
    with pyro.plate('d_plate', n_drug):
        a_d = pyro.sample('a_d', dist.Normal(torch.zeros(n_drug), a_d_sigma * torch.ones(n_drug)))
        d = pyro.sample('d', dist.Normal(torch.zeros(n_drug), d_sigma * torch.ones(n_drug)))
    # create data
    mean = s[s_idx] * d[d_idx] + a_s[s_idx] + a_d[d_idx] + a
    sigma = pyro.sample('sigma', dist.Gamma(params['alpha'], params['beta']))
    with pyro.plate('data_plate', n_obs):
        data = pyro.sample('data', dist.Normal(mean, sigma * torch.ones(n_obs)), obs=obs)
    return data
