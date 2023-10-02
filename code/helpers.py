import numpy as np
import pandas as pd
import pickle
import pyro.util
import sys

def write_pickle(obj, fn):
    with open(fn, 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)

def read_pickle(fn):
    with open(fn, 'rb') as handle:
        obj = pickle.load(handle)
    return obj

def get_unique_pairs(df, col1, col2):
    a = df[[col1, col2]].drop_duplicates()
    pairs = list(zip(a[col1], a[col2]))
    assert len(pairs) == len(set(pairs))
    return pairs

def index_pairs(pairs, indices):
    p = np.array(pairs)
    idx = np.array(indices)
    return list(map(tuple, p[idx]))

def split_by_pairs(df, col1, col2, train_pairs, test_pairs):
    df['pair'] = list(zip(df[col1], df[col2]))
    train_df = df.loc[df['pair'].isin(train_pairs)]
    test_df = df.loc[df['pair'].isin(test_pairs)]
    assert set(train_df['pair']).isdisjoint(set(test_df['pair']))
    assert len(train_df) + len(test_df) == len(df)
    return train_df, test_df

def random_split(df, split_seed, holdout_frac):
    # get unique sample-drug pairs
    a = df[['sample_id', 'drug_id']].drop_duplicates()
    pairs = list(zip(a['sample_id'], a['drug_id']))
    assert len(pairs) == len(set(pairs))
    n_train = int(np.ceil(len(pairs) * (1 - holdout_frac)))
    # randomly permute indices of pairs, select train and test indices into pairs
    pyro.util.set_rng_seed(split_seed)
    idx = np.random.permutation(len(pairs))
    train_idx = idx[:n_train]
    test_idx = idx[n_train:]
    assert set(train_idx).isdisjoint(set(test_idx))
    # use randomly-permuted indices to index into pairs
    train_pairs = index_pairs(pairs, train_idx)
    test_pairs = index_pairs(pairs, test_idx)
    return train_pairs, test_pairs

# get all (sample_id, drug_id) pairs in dataframe df with sample_id in samples
def pairs_from_samples(df, sample_ids):
    samp_rows = df.loc[df.sample_id.isin(sample_ids)]
    return samp_rows['pair'].to_numpy()

def fold_split(df, fold_fn, split_seed):
    fold_list = read_pickle(fold_fn)
    # split_seed is 1-indexed, but fold_list is 0-indexed --> subtract 1 off
    samples = list(fold_list[split_seed])
    # get all pairs corresponding to a set of samples
    train_df = df.loc[~df.sample_id.isin(samples)].drop_duplicates()
    train_pairs = list(zip(train_df['sample_id'], train_df['drug_id']))
    test_df = df.loc[df.sample_id.isin(samples)].drop_duplicates()
    test_pairs = list(zip(test_df['sample_id'], test_df['drug_id']))
    # check that train and test split pairs correctly
    assert set(train_pairs).isdisjoint(set(test_pairs))
    df_pairs = list(zip(df['sample_id'], df['drug_id']))
    assert set(train_pairs).union(set(test_pairs)) == set(df_pairs)
    return train_pairs, test_pairs

# returns the list of (sample, drug) pairs in the test set
def get_train_test_pairs(df, fold_fn, split_seed, holdout_frac, split_type):
    assert split_type in ['random_split', 'sample_split']
    if split_type == 'random_split':
        assert holdout_frac >= 0 and holdout_frac <= 1
        return random_split(df, split_seed, holdout_frac)
    else:
        return fold_split(df, fold_fn, split_seed)

def split_train_test(df, fold_fn, split_seed, holdout_frac, split_type):
    # read in dataframe, and restrict to sample_id and drug_id
    train_pairs, test_pairs = get_train_test_pairs(df[['sample_id', 'drug_id']], fold_fn, split_seed, holdout_frac, split_type)
    train_df, test_df = split_by_pairs(df, 'sample_id', 'drug_id', train_pairs, test_pairs)
    return train_df, test_df

def get_source(data_fn, source_col):
    df = pd.read_csv(data_fn)
    assert len(df) == len(df.drop_duplicates())
    source_df = df[['sample_id', 'drug_id', source_col]]
    assert len(source_df) == len(df)
    return source_df

def get_target(data_fn, fold_fn, target_col, split_seed, holdout_frac, split_type):
    df = pd.read_csv(data_fn)
    assert len(df) == len(df.drop_duplicates())
    n_samp = df.sample_id.nunique()
    n_drug = df.drug_id.nunique()
    train_df, test_df = split_train_test(df, fold_fn, split_seed, holdout_frac, split_type)
    target_train_df = train_df[['sample_id', 'drug_id', target_col]]
    target_test_df = test_df[['sample_id', 'drug_id', target_col]]
    assert len(target_train_df) + len(target_test_df) == len(df)
    return target_train_df, target_test_df, n_samp, n_drug

def get_sample_drug_ids(df):
    sd = df[['sample_id', 'drug_id']].drop_duplicates()
    # confirming that df has no duplicate pairs
    assert len(sd) == len(df)
    return sd

def pearson_correlation(vec1, vec2):
    assert len(vec1) == len(vec2)
    pearson_corr = np.corrcoef(vec1, vec2)
    return pearson_corr[0, 1]

def get_sample_drug_indices(df):
    s_idx = df['sample_id'].to_numpy()
    d_idx = df['drug_id'].to_numpy()
    return s_idx, d_idx

def zscore(a):
    mu = np.mean(a)
    sigma = np.std(a)
    res = (a - mu) / sigma
    return mu, sigma, res

def inverse_zscore(b, mu, sigma):
    return (b * sigma) + mu