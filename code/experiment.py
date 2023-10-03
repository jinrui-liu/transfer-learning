import sys
import numpy as np
import pandas as pd
import pyro
import pyro.util
import torch
import tqdm
from pyro.infer import SVI, Trace_ELBO
from pyro.infer.autoguide import AutoNormal
from pyro.optim import Adam
from sklearn.model_selection import KFold

import cross_val
import helpers
import model as modeling

# Real values for K_LIST, used for train_target_only
K_LIST = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]

N_MODELS = 5
N_SPLITS = 5


def check_params(method, source, target, split_type, holdout_frac, fold_fn, split_seed, n_steps, k_for_transfer):
    assert method in ['raw', 'transfer', 'target_only']
    assert target in ['REP', 'GDSC', 'CTD2']
    assert split_type in ['random_split', 'sample_split']
    assert split_seed >= 0
    assert n_steps >= 5
    if method == 'target_only':
        assert split_type == 'random_split'
    if method == 'transfer' or method == 'raw':
        assert k_for_transfer != ""
        assert source in ['REP', 'GDSC', 'CTD2']
    if split_type == 'random_split':
        assert 0 <= holdout_frac <= 1
    if split_type == 'sample_split':
        assert fold_fn != ""


def get_raw_args(args, n):
    if len(args) != n + 1:
        print('Error! Expected ' + str(n + 1) + ' arguments but got ' + str(len(args)))
        return
    method = args[1].split("=")[1]
    source = args[2].split("=")[1]
    target = args[3].split("=")[1]
    split_type = args[4].split("=")[1]
    holdout_frac = float(args[5].split("=")[1])
    data_fn = args[6].split("=")[1]
    write_dir = args[7].split("=")[1]
    fold_fn = args[8].split("=")[1]
    n_steps = int(args[9].split("=")[1])
    split_seed = int(args[10].split("=")[1])
    k_for_transfer = int(args[11].split("=")[1])

    # verify that params are valid for method given and save
    check_params(method, source, target, split_type, holdout_frac, data_fn, split_seed, n_steps, k_for_transfer)
    pd.DataFrame({'method': [method], 'source': [source], 'target': [target], 'split_type': [split_type],
                  'holdout_frac': [holdout_frac], 'data_fn': [data_fn], 'write_dir': [write_dir], 'fold_fn': [fold_fn],
                  'split_seed': [split_seed], 'n_steps': [n_steps], 'k_for_transfer': [k_for_transfer]}).to_csv(
        write_dir + '/params.csv', index=False)
    return method, source, target, split_type, holdout_frac, data_fn, write_dir, fold_fn, split_seed, n_steps, \
        k_for_transfer


def predict_raw_helper(source_df, source_col, target_sd):
    d = source_df.merge(target_sd, on=['sample_id', 'drug_id'], validate='one_to_one')
    assert len(d) == len(target_sd)
    predictions = d[source_col].to_numpy()
    return predictions


def predict_raw(source_df, source_col, target_train_sd, target_test_sd):
    train_predict = predict_raw_helper(source_df, source_col, target_train_sd)
    test_predict = predict_raw_helper(source_df, source_col, target_test_sd)
    return train_predict, test_predict


def matrix_transfer(s, d, c, u_loc, v_loc, k):
    j = np.zeros((k, k))
    for u, v in zip(u_loc, v_loc):
        j += np.matmul(u, v)
    w = c * np.diag(np.ones(k)) + j
    # s already comes transposed, as defined in model
    assert s.shape[0] == k
    assert w.shape[0] == k and w.shape[0] == k
    s_prime = np.matmul(w, s)
    mat2 = np.matmul(np.transpose(s_prime), d)
    return mat2


# Fit a model, extract S, W, and D, and make predictions on the test set
def predict_transfer(model_seed, s_idx_src, d_idx_src, obs_src, s_idx_train, d_idx_train, obs_train, s_idx_test,
                     d_idx_test, n_samp, n_drug, n_steps, k, r, write_dir=None):
    # print('Transfer, k: ' + str(k) + ', r: ' + str(r))
    # FIT MODEL
    pyro.clear_param_store()
    pyro.util.set_rng_seed(model_seed)
    adam_params = {"lr": 0.05}
    optimizer = Adam(adam_params)
    autoguide = AutoNormal(modeling.transfer_model)
    svi = SVI(modeling.transfer_model, autoguide, optimizer, Trace_ELBO())
    losses = []
    for _ in tqdm.trange(n_steps):
        svi.step(n_samp, n_drug, s_idx_src, d_idx_src, s_idx_train, d_idx_train, obs_src, len(obs_src), obs_train,
                 len(obs_train), k=k, r=r)
        loss = svi.evaluate_loss(n_samp, n_drug, s_idx_src, d_idx_src, s_idx_train, d_idx_train, obs_src, len(obs_src),
                                 obs_train, len(obs_train), k=k, r=r)
        losses.append(loss)
    # save the loss for all steps only for actual training
    if write_dir is not None:
        pd.DataFrame({'loss': losses, 'step': np.arange(1, n_steps+1)}).to_csv(write_dir + '/losses.csv', index=False)
    # MAKE INITIAL PREDICTIONS BASED ON MODEL
    # retrieve values out for s and d vectors
    s_loc = pyro.param("AutoNormal.locs.s").detach().numpy()
    d_loc = pyro.param("AutoNormal.locs.d").detach().numpy()
    # need to retrieve c, u, v and reconstruct s'!
    c_loc = pyro.param("AutoNormal.locs.c").detach().numpy()
    u_loc = []
    v_loc = []
    ranks = [i + 1 for i in list(range(r))]
    for i in ranks:
        u_loc.append(pyro.param(f"AutoNormal.locs.u_{i}").detach().numpy())
        v_loc.append(pyro.param(f"AutoNormal.locs.v_{i}").detach().numpy())
    # predict function: takes in c, u, v, s, d --> mat2
    mat = matrix_transfer(s_loc, d_loc, c_loc, u_loc, v_loc, k)
    train_means = mat[s_idx_train, d_idx_train]
    test_means = mat[s_idx_test, d_idx_test]
    return train_means, test_means


# runs prediction model; transforms initial data to pass into predict_transfer
def run_predict_transfer(model_seed, source_df, source_col, target_train_df, target_col, s_idx_test, d_idx_test, n_samp,
                         n_drug, n_steps, k, r, write_dir=None):
    s_idx_train, d_idx_train = helpers.get_sample_drug_indices(target_train_df)
    obs_train = target_train_df[target_col].to_numpy()
    mu, sigma, obs_train = helpers.zscore(obs_train)
    obs_train = torch.Tensor(obs_train)
    s_idx_src, d_idx_src = helpers.get_sample_drug_indices(source_df)
    obs_src = source_df[source_col].to_numpy()
    _, _, obs_src = helpers.zscore(obs_src)
    obs_src = torch.Tensor(obs_src)
    train_initial, test_initial = predict_transfer(model_seed, s_idx_src, d_idx_src, obs_src, s_idx_train, d_idx_train,
                                                   obs_train, s_idx_test,
                                                   d_idx_test, n_samp, n_drug, n_steps, k, r, write_dir)
    train_predict = helpers.inverse_zscore(train_initial, mu, sigma)
    test_predict = helpers.inverse_zscore(test_initial, mu, sigma)
    assert len(train_predict) == len(s_idx_train)
    assert len(test_predict) == len(s_idx_test)
    return train_predict, test_predict


# wrapper to run transfer model to N_MODELS model seeds
def predict_transfer_wrapper(source_df, source_col, target_train_df, target_col, s_idx_test, d_idx_test, n_samp, n_drug,
                             n_steps, k, r, write_dir=None):
    train_predict_list = []
    test_predict_list = []
    for model_seed in range(0, N_MODELS):
        train_predict, test_predict = run_predict_transfer(model_seed, source_df, source_col, target_train_df,
                                                           target_col, s_idx_test, d_idx_test, n_samp,
                                                           n_drug, n_steps, k, r, write_dir)
        train_predict_list.append(train_predict)
        test_predict_list.append(test_predict)
    return train_predict_list, test_predict_list


def matrix_target_only(s, d, k):
    assert s.shape[0] == k
    assert d.shape[0] == k
    return np.matmul(np.transpose(s), d)


def predict_target_only(model_seed, s_idx_train, d_idx_train, obs_train, s_idx_test, d_idx_test, n_samp, n_drug,
                        n_steps, k):
    # print('Target only, k: ' + str(k))
    # FIT MODEL
    pyro.clear_param_store()
    pyro.util.set_rng_seed(model_seed)
    adam_params = {"lr": 0.05}
    optimizer = Adam(adam_params)
    autoguide = AutoNormal(modeling.target_only_model)
    svi = SVI(modeling.target_only_model, autoguide, optimizer, Trace_ELBO())
    losses = []
    for _ in tqdm.trange(n_steps):
        svi.step(n_samp, n_drug, s_idx_train, d_idx_train, obs_train, len(obs_train), k=k)
        loss = svi.evaluate_loss(n_samp, n_drug, s_idx_train, d_idx_train, obs_train, len(obs_train), k=k)
        losses.append(loss)
    # MAKE INITIAL PREDICTIONS BASED ON MODEL
    # retrieve values out for s and d vectors
    s_loc = pyro.param("AutoNormal.locs.s").detach().numpy()
    d_loc = pyro.param("AutoNormal.locs.d").detach().numpy()
    mat = matrix_target_only(s_loc, d_loc, k)
    train_means = mat[s_idx_train, d_idx_train]
    test_means = mat[s_idx_test, d_idx_test]
    return train_means, test_means


def run_predict_target_only(model_seed, target_train_df, target_col, s_idx_test, d_idx_test, n_samp, n_drug, n_steps,
                            k):
    s_idx_train, d_idx_train = helpers.get_sample_drug_indices(target_train_df)
    obs_train = target_train_df[target_col].to_numpy()
    mu, sigma, obs_train = helpers.zscore(obs_train)
    obs_train = torch.Tensor(obs_train)
    train_initial, test_initial = predict_target_only(model_seed, s_idx_train, d_idx_train, obs_train, s_idx_test,
                                                      d_idx_test, n_samp, n_drug, n_steps, k)
    train_predict = helpers.inverse_zscore(train_initial, mu, sigma)
    test_predict = helpers.inverse_zscore(test_initial, mu, sigma)
    assert len(train_predict) == len(s_idx_train)
    assert len(test_predict) == len(s_idx_test)
    return train_predict, test_predict


def predict_target_only_wrapper(target_train_df, target_col, s_idx_test, d_idx_test, n_samp, n_drug, n_steps, k):
    train_predict_list = []
    test_predict_list = []
    for model_seed in range(0, N_MODELS):
        train_predict, test_predict = run_predict_target_only(model_seed, target_train_df, target_col, s_idx_test,
                                                              d_idx_test, n_samp, n_drug, n_steps, k)
        train_predict_list.append(train_predict)
        test_predict_list.append(test_predict)
    return train_predict_list, test_predict_list


def evaluate_correlation(predictions, df, col):
    test = df[col].to_numpy()
    return helpers.pearson_correlation(predictions, test)


def evaluate(train_predict_list, test_predict_list, target_train_df, target_test_df, target_col):
    assert len(train_predict_list) == len(test_predict_list)
    train_corr_list = []
    test_corr_list = []
    for i in range(0, N_MODELS):
        train_corr = evaluate_correlation(train_predict_list[i], target_train_df, target_col)
        test_corr = evaluate_correlation(test_predict_list[i], target_test_df, target_col)
        train_corr_list.append(train_corr)
        test_corr_list.append(test_corr)
    idx = np.argmax(train_corr_list)
    train_result = train_corr_list[idx]
    test_result = test_corr_list[idx]
    train_predictions = train_predict_list[idx]
    test_predictions = test_predict_list[idx]
    # print('test_corr_list:')
    # print(test_corr_list)
    # print('train_corr_list:')
    # print(train_corr_list)
    return train_result, test_result, train_predictions, test_predictions


# Returns column names in dataset for given method
def get_column_names(method, source_name, target_name):
    suffix = '_published_auc_mean'
    prefix = ''
    if method == 'raw':
        # use published mean auc as raw baseline
        prefix = ''
    elif method == 'transfer' or method == 'target_only':
        # use log(published mean auc) for ML models
        prefix = 'log_'
    source_col = prefix + source_name + suffix
    target_col = prefix + target_name + suffix
    return source_col, target_col


def obs_to_tensor(vec):
    return torch.Tensor(vec)


def choose_k_target_only(method, target_train_df, target_col, split_type, n_samp, n_drug, n_steps):
    # print('Choose k via cross validation')
    assert method == 'target_only'
    assert split_type == 'random_split'
    # get data (either pairs or samples) based on split_type
    x = cross_val.get_items_to_split(target_train_df, split_type)
    kf = KFold(n_splits=N_SPLITS, random_state=707, shuffle=True)
    kf.get_n_splits(x)
    # array where cell (i,j) holds validation score from running i-th fold with k = K_LIST[j]
    v = np.ones((N_SPLITS, len(K_LIST))) * -np.inf
    for i, (train_index, val_index) in enumerate(kf.split(x)):
        # print("Fold: " + str(i + 1))
        train_df, val_df = cross_val.split_dataframe(target_train_df, 'sample_id', 'drug_id', x, split_type,
                                                     train_index, val_index)
        for j in range(0, len(K_LIST)):
            k = K_LIST[j]
            s_idx_val, d_idx_val = helpers.get_sample_drug_indices(val_df)
            # print('Run ' + str(N_MODELS) + ' model restarts')
            train_predict_list, val_predict_list = predict_target_only_wrapper(train_df, target_col, s_idx_val,
                                                                               d_idx_val, n_samp, n_drug, n_steps, k)
            _, val_corr, _, _ = evaluate(train_predict_list, val_predict_list, train_df, val_df, target_col)
            v[i, j] = val_corr
    # check that all entries have been filled in
    assert np.sum(np.sum(v == -np.inf)) == 0
    avg_v = np.mean(v, axis=0)
    assert len(avg_v == len(K_LIST))
    return K_LIST[np.argmax(avg_v)]


def save_k_r_transfer(write_fn, k, r, optimal_v_corr):
    pd.DataFrame({'k': k, 'optimal_r': r, 'optimal_v_corr': optimal_v_corr}).to_csv(write_fn, index=False)


def choose_k_r_transfer(method, target_train_df, target_col, split_type, n_samp, n_drug, n_steps, source_df=None,
                        source_col=None, k_for_transfer=None, write_fn=None):
    # print('Choose k via cross validation')
    assert method == 'transfer'
    assert split_type in ['random_split', 'sample_split']
    if method == 'transfer':
        assert source_df is not None
        assert source_col is not None
        assert k_for_transfer is not None
        assert write_fn is not None
    # get data (either pairs or samples) based on split_type
    x = cross_val.get_items_to_split(target_train_df, split_type)
    kf = KFold(n_splits=N_SPLITS, random_state=707, shuffle=True)
    kf.get_n_splits(x)
    # set optimal parameters to -inf for cross validation
    # start loop through k in K_LIST
    k = k_for_transfer
    ranks = [i + 1 for i in list(range(k))]
    # array where cell (i,j) holds validation score from running i-th fold with r from 1 to k
    v = np.ones((N_SPLITS, len(ranks))) * -np.inf
    for i, (train_index, val_index) in enumerate(kf.split(x)):
        # print("Fold: " + str(i + 1))
        train_df, val_df = cross_val.split_dataframe(target_train_df, 'sample_id', 'drug_id', x, split_type,
                                                     train_index, val_index)
        for j in range(len(ranks)):
            r = ranks[j]
            s_idx_val, d_idx_val = helpers.get_sample_drug_indices(val_df)
            # print('Run ' + str(N_MODELS) + ' model restarts')
            assert r <= k
            train_predict_list, val_predict_list = predict_transfer_wrapper(source_df, source_col, train_df,
                                                                            target_col, s_idx_val, d_idx_val,
                                                                            n_samp,
                                                                            n_drug, n_steps, k, r)
            _, val_corr, _, _ = evaluate(train_predict_list, val_predict_list, train_df, val_df, target_col)
            v[i, j] = val_corr
    # check that all entries have been filled in
    assert np.sum(np.sum(v == -np.inf)) == 0
    avg_v = np.mean(v, axis=0)
    assert len(avg_v == len(ranks))
    # set the optimal parameters with k and r where avg_v is maximized
    optimal_v_corr = max(avg_v)
    optimal_r = ranks[np.argmax(avg_v)]
    assert optimal_r != -np.inf and optimal_v_corr != -np.inf and optimal_r <= k
    print('under k: ' + str(k), 'optimal r: ' + str(optimal_r))
    d = pd.DataFrame({"k": [k], "optimal_r": [optimal_r], "optimal_v_corr": [optimal_v_corr]})
    d.to_csv(write_fn + f'/k{k}_optimal_param.csv', index=False)
    return k, optimal_r


def save_predictions(write_fn, predictions, df):
    assert len(predictions) == len(df)
    d = {'predictions': predictions, 'sample_ids': df['sample_id'].to_numpy(), 'drug_id': df['drug_id'].to_numpy()}
    df_out = pd.DataFrame(data=d)
    df_out.to_csv(write_fn, index=False)


def main():
    method, source_name, target_name, split_type, holdout_frac, data_fn, write_dir, fold_fn, split_seed, n_steps, \
        k_for_transfer = get_raw_args(sys.argv, 11)
    source_col, target_col = get_column_names(method, source_name, target_name)
    # Split target dataset into train and test, get number of samples and drugs in dataset
    target_train_df, target_test_df, n_samp, n_drug = helpers.get_target(data_fn, fold_fn, target_col, split_seed,
                                                                         holdout_frac, split_type)
    train_predict_list = []
    test_predict_list = []
    # ================================
    # MAKE PREDICTIONS BY METHOD
    assert method in ['raw', 'target_only', 'transfer']
    if method == 'raw':
        target_train_sd = helpers.get_sample_drug_ids(target_train_df)
        target_test_sd = helpers.get_sample_drug_ids(target_test_df)
        source_df = helpers.get_source(data_fn, source_col)
        train_predictions, test_predictions = predict_raw(source_df, source_col, target_train_sd, target_test_sd)
        train_corr = evaluate_correlation(train_predictions, target_train_df, target_col)
        test_corr = evaluate_correlation(test_predictions, target_test_df, target_col)
    else:
        # Get sample_ids and drug_ids for test set
        s_idx_test, d_idx_test = helpers.get_sample_drug_indices(target_test_df)
        if method == 'target_only':
            # cross validation to choose parameter k
            k = choose_k_target_only('target_only', target_train_df, target_col, split_type, n_samp, n_drug, n_steps)
            # given the above k, fit the model on the entire training set and make predictions
            train_predict_list, test_predict_list = predict_target_only_wrapper(target_train_df, target_col, s_idx_test,
                                                                                d_idx_test, n_samp, n_drug, n_steps, k)
        elif method == 'transfer':
            # get source data
            source_df = helpers.get_source(data_fn, source_col)
            # cross validation to choose parameter k
            k, r = choose_k_r_transfer('transfer', target_train_df, target_col, split_type, n_samp, n_drug, n_steps,
                                       source_df, source_col, k_for_transfer, write_dir)
            print('done choosing k and r')
            # given the above k, fit the model on the entire training set and make predictions
            train_predict_list, test_predict_list = predict_transfer_wrapper(source_df, source_col, target_train_df,
                                                                             target_col, s_idx_test, d_idx_test,
                                                                             n_samp,
                                                                             n_drug, n_steps, k, r, write_dir)
        train_corr, test_corr, train_predictions, test_predictions = evaluate(train_predict_list, test_predict_list,
                                                                              target_train_df, target_test_df,
                                                                              target_col)
    # ================================
    # EVALUATE PREDICTIONS AND SAVE
    pd.DataFrame({'train_corr': [train_corr], 'test_corr': [test_corr]}).to_csv(write_dir + '/correlations.csv',
                                                                                index=False)
    save_predictions(write_dir + '/train_predictions.csv', train_predictions, target_train_df)
    save_predictions(write_dir + '/test_predictions.csv', test_predictions, target_test_df)

    print('train_corr: ' + str(train_corr))
    print('test_corr: ' + str(test_corr))
    print('Done!')


if __name__ == "__main__":
    main()
