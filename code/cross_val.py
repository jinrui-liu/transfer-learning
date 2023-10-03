import numpy as np
import pandas as pd

import helpers

def split_by_samples(df, train_samples, test_samples):
    train_df = df.loc[df['sample_id'].isin(train_samples)]
    test_df = df.loc[df['sample_id'].isin(test_samples)]
    assert set(train_df['sample_id']).isdisjoint(set(test_df['sample_id']))
    assert len(train_df) + len(test_df) == len(df)
    return train_df, test_df

def split_dataframe(df, col1, col2, X, split_type, train_index, test_index):
    train_vals = X[train_index]
    test_vals = X[test_index]
    if split_type == 'random_split':
        train_vals = list(map(tuple, train_vals))
        test_vals = list(map(tuple, test_vals))
        train_df, test_df = helpers.split_by_pairs(df, col1, col2, train_vals, test_vals)
    elif split_type == 'sample_split':
        train_df, test_df = split_by_samples(df, train_vals, test_vals)
    else:
        print('Error! Need valid split_type.')
        return
    return train_df, test_df
        
def get_items_to_split(df, split_type):
    if split_type == 'random_split':
        # split by pairs
        X = df[['sample_id', 'drug_id']].to_numpy()
    elif split_type == 'sample_split':
        X = np.array(list(df['sample_id'].unique()))
    else:
        print('Error! split_type must be in random_split or sample_split.')
        return
    return X