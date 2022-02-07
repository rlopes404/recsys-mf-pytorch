import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F


from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import pandas as pd
import numpy as np
import sys

df = pd.read_csv('/home/ramon/Downloads/ml-1m/ratings.dat', header=None, names=['user', 'item', 'rating', 'timestamp'], sep='::')

from datetime import datetime
dates = df['timestamp'].apply(datetime.fromtimestamp)
df['ts'] = dates.dt.year.astype(int)*100 + dates.dt.month.astype(int)

df = df.sort_values(by=['ts']).reset_index(drop=True)

def encode_columns(col):
    keys = col.unique()
    key_to_id = {key:idx for idx, key in enumerate(keys)}
    return key_to_id

u_map = encode_columns(df['user'])
i_map = encode_columns(df['item'])

df['user_id'] = df['user'].map(u_map)
df['item_id'] = df['item'].map(i_map)

ts_cut = df.loc[int(len(df)*0.7)]['ts']
idx_train = df['ts'] <= ts_cut

train = df[idx_train].copy().reset_index(drop=True)
test = df[~idx_train].copy().reset_index(drop=True)
assert test['ts'].isin(train['ts']).sum() == 0


foo = train.groupby(['ts', 'item_id']).agg(popularity=('user_id','nunique')).reset_index()

foo['cumsum'] = foo.groupby(['item_id'])['popularity'].cumsum()

if False:
    topK = 10

    bar = df.groupby(['item_id']).agg(count=('user', 'count')).reset_index()
    items = bar.sort_values('count', ascending=False)[:topK]['item_id']

    idx = foo['item_id'].isin(items)

    sns.lineplot(data=foo[idx], x="ts", y="cumsum", hue='item_id')
    plt.show()

# como saber se é burst ou n items ou n usuarios aumentou