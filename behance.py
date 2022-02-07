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

df = pd.read_csv('/home/ramon/Downloads/behance/Behance_appreciate_1M.csv', header=None, names=['user', 'item', 'timestamp'], sep=' ')

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



bar = df.groupby(['ts']).agg(popularity=('user_id','count')).reset_index()
bar['cumsum'] = bar['popularity'].cumsum()
sns.lineplot(data=bar, x="ts", y="cumsum")
plt.show()

foo = df.groupby(['ts', 'item_id']).agg(popularity=('user_id','nunique')).reset_index()

foo['cumsum'] = foo.groupby(['item_id'])['popularity'].cumsum()


topK=10
bar = df.groupby(['item_id']).agg(count=('user_id', 'count')).reset_index()
items = bar.sort_values('count', ascending=False)[:topK]['item_id']

idx = foo['item_id'].isin(items)

sns.lineplot(data=foo[idx], x="ts", y="cumsum", hue='item_id')
plt.show()