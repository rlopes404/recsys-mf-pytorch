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

df = pd.read_csv('/home/ramon/Downloads/Douban-movies/movie/douban_movie.tsv', header=0, names=['user', 'item', 'rating', 'timestamp'], sep='\t')
df = df[df.rating > 0]


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



bar = df.groupby(['ts']).agg(popularity=('user_id','count'), unique=('user_id', 'nunique')).reset_index()
bar['cumsum'] = bar['popularity'].cumsum()
#bar['cumsum_user'] = bar['unique'].cumsum()
sns.lineplot(data=bar, x="ts", y="cumsum")
#sns.lineplot(data=bar, x="ts", y="cumsum_user")
plt.show()

foo = df.groupby(['ts', 'item_id']).agg(popularity=('user_id','nunique'), avg_rating=('rating', 'mean')).reset_index()
foo['cumsum'] = foo.groupby(['item_id'])['popularity'].cumsum()


topK=1
bar = df.groupby(['item_id']).agg(count=('user_id', 'count')).reset_index()
items = bar.sort_values('count', ascending=False)[:topK]['item_id']

idx = foo['item_id'].isin(items)


d = foo[idx]
fig,ax=plt.subplots()
ax.plot(d['ts'], d['popularity'], color="red")
ax.set_xlabel("ts")
ax.set_ylabel("popularity", color="red")

# twin object for two different y-axis on the sample plot
ax2=ax.twinx()
# make a plot with different y-axis using second axis object
ax2.plot(d['ts'], d['avg_rating'], color="blue")
ax2.set_ylabel("avg_rating", color="blue", fontsize=14)
plt.show()



sns.lineplot(data=foo[idx], x="ts", y="popularity", hue='item_id')
plt.show()