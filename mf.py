import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F


from torch.utils.data import Dataset
from torch.utils.data import DataLoader


import pickle
from fairness_opt import FairnessMF

import pandas as pd
import numpy as np
np.random.seed(0)
import sys
import time

load_model = False

#df = pd.read_csv('/home/ramon/Downloads/ml-1m/ratings.dat', header=None, names=['user', 'item', 'rating', 'timestamp'], sep='::')
df = pd.read_csv('/home/ramon/Downloads/ml-100k/u.data', header=None, names=['user', 'item', 'rating', 'timestamp'], sep='\t')
#df = pd.read_csv('/home/ramon/Downloads/Douban-movies/movie/douban_movie.tsv', header=0, names=['user', 'item', 'rating', 'timestamp'], sep='\t')
#df = pd.read_csv('/home/ramon/Downloads/Ciao/douban.txt', header=0, names=['user', 'item', 'category', 'rating', 'helpness', 'timestamp'], sep=',')

df = df.drop_duplicates(subset=['user','item', 'timestamp'])
df = df[df.rating > 0].reset_index(drop=True)

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

ts_cut = df.loc[int(len(df)*0.8)]['ts']
idx_train = df['ts'] <= ts_cut

train = df[idx_train].copy().reset_index(drop=True)
test = df[~idx_train].copy().reset_index(drop=True)
assert test['ts'].isin(train['ts']).sum() == 0

cutoff = int(0.5*len(test))
valid = test[:cutoff].copy().reset_index(drop=True)
test = test[cutoff:].copy().reset_index(drop=True)

pop = train.groupby(['item_id']).agg(popularity=('user_id','nunique')).reset_index()
#denominator = len(train)
denominator = train['user_id'].nunique()
pop['popularity'] = (pop['popularity']/denominator).astype(np.float32)

pop_map = {item : count for item, count in zip(pop['item_id'], pop['popularity'])}

counts = train.groupby(['ts', 'item_id']).agg(numerator=('user_id','nunique')).reset_index()
sum_counts = train.groupby(['ts']).agg(denominator=('user_id', 'nunique')).reset_index()
merged = counts.merge(sum_counts, on='ts')
merged['propensity'] = (merged['numerator']/merged['denominator']).astype(np.float32)

merged = merged.merge(pop, on='item_id')
merged = merged.merge(train[['user_id', 'item_id', 'ts', 'rating']], on=['item_id','ts'])

assert len(train) == len(merged)
# plt.hist(merged['propensity'], bins=100, label='propensity')
# plt.hist(merged['popularity'], bins=100, label='popularity')
# plt.legend()
# plt.show()

train = merged[['user_id', 'ts', 'item_id', 'popularity', 'propensity', 'rating']]


assert train.duplicated(subset=['user_id', 'item_id', 'ts']).sum() == 0

user_avg_rating = {}
user_train_items = {}
for user_id, group in train.groupby(['user_id']):
    user_avg_rating[user_id] = np.mean(group['rating'].values)
    user_train_items[user_id] = np.asarray(group['item_id'])


idx = (test['item_id'].isin(train['item_id'])) & (test['user_id'].isin(train['user_id']))
test = test[idx].reset_index(drop=True)
#(~test['user_id'].isin(train['user_id'])).sum()

def get_items_eval(df):
    user_items = {}
    user_relevant_items = {}
    for user_id, group in df.groupby(['user_id']):
        if user_id in user_avg_rating:
            user_items[user_id] = set(group['item_id'].values)

            idx = group['rating'] >= user_avg_rating[user_id]
        
            user_relevant_items[user_id] = set(group.loc[idx, 'item_id'])
    return user_items, user_relevant_items
    
#val_user_items, val_user_relevant_items = get_items_eval(valid)
test_user_items, test_user_relevant_items = get_items_eval(test)


pop = train.groupby(['item_id']).agg(popularity=('user_id','count')).reset_index().sort_values(by=['popularity'], ascending=False).reset_index(drop=True)
pop['cum'] = pop['popularity'].cumsum()/pop['popularity'].sum()
pop['item_pct'] = np.arange(1, len(pop)+1)/len(pop)

pop[pop.cum <= 0.4]

#sns.lineplot(data=pop, x='item_pct', y="cum")
#plt.show()


if False:
    topK = 10
    topK = -1
    t = 0
    t = 200100


    bar = df.groupby(['item']).agg(count=('user', 'count')).reset_index()
    items = bar.sort_values('count', ascending=False)[:topK]['item']

    idx = df['item'].isin(items)

    foo = df[idx & (df.ts > t)].groupby(['ts', 'item']).agg(count=('user', 'count')).reset_index()

    sns.lineplot(data=foo, x="ts", y="count", hue='item')
    plt.show()

    

# # dataset definition
class MFDataset(Dataset):
    # load the dataset
    def __init__(self, interactions):
        # ['user_id', 'ts', 'item_id', 'popularity', 'propensity', 'rating']
        self.user_id = interactions['user_id'].values
        self.item_id = interactions['item_id'].values        
        self.rating = interactions['rating'].values
        self.popularity = interactions['popularity'].values
        self.propensity = interactions['propensity'].values

    # number of rows in the dataset
    def __len__(self):
        return len(self.user_id)

    # get a row at an index
    def __getitem__(self, idx):
        return [self.user_id[idx], self.item_id[idx], self.rating[idx], self.popularity[idx], self.propensity[idx]]


class MF(nn.Module):

    def __init__(self, num_users, num_items, emb_size=100):
        super(MF, self).__init__()

        self.n_users = num_users
        self.n_items = num_items
        self.emb_size = emb_size

        self.user_emb = nn.Embedding(num_users, emb_size)
        self.item_emb = nn.Embedding(num_items, emb_size)
        
        nn.init.xavier_normal_(self.user_emb.weight)
        nn.init.xavier_normal_(self.item_emb.weight)

    def forward(self, user, item):
        u_emb = self.user_emb(user)
        i_emb = self.item_emb(item)
        return torch.mul(u_emb, i_emb).sum(-1).float()

    def calculate_loss(self, pred, target, weight):
        loss = (pred - target)**2        
        return torch.mul(torch.reciprocal(weight), loss).mean()

    def predict(self, user, item):
        pred = self.forward(user, item)
        return pred

    def full_predict(self, user):
        #test_item_emb = self.item_emb.weight.view(self.n_items, 1, self.emb_size)
        scores = torch.matmul(self.user_emb(user), self.item_emb.weight.transpose(0,1))
        return scores

    def full_predict(self, user):
        #test_item_emb = self.item_emb.weight.view(self.n_items, 1, self.emb_size)
        scores = torch.matmul(self.user_emb(user), self.item_emb.weight.transpose(0,1))
        return scores    


def test_model(model):
    model.eval()
    user_ids = torch.LongTensor(test['user_id'].values)
    item_ids = torch.LongTensor(test['item_id'].values)
    ratings = torch.FloatTensor(test['rating'].values)
    y_hat = model(user_ids, item_ids)
    loss = F.mse_loss(y_hat, ratings)
    print("test loss %.3f " % loss.item())


def weighted_mse_loss(input, target, weight):
    return torch.sum(torch.reciprocal(weight)*((input - target)**2))

def train_data_loader(model, train_loader, epochs=10, lr=0.001, wd=0, type=None):
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    #optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wd)
        
    model.train()
    
    for _ep in range(epochs):
        print(_ep)
        for user_ids, item_ids, ratings, popularity, propensity in train_loader:
            if type=='pop':
                weights = torch.FloatTensor(popularity)
            elif type=='prop':
                weights = torch.FloatTensor(propensity)
            else:
                weights = torch.ones(len(item_ids))

            # clear the gradients
            optimizer.zero_grad()  
            # compute the model output / forward pass
            y_hat = model(user_ids, item_ids)       
            # compute the loss
            #loss = F.mse_loss(y_hat, ratings)
            loss = model.calculate_loss(y_hat, ratings, weights)
            # backpropagate the error through the model
            loss.backward()
            # update model weights
            optimizer.step()

def train_epocs(model, interactions, epochs=10, lr=0.001, wd=0, weights=None):
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    #optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wd)
        
    model.train()
    
    if weights == None:
        weights = torch.ones(len(interactions))

    for _ in range(epochs):
        user_ids = torch.LongTensor(interactions['user_id'].values)
        item_ids = torch.LongTensor(interactions['item_id'].values)
        ratings = torch.FloatTensor(interactions['rating'].values)

        
        # clear the gradients
        optimizer.zero_grad()  
        # compute the model output / forward pass
        y_hat = model(user_ids, item_ids)       
        # compute the loss
        #loss = F.mse_loss(y_hat, ratings)
        loss = model.calculate_loss(y_hat, ratings, weights)
        # backpropagate the error through the model
        loss.backward()
        # update model weights
        optimizer.step()

       # print(loss.item())
    
    #test(model)

#https://github.com/MaurizioFD/RecSys2019_DeepLearning_Evaluation/blob/0fb6b7f5c396f8525316ed66cf9c9fdb03a5fa9b/Base/Evaluation/metrics.py#L247

def rr(is_relevant):
    """
    Reciprocal rank of the FIRST relevant item in the ranked list (0 if none)
    :param is_relevant: boolean array
    :return:
    """

    ranks = np.arange(1, len(is_relevant) + 1)[is_relevant]

    if len(ranks) > 0:
        return 1. / ranks[0]
    else:
        return 0.0

def dcg(relevance_scores):
    return np.sum(np.divide(np.power(2, relevance_scores) - 1, np.log2(np.arange(relevance_scores.shape[0], dtype=np.float64) + 2)),
                  dtype=np.float64)

def ndcg(ranked_relevance, pos_items, at=None):
    
    relevance = np.ones_like(pos_items[:at])   

    rank_dcg = dcg(ranked_relevance[:at])

    if rank_dcg == 0.0:
        return 0.0

    ideal_dcg = dcg(relevance)
    if ideal_dcg == 0.0:
        return 0.0
        
    return rank_dcg / ideal_dcg    


def compute_metrics(model, user_id, train_items, test_items, topK, n_groups, item2group):
    preds = model.full_predict(torch.LongTensor([user_id])).detach().numpy().squeeze()
    preds[train_items] = -sys.maxsize
    
    ranked_list = np.argsort(-preds)[:topK]
    # index_partition = np.argpartition(-preds, topK-1)[:topK]
    # index_sorted = np.argsort(-preds[index_partition])
    # ranked_list = index_partition[index_sorted]


    _rank_group = np.array([0.0, 0.0])
    _count_group = np.array([0.0, 0.0])
    _pop_group = np.array([0.0, 0.0])
    _pop = 0
    for rank, item in enumerate(ranked_list):
        _rank_group[item2group[item]] += 1/np.log2(rank+2)
        _count_group[item2group[item]] += 1
        _pop_group[item2group[item]] += pop_map[item]
        _pop += pop_map[item]

    rank_scores = np.asarray([item in test_items for item in ranked_list])

    _test_items = np.array(list(test_items))
    _ndcg = ndcg(rank_scores.astype(int), _test_items, at=topK)

    _rr = rr(rank_scores)
    return _ndcg, _rr, _rank_group, _count_group, _pop_group, _pop

def compute_metrics_fairness(model, user_id, train_items, test_items, topK, n_groups, item2group, alpha):

    y_hat = model.full_predict(torch.LongTensor([user_id])).detach().numpy().squeeze()

    train_items = user_train_items[user_id]
    y_hat[train_items] = -sys.maxsize


    t0 = time.time()
    opt_model = FairnessMF(num_items, y_hat, n_groups, item2group, topK, alpha) #n_items, costs, n_groups, item2group, topK, alpha
    ranked_list = opt_model.get_fair_ranking()
    t1 = time.time()
    delta = t1-t0
    
    _rank_group = np.array([0.0, 0.0])
    _count_group = np.array([0.0, 0.0])
    _pop_group = np.array([0.0, 0.0])
    _pop = 0.0
    for rank, item in enumerate(ranked_list):
        _rank_group[item2group[item]] += 1/np.log2(rank+2)
        _count_group[item2group[item]] += 1
        _pop_group[item2group[item]] += pop_map[item]
        _pop += pop_map[item]

    rank_scores = np.asarray([item in test_items for item in ranked_list])

    _test_items = np.array(list(test_items))
    _ndcg = ndcg(rank_scores.astype(int), _test_items, at=topK)

    _rr = rr(rank_scores)
    return _ndcg, _rr, _rank_group, _count_group, _pop_group, _pop, delta

def evaluate(model, user_test_relevance, user_train_items, topK, n_groups, item2group):
    total_ndcg = 0.0
    total_rr = 0.0
    n_user = 0
    rank_group = np.array([0.0, 0.0])
    count_group = np.array([0.0, 0.0])
    pop_group = np.array([0.0, 0.0])
    pop = 0

    for user_id, pos_items in user_test_relevance.items():    
        if user_id not in user_train_items: 
            continue

        n_user += 1

        train_items = user_train_items.get(user_id)

        _ndcg, _rr, _rank_group, _count_group, _pop_group, _pop = compute_metrics(model, user_id, train_items, pos_items, topK, n_groups, item2group)
        
        total_ndcg += _ndcg
        total_rr += _rr
        rank_group += _rank_group
        count_group += _count_group
        pop_group += _pop_group
        pop += _pop

    total_ndcg /= n_user
    total_rr  /= n_user
    rank_group /= (n_user*topK)
    count_group /= n_user 
    pop_group /= (n_user*topK)
    pop /= (n_user*topK)
    return total_ndcg, total_rr, rank_group, count_group, pop_group, pop

def evaluate_fairness(model, user_test_relevance, user_train_items, topK, n_groups, item2group, alpha):
    total_ndcg = 0.0
    total_rr = 0.0
    rank_group = np.array([0.0, 0.0])
    count_group = np.array([0.0, 0.0])
    pop_group = np.array([0.0, 0.0])
    pop = 0.0
    n_user = 0
    avg_time = 0.0

    for user_id, pos_items in user_test_relevance.items():    
        if user_id not in user_train_items: 
            continue

        n_user += 1

        train_items = user_train_items.get(user_id)
        
        _ndcg, _rr, _rank_group, _count_group, _pop_group, _pop, _time = compute_metrics_fairness(model, user_id, train_items, pos_items, topK, n_groups, item2group, alpha)

        total_ndcg += _ndcg
        total_rr += _rr
        rank_group += _rank_group
        count_group += _count_group
        pop_group += _pop_group
        pop += _pop
        avg_time += _time

    total_ndcg /= n_user
    total_rr  /= n_user
    rank_group /= (n_user*topK)
    count_group /= n_user
    pop_group /= (n_user*topK)
    pop /= (n_user*topK)
    avg_time /= n_user
    return total_ndcg, total_rr, rank_group, count_group, pop_group, pop, avg_time

def evaluate_ratings(model, user_train_items, test):    
    mse = 0.0
    n_users = 0
    for user_id, group in test.groupby(['user_id']):
        if user_id not in user_train_items: 
            continue

        n_users += 1
        n_items = len(group)

        user_ids = torch.LongTensor(np.repeat(user_id, n_items))
        item_ids = torch.LongTensor(group['item_id'].values)

        
        ratings = torch.FloatTensor(group['rating'].values)

        weights = torch.LongTensor(np.repeat(1, n_items))

        y_hat = model(user_ids, item_ids)       
        # compute the loss
        #loss = F.mse_loss(y_hat, ratings)
        mse += model.calculate_loss(y_hat, ratings, weights)
    return mse.item()/n_users


num_users, num_items = len(u_map), len(i_map)

train_ds = MFDataset(train)
train_dl = DataLoader(train_ds, batch_size=int(len(train_ds)/10), shuffle=True)

emb_size = 30
n_epochs = 200


#lr = [1e-6, 1e-4, 1e-2]
lr = [1e-2]
wd = [1e-6, 1e-4, 1e-2, 0, 1]

best_model = None
best_val_error = float('inf')

if(not load_model):
    for _lr in lr:
        for _wd in wd:

            model1 = MF(num_users, num_items, emb_size=emb_size)
            train_epocs(model1, train, epochs=n_epochs, lr=_lr, wd=_wd)
            #train_data_loader(model1, train_dl, n_epochs, _lr, _wd)

            #ndcg1, mrr1 = evaluate(model1, user_test_relevance, user_train_items, 5)

            mse1 = evaluate_ratings(model1, user_train_items, valid)
            print(f'{_lr:.6f}, {_wd:.6f}, {mse1:.4f}')
            if mse1 < best_val_error:
                best_model = model1
                best_val_error = mse1

            # model2 = MF(num_users, num_items, emb_size=emb_size)
            # weights = torch.FloatTensor(train['propensity'].values)
            # train_epocs(model2, train, epochs=n_epochs, lr=_lr, wd=_wd, weights=weights)
            # #train_data_loader(model2, train_dl, n_epochs, _lr, _wd, type='prop')

            # model3 = MF(num_users, num_items, emb_size=emb_size)
            # weights = torch.FloatTensor(train['popularity'].values)
            # train_epocs(model3, train, epochs=n_epochs, lr=_lr, wd=_wd, weights=weights)
            # #train_data_loader(model3, train_dl, n_epochs, _lr, _wd, type='pop')

            #ndcg1, mrr1 = evaluate(model1, user_test_relevance, user_train_items, 5)
            # ndcg2, mrr2 = evaluate(model2, user_test_relevance, user_train_items, 5)
            # ndcg3, mrr3 = evaluate(model3, user_test_relevance, user_train_items, 5)
            #print(_lr,_wd, ndcg1, mrr1, ndcg2, mrr2, ndcg3, mrr3)

            # mse1 = evaluate_ratings(model1, user_train_items, test)
            # mse2 = evaluate_ratings(model2, user_train_items, test)
            # mse3 = evaluate_ratings(model3, user_train_items, test)
            # print(f'{_lr:.3f}, {_wd:.3f}, {mse1:.3f}, {ndcg1:.3f}, {mrr1:.3f}, {mse2:.3f}, {ndcg2:.3f}, {mrr2:.3f}, {mse3:.3f}, {ndcg3:.3f}, {mrr3:.3f}')

    filename = 'mf_model.pkl'
    pickle.dump(best_model, open(filename, 'wb'))
else:
    filename = '/home/ramon/mf_model.pkl'
    best_model = pickle.load(open(filename, 'rb')) 

topK = 20
n_groups = 2

#alpha = 0.8
#top_percentage=0.5

unfair_file = open('unfair-ml-100k.out', 'a')
fair_file = open('fair-ml-100k.out', 'a')

out_file = open('ml-100k.out', 'a')

#total_ndcg, total_rr, rank_group, count_group = evaluate_fairness(best_model, test_user_relevant_items, user_train_items, topK, alpha)

for top_percentage in [0.1, 0.3, 0.5]:
    for alpha in [0.1, 0.2, 0.5, 0.8]:
        
        print(f'{top_percentage} {alpha}')

        top_train = train.groupby(['item_id']).agg(count=('user_id', 'count')).reset_index().sort_values(by=['count'], ascending=False)
        cuttoff = int(len(top_train)*top_percentage)
        item2group = {item_id : 1 if idx < cuttoff else 0  for idx, item_id  in enumerate(top_train['item_id'].values) }

        ndcg1, mrr1, rank_group1, count_group1, pop_group1, pop = evaluate(best_model, test_user_relevant_items, user_train_items, topK, n_groups, item2group)
        #print(ndcg1, mrr1, rank_group1, count_group1)
        #with open(f'{unfair_name}.out', 'w') as out_file:
        s = f'{top_percentage:.4f},{alpha:.4f},{ndcg1:.4f},{mrr1:.4f},{rank_group1[0]:.4f},{rank_group1[1]:.4f},{count_group1[0]:.4f},{count_group1[1]:.4f},{pop_group1[0]:.4f},{pop_group1[1]:.4f},{pop:.4f}\n'
        out_file.write(s)  

        # total_ndcg = 0.0
        # total_rr = 0.0
        # rank_group = np.array([0.0, 0.0])
        # count_group = np.array([0.0, 0.0])
        # pop_group = np.array([0.0, 0.0])
        # pop = 0.0
        # n_user = 0
        # avg_time = 0.0

        # for user_id, pos_items in test_user_relevant_items.items():    
        #     if user_id not in user_train_items: 
        #         continue

        #     n_user += 1

        #     train_items = user_train_items.get(user_id)
            
        #     _ndcg, _rr, _rank_group, _count_group, _pop_group, _pop, _time  = compute_metrics_fairness(best_model, user_id, train_items, pos_items, topK, n_groups, item2group, alpha)

        #     total_ndcg += _ndcg
        #     total_rr += _rr
        #     rank_group += _rank_group
        #     count_group += _count_group
        #     pop_group += _pop_group
        #     pop += _pop
        #     avg_time += _time

        # total_ndcg /= n_user
        # total_rr  /= n_user
        # rank_group /= (n_user*topK)
        # count_group /= n_user
        # pop_group /= (n_user*topK)
        # pop /= (n_user*topK)
        # avg_time /= n_user

        #print(total_ndcg, total_rr, rank_group, count_group, avg_time)

        total_ndcg, total_rr, rank_group, count_group, pop_group, pop, avg_time = evaluate_fairness(best_model, test_user_relevant_items, user_train_items, topK, n_groups, item2group, alpha)


        s = f'{top_percentage:.4f},{alpha:.4f},{total_ndcg:.4f},{total_rr:.4f},{rank_group[0]:.4f},{rank_group[1]:.4f},{count_group[0]:.4f},{count_group[1]:.4f},{pop_group[0]:.4f},{pop_group[1]:.4f},{pop:.4f},{avg_time:.4f}\n'
        out_file.write(s)
        out_file.write('\n')
        
fair_file.close()
unfair_file.close()
out_file.close()

# testar alpha = 1 e 0.9
# testar upper bound
# testar balancear numero recomendado entre grupos

# ver metrica "Popularity-Opportunity Bias in Collaborative Filtering"