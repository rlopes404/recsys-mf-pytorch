import numpy as np
import gurobipy as gp
from gurobipy import GRB

m = gp.Model("recsys")

y_hat = {
    0: 1.5,
    1: 2.5,
    2: 4.5
}

n_items = 3
K = 2

n_groups = 2
group = {
    0: 0,
    1: 0,
    2: 1
}

x = []
for i in range(n_items):
    x.append([])
    for k in range(K):
        x[i].append(m.addVar(vtype=GRB.BINARY, obj=y_hat[i], name=f'x[{i},{k}]'))

m.ModelSense = GRB.MAXIMIZE
m.addConstr(sum(x[i][k] for i in range(n_items) for k in range(K)) == K, "mochila")

for i in range(n_items):
    m.addConstr(sum(x[i][k] for k in range(K)) <= 1, f'mochila_item[{i}]')

def is_group(i,k):
    return int(group[i] == k)

def p_click(k):
    return 1/np.log(2+k)

sum_pclick = np.sum([p_click(k) for k in range(K)])/n_groups
alpha = 0.8
rhs = alpha*sum_pclick

for g in range(n_groups):
    m.addConstr(sum(x[i][k]*is_group(i,k)*p_click(k)  for i in range(n_items) for k in range(K)) >= rhs, f'fairness[{g}]')

m.write('recsys-fairness.lp')

m.optimize()

if m.status == GRB.OPTIMAL:
    for k in range(K):
        for i in range(n_items):
            if x[i][k].X > 0.99:
                print(f'{k}. {i}')
else:
    print('No solution')