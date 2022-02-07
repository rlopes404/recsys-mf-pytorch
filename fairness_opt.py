import numpy as np
import gurobipy as gp
from gurobipy import GRB

class FairnessMF():
    def __init__(self, n_items, costs, n_groups, item2group, topK, alpha):
        self.n_items = n_items
        self.K = topK
        self.n_groups = n_groups
        self.item2group = item2group
        self.costs = {idx:cost for idx, cost in enumerate(costs)}
        self.alpha = alpha
        self.x = []

        # env = gp.Env(empty=True)
        # env.setParam("OutputFlag",0)
        # env.start()
        # m = gp.Model("mip1", env=env)

        self.model = gp.Model("recsys")
        self.model.Params.LogToConsole = 0
        
        self._create_model()


    def _constraint2(self, alpha):
        adv_group = 1
        adv_const = sum(self.x[i][k]*self.is_item_in_group(i, adv_group)*self.p_click(k) for i in range(self.n_items) for k in range(self.K))

        dis_group = 0
        dis_const = sum(self.x[i][k]*self.is_item_in_group(i, dis_group)*self.p_click(k) for i in range(self.n_items) for k in range(self.K))

        self.model.addConstr(dis_const >= alpha*adv_const, f'fairness_2')       

    def _create_model(self):

        for i in range(self.n_items):
            self.x.append([])
            for k in range(self.K):
                _cost = self.costs[i]*self.p_click(k)
                self.x[i].append(self.model.addVar(vtype=GRB.BINARY, obj=_cost, name=f'x[{i},{k}]'))

        self.model.ModelSense = GRB.MAXIMIZE

        for k in range(self.K):
            self.model.addConstr(sum(self.x[i][k] for i in range(self.n_items)) == 1, f'knapsack_{k}')

        for i in range(self.n_items):
            self.model.addConstr(sum(self.x[i][k] for k in range(self.K)) <= 1, f'upper_bound_item[{i}]')

        sum_pclick = np.sum([self.p_click(k) for k in range(self.K)])/self.n_groups
        
        rhs = self.alpha*sum_pclick
       
        for g in range(self.n_groups):
            self.model.addConstr(sum(self.x[i][k]*self.is_item_in_group(i, g)*self.p_click(k) for i in range(self.n_items) for k in range(self.K)) >= rhs, f'fairness[{g}]')       
        
        self.model.write('recsys-fairness.lp')

    def is_item_in_group(self, i, k):        
        try:
            return int(self.item2group[i] == k)
        except KeyError:
            return 0

    def p_click(self, k):
        return 1/np.log2(k+2)

    def get_fair_ranking(self):
        self.model.optimize()

        if self.model.status == GRB.OPTIMAL:
            solution = np.repeat(-1, self.K)
            for k in range(self.K):
                for i in range(self.n_items):
                    if self.x[i][k].X > 0.99:
                        solution[k] = i
            return solution
        else:
            return []