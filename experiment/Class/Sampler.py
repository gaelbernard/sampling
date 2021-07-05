import pandas as pd
import numpy as np
import time
import math
import faiss
from pulp import *
import stopit
from sklearn.neighbors import kneighbors_graph
from fast_pagerank import pagerank
class _AbstractSampler():
    def __init__(self, variants, p, seed=None, distanceMatrix=None, signature=None, expectedOccReduction=False, max_sampling_time=600):
        self.variants = variants.copy()
        self.p = p
        self.seed = seed
        self.distanceMatrix = distanceMatrix
        self.signature = signature
        self.expectedOccReduction = expectedOccReduction
        self.time_sampling = 0
        self.timeout = False
        self.max_sampling_time = max_sampling_time
        assert self.p < self.variants['count'].sum()

        # Capacity of each trace (i.e., how many traces can be assigned to it)
        self.capacity = math.floor(self.variants['count'].sum()/self.p)

        # Perform subsampling
        self.subsample_count = self.sample()

    def randomlyReOrder(self):

        # The order of the distance matrix
        # or signature will influence the results
        # For reproducibility purpose, we manage the order
        np.random.seed(self.seed)
        self.variants['random'] = np.random.random(self.variants.shape[0])
        new_order = self.variants.sort_values(['count', 'random'], ascending=False).index
        self.variants = self.variants.loc[new_order,:].reset_index()
        if self.distanceMatrix is not None:
            self.distanceMatrix = self.distanceMatrix[new_order,:][:,new_order]
        if self.signature is not None:
            self.signature = self.signature[new_order,:]

    def sample(self):
        t = time.time()

        count = np.zeros(self.variants.shape[0])
        if self.expectedOccReduction:
            count = np.floor(((self.variants['count']/self.variants['count'].sum()))*self.p).values
            self.variants['count'] -= (count*self.capacity)
            self.variants.loc[self.variants['count']<0,'count'] = 0


        @stopit.threading_timeoutable(self.max_sampling_time)
        def start_sampling_with_timer():
            return self._sample(count)

        count_subsample = start_sampling_with_timer(timeout=self.max_sampling_time)

        if not isinstance(count_subsample, (np.ndarray)):
            self.timeout = True
            count_subsample = count
        else:
            if count_subsample.sum() != self.p:
                raise ValueError('The count_subsample is not summing to p')
        self.time_sampling = time.time() - t

        return count_subsample

    def _sample(self, initial_count_rep):
        raise NotImplementedError

class _VariantBased(_AbstractSampler):
    def __init__(self, variants, p, seed, expectedOccReduction=None, distanceMatrix=None, signature=None):
        if expectedOccReduction is False:
            raise ValueError('a variant-based sampling cannot be used for trace-based sampling without the EOR reduction')
        _AbstractSampler.__init__(self, variants, p, seed, expectedOccReduction=expectedOccReduction, distanceMatrix=distanceMatrix, signature=signature)

class _TraceBased(_AbstractSampler):
    def __init__(self, variants, p, seed, expectedOccReduction=None, distanceMatrix=None, signature=None):
        _AbstractSampler.__init__(self, variants, p, seed, expectedOccReduction=expectedOccReduction, distanceMatrix=distanceMatrix, signature=signature)


class RandomSampling(_TraceBased):
    def __init__(self, variants, p, seed, expectedOccReduction):
        _TraceBased.__init__(self, variants, p, seed, expectedOccReduction=expectedOccReduction)

    def _sample(self, initial_count_rep):
        i = self.variants.sample(n=int(self.p-initial_count_rep.sum()), replace=True, weights='count', random_state=self.seed).index
        return np.bincount(i.values, minlength=self.variants.shape[0]) + initial_count_rep

class BiasedSamplingVariant(_TraceBased):
    def __init__(self, variants, p, seed, expectedOccReduction):
        _TraceBased.__init__(self, variants, p, seed, expectedOccReduction=expectedOccReduction)

    def _sample(self, initial_count_rep):

        for i,row in self.variants.reset_index().iterrows():
            r = self.p - int(initial_count_rep.sum())
            initial_count_rep[i] = initial_count_rep[i] + min(row['count'], r)
            if r == 0:
                break
        return initial_count_rep

class SimilarityBased(_VariantBased):
    def __init__(self, variants, p, seed, expectedOccReduction, q=0.33):
        assert q < 1 and q>0
        self.q=q

        _VariantBased.__init__(self, variants, p, seed, expectedOccReduction=expectedOccReduction)

    def _sample(self, initial_count_rep):
        # Extract pairs
        pair = pd.DataFrame({
            'pair':[y for x in self.variants['seq'] for y in ['$$start$$']+x+['$$end$$']],
            'index':self.variants.index.repeat(self.variants['length']+2),
            'count':self.variants['count'].repeat(self.variants['length']+2)
        })
        pair = pair.groupby('pair').sum()
        pair = pair.drop(['$$start$$', '$$end$$'])
        often_treshold = pair['count'].quantile(1-self.q)
        rare_treshold = pair['count'].quantile(self.q)
        pair['point'] = 0
        pair.loc[pair['count']>=often_treshold,'point'] = 1
        pair.loc[pair['count']<=rare_treshold,'point'] = -1
        r = self.p - int(initial_count_rep.sum())
        points = pair.groupby('index')['point'].sum()
        points = self.variants.join(points)
        points['point'] = points['point'].fillna(0)


        # Normalize by length
        points['point'] /= points['length']
        initial_count_rep[points['point'].nlargest(r).index] += 1

        return initial_count_rep

class LinearProg(_TraceBased):
    def __init__(self, variants, p, seed, distanceMatrix, expectedOccReduction):
        _TraceBased.__init__(self, variants, p, seed, distanceMatrix=distanceMatrix, expectedOccReduction=expectedOccReduction)

    def _sample(self, initial_count_rep):
        int_dm = (self.distanceMatrix * 100).astype(int) # Make the float integer
        int_dm = int_dm.repeat(self.variants['count'], axis=0).repeat(self.variants['count'], axis=1)

        p_remaining = int(self.p-initial_count_rep.sum())

        #reps = np.arange(self.variants['count'].shape[0]).astype(int)
        traces = np.arange(self.variants['count'].sum()).astype(int)

        representative_binary = LpVariable.dicts('X', (traces), 0, 1, LpInteger)
        allocation_matrix = LpVariable.dicts('Y', (traces, traces), 0, 1, LpInteger)

        # Objective: minimize the cost of the allocation matrix
        prob = LpProblem('PMedian', LpMinimize)
        prob += sum(sum(int_dm[t,r] * allocation_matrix[t][r] for t in traces) for r in traces)

        # CONSTRAINTS:
        # 1. we should have only k representatives
        prob += lpSum([representative_binary[r] for r in traces]) == p_remaining

        # 1. Each trace should be represented exactly once
        for t in traces:
            prob += sum(allocation_matrix[t][r] for r in traces) == 1

        # 2. A representative journey could be 'representative' only if it is 'activated' (in representative_binary)
        for r in traces:
            for t in traces:
                prob += allocation_matrix[t][r] <= representative_binary[r]

        # 5. A representative should represents a fair fraction of the population
        for r in traces:
            prob += sum(allocation_matrix[t][r] for t in traces) <= math.ceil(self.variants['count'].sum()/p_remaining)

        print (prob.solve())
        reps = []
        for v in prob.variables():
            subV = v.name.split('_')
            if subV[0] == "X" and v.varValue == 1:
                reps.append(subV[1])
        o_index = np.arange(self.variants.shape[0]).repeat(self.variants['count'])

        return np.bincount(o_index[np.array(reps, dtype=int)], minlength=self.variants.shape[0])+initial_count_rep

class IterativeCminSum(_TraceBased):
    def __init__(self, variants, p, seed, distanceMatrix, expectedOccReduction):
        _TraceBased.__init__(self, variants, p, seed, distanceMatrix=distanceMatrix, expectedOccReduction=expectedOccReduction)

    def _sample(self, initial_count_rep):

        # We repeat on both axis so that each line/rows represent a trace
        # ...and not a variant
        m = self.distanceMatrix.repeat(self.variants['count'], axis=1)
        m = m.repeat(self.variants['count'], axis=0)
        o_index = np.arange(self.variants.shape[0]).repeat(self.variants['count'])

        # Keep track of the traces that are assigned (filtering)
        not_assigned = np.ones(m.shape[1], dtype=bool)

        # Keep track of the representative journeys
        output_count = []

        while len(output_count) != self.p-initial_count_rep.sum():
            # Retrieve traces not assigned
            index_not_assigned = np.where(not_assigned == True)[0]
            subset_m = m[:,not_assigned][not_assigned,:]
            if subset_m.shape[1]>self.capacity:
                # Look for the {self.capacity} smallest sum
                nsmallest_sum_index = np.argpartition(subset_m, self.capacity, axis=1)[:,:self.capacity]
                n_smallest_sum_value = np.take_along_axis(subset_m, nsmallest_sum_index, 1)

                # retrieve the index of the best one
                most_central_i = n_smallest_sum_value.sum(axis=1).argmin()

                # the {self.capacity} closest to best one are removed for future iterations
                not_assigned[index_not_assigned[nsmallest_sum_index[most_central_i]]] = False
            else:
                most_central_i = subset_m.sum(axis=1).argmin()

            output_count.append(o_index[index_not_assigned[most_central_i]])

        return np.bincount(output_count, minlength=self.variants.shape[0]) + initial_count_rep

class LogRank(_VariantBased):
    def __init__(self, variants, p, seed, signature, expectedOccReduction, n_neighbors=30):
        self.n_neighbors = n_neighbors
        _VariantBased.__init__(self, variants, p, seed, signature=signature, expectedOccReduction=expectedOccReduction)

    def _sample(self, initial_count_rep):
        p = int(self.p-initial_count_rep.sum())
        g = kneighbors_graph(self.signature, n_neighbors=min(self.n_neighbors, int(self.signature.shape[0]/2)), mode='distance')
        g /= g.max()
        g[g>0] = 1 - g[g>0]
        self.variants['pagerank'] = pagerank(g)
        self.variants['pagerank'] *= self.variants['count']
        output_count = self.variants.sort_values('pagerank', ascending=False).head(p).index.values
        return np.bincount(output_count, minlength=self.variants.shape[0]) + initial_count_rep


class IterativeCentralityWithRedundancyCheck(_VariantBased):
    def __init__(self, variants, p, seed, distanceMatrix, expectedOccReduction, minDist=.1):
        self.minDist = minDist
        _VariantBased.__init__(self, variants, p, seed, distanceMatrix=distanceMatrix, expectedOccReduction=expectedOccReduction)

    def _sample(self, initial_count_rep):

        centrality = self.distanceMatrix.repeat(self.variants['count'], axis=1).sum(axis=1)
        minDist = self.minDist

        while initial_count_rep.sum() != self.p:
            selected = initial_count_rep>0

            if selected.sum()>0:
                exceedingTreshold = self.distanceMatrix[:,selected].min(axis=1) >= minDist
            else:
                exceedingTreshold = np.ones(self.distanceMatrix.shape[0], dtype=bool)
            valid_index = np.where(exceedingTreshold==True)[0]

            if valid_index.size == 0:
                minDist -= minDist * 0.2
                print ('reducing the min Dist to ', minDist)
            else:
                initial_count_rep[valid_index[centrality[valid_index].argmin()]] += 1

        return initial_count_rep


class IterativeCminSumEuclidean(_TraceBased):
    def __init__(self, variants, p, seed, signature, expectedOccReduction):
        _TraceBased.__init__(self, variants, p, seed, expectedOccReduction=expectedOccReduction, signature=signature)

    def _sample(self, initial_count_rep):

        # We repeat so that each line is a trace and not a variant
        data = self.signature.repeat(self.variants['count'], axis=0)
        original_seq_index = np.arange(self.variants.shape[0]).repeat(self.variants['count'])
        not_assigned = np.ones(data.shape[0]).astype(bool)

        output_count = []
        while len(output_count) != self.p - initial_count_rep.sum():
            i_not_assigned = np.where(not_assigned==True)[0]
            ldata = data[not_assigned,:]
            index = faiss.IndexFlatL2(ldata.shape[1])   # build the index
            index.add(ldata)                  # add vectors to the index
            D, I = index.search(ldata, self.capacity)
            best_id = D.sum(axis=1).argmin()
            closest_to_best = I[best_id,:]

            output_count.append(original_seq_index[i_not_assigned[best_id]])
            not_assigned[i_not_assigned[closest_to_best]] = False

        return np.bincount(output_count, minlength=self.variants.shape[0]) + initial_count_rep
