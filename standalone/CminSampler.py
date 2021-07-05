from collections import Counter, OrderedDict
import numpy as np
from itertools import combinations
import editdistance
import math
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD
import faiss

class CminSampler:
    def __init__(self, seqs, p, max_variants_dm=1000):
        assert isinstance(seqs, list)
        assert p>0 and p<len(seqs)

        self.p = p
        self.max_variants_dm = max_variants_dm

        # variants is a dictionary where key=variant and value=count
        self.variants = OrderedDict(Counter([tuple(x) for x in seqs]))
        self.multiplicity = np.array(list(self.variants.values()))
        self.variant_seq = list(self.variants.keys())

        # capacity it the number of traces assigned to each rep'
        self.capacity = math.floor(self.multiplicity.sum()/self.p)

        # Expected occurrence reduction reduces the number of operations needed (see paper)
        self.eo_counts = np.floor(((self.multiplicity / self.multiplicity.sum())) * self.p)
        self.variants = OrderedDict({x:int(y-self.eo_counts[i]) for i,(x,y) in enumerate(self.variants.items())})

    def sample(self):
        # If number of variants is small enough, use the DM approach
        # i.e., build and work on cost matrix of edit distance
        if len(self.variants) <= self.max_variants_dm:
            sampler = self.samplingWithDM()

        # If the number of variants is too big, this approach would take too long
        # so we work in Euclidean space instead
        else:
            sampler = self.samplingWithEucl()

        return [y for i, x in enumerate(sampler) if x > 0 for y in [self.variant_seq[i]]*int(x)]

    def samplingWithDM(self):
        '''
        Apply the sampling on the DM matrix. Run in O^2
        Do not use if you have more than 2-3k variants (controlled by max_variants_dm)
        :return: count vector
        '''
        dm = self.buildDistanceMatrix()

        # We repeat on both axis so that each line/rows represent a trace
        # ...and not a variant
        m = dm.repeat(self.multiplicity, axis=1)
        m = m.repeat(self.multiplicity, axis=0)
        o_index = np.arange(len(self.variants)).repeat(self.multiplicity)

        # Keep track of the traces that are assigned (filtering)
        not_assigned = np.ones(m.shape[1], dtype=bool)

        # Keep track of the representative journeys
        output_count = []

        while len(output_count) != self.p-self.eo_counts.sum():
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

        return np.bincount(output_count, minlength=len(self.variants)) + self.eo_counts


    def samplingWithEucl(self):
        '''
        Run the sampling algorithm in the Euclidean space
        Leverage Faiss for fast KNN so we don't have to build a distance matrix
        :return: count vector
        '''
        data = self.buildSignature()
        data = data.repeat(self.multiplicity, axis=0)
        original_seq_index = np.arange(self.multiplicity.shape[0]).repeat(self.multiplicity)
        not_assigned = np.ones(data.shape[0]).astype(bool)

        output_count = []
        while len(output_count) != self.p - self.eo_counts.sum():
            i_not_assigned = np.where(not_assigned==True)[0]
            ldata = data[not_assigned,:]
            index = faiss.IndexFlatL2(ldata.shape[1])   # build the index
            index.add(ldata)                  # add vectors to the index
            D, I = index.search(ldata, self.capacity)
            best_id = D.sum(axis=1).argmin()
            closest_to_best = I[best_id,:]

            output_count.append(original_seq_index[i_not_assigned[best_id]])
            not_assigned[i_not_assigned[closest_to_best]] = False

        return np.bincount(output_count, minlength=self.multiplicity.shape[0]) + self.eo_counts



    def buildDistanceMatrix(self):
        m = np.zeros([len(self.variants), len(self.variants)])
        seq = list(self.variants.keys())

        d = lambda x1, x2: editdistance.eval(x1, x2) / max([len(x1), len(x2)])

        for x, y in combinations(range(0,len(seq)), 2):
            dist = d(seq[x], seq[y])
            m[x,y] = dist
            m[y,x] = dist
        for x in range(len(seq)):
            m[x,x] = 0
        return m.astype(np.float64)

    def buildSignature(self):
        cv = CountVectorizer(ngram_range=(1,2), tokenizer=lambda doc: doc, lowercase=False, max_features=1024)
        data = cv.fit_transform([['$$START$$']+list(x)+['$$END$$'] for x in list(self.variants.keys())])
        data = TruncatedSVD(min(64, int(data.shape[1]/2)+1)).fit_transform(data).astype(np.float32)
        return data


