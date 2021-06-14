import pandas as pd
import numpy as np
import time
import math
import faiss
import copy
import networkx as nx
from nltk import ngrams
from pyemd import emd

class EMD():
    def __init__(self, variants, sample_count, distanceMatrix):
        self.variants = variants
        self.sample_count = sample_count
        self.distanceMatrix = distanceMatrix
        self.score = self.score()

    def score(self):
        sub_hist = (self.sample_count/self.sample_count.sum()).astype(np.float64)
        full_hist = (self.variants['count']/self.variants['count'].sum()).values.astype(np.float64)
        return emd(sub_hist,full_hist,self.distanceMatrix.astype(np.float64))

class KnolsBehavior():
    artificial_start_activities = '$$start$$'
    artificial_end_activities = '$$end$$'

    def __init__(self, variants, sample_count, bandwidth=.05):
        self.variants = variants
        self.sample_count = sample_count
        self.bandwidth = bandwidth
        self.activity_mapping = list(set(a for s in self.variants['seq'] for a in s)) + \
                                [self.artificial_end_activities, self.artificial_start_activities]
        self.undersampled, self.oversampled, self.trulysampled = self.score()

    def score(self):
        # Original behavior
        ob = self.extract_behavior(self.variants['count']).reshape(-1)
        ob_normalized = ob/self.variants['count'].sum()

        # Sampled behavior
        sb = self.extract_behavior(self.sample_count).reshape(-1)
        sb_normalized = sb/self.sample_count.sum()

        diff = sb_normalized - ob_normalized
        diff = diff[ob!=0]
        s = diff.shape[0]
        undersampled_ratio = diff[diff<-(self.bandwidth/2)].shape[0] / s
        oversampled_ratio = diff[diff>(self.bandwidth/2)].shape[0] / s
        trulysampled_ratio = diff[np.abs(diff)<self.bandwidth/2].shape[0] / s

        return undersampled_ratio, oversampled_ratio, trulysampled_ratio


    def extract_behavior(self, sample_count):
        seq = self.produce_seq(sample_count)
        map = self.activity_mapping
        pairs = [tuple([map.index(pair[0]), map.index(pair[1])]) for s in seq for pair in ngrams(s,2)]

        m = np.zeros([len(map), len(map)], dtype=np.int64)
        for x, y in pairs:
            m[x,y] += 1
        return m

    def produce_seq(self, count):
        seq = self.variants['seq'].values.repeat(count.astype(np.int64)).tolist()
        # Add artificial starts and ends otherwise we loose some behaviors
        seq = [[self.artificial_start_activities]+x+[self.artificial_end_activities] for x in seq]

        return seq

    def _sample(self, initial_count_rep):
        raise NotImplementedError

