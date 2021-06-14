import pandas as pd
import os
import numpy as np
import editdistance
import time
from itertools import combinations
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD

'''

'''

class DataPreparation():
    ca = 'case:concept:name'
    ac = 'concept:name'
    ts = 'time:timestamp'
    if not os.path.isdir('../pickles'):
        os.mkdir('../pickles')

    def __init__(self, name, path, ca=ca, ac=ac, ts=ts, forceReload=False):
        self.name = name
        self.path = path
        self.ca = ca
        self.ac = ac
        self.ts = ts
        self.distanceMatrix, self.signature, self.variants, self.facts = [None]*4
        if forceReload:
            self.distanceMatrix, self.signature, self.variants, self.facts = self.load_from_csv()
            np.save('pickles/{}_dm.pickle.npy'.format(self.name), self.distanceMatrix)
            np.save('pickles/{}_sig.pickle.npy'.format(self.name), self.signature)
            self.variants.to_pickle('pickles/{}_var.pickle'.format(self.name))
            self.facts.to_pickle('pickles/{}_facts.pickle'.format(self.name))
        else:
            self.distanceMatrix = np.load('pickles/{}_dm.pickle.npy'.format(self.name))
            self.signature = np.load('pickles/{}_sig.pickle.npy'.format(self.name))
            self.variants = pd.read_pickle('pickles/{}_var.pickle'.format(self.name))
            self.facts = pd.read_pickle('pickles/{}_facts.pickle'.format(self.name))

    def load_from_csv(self):
        facts = {}

        # Load the CSV
        s = time.time()
        df = self.load_csv()
        facts['time_load_csv'] = time.time()-s

        # Extract variants
        s = time.time()
        variants = self.extract_variants(df)
        facts['time_extract_variants'] = time.time()-s

        # Build the distance matrix between variants
        s = time.time()
        distanceMatrix = self.buildDistanceMatrix(variants['seq'].tolist())
        facts['time_build_distance_matrix'] = time.time()-s

        s = time.time()
        signature = self.buildSignature(variants['seq'].tolist())
        facts['time_to_build_signature'] = time.time()-s

        # Extract more facts about the dataset
        # (for descriptive statistics purpose)
        facts['dataset'] = self.name
        facts['ds_n_variants'] = variants.shape[0]
        facts['ds_n_events'] = df.shape[0]
        facts['ds_n_unique_activity'] = df[self.ac].nunique()
        facts['ds_n_unique_trace'] = df[self.ca].nunique()
        facts['cov_top5_vars'] = variants.head(5)['count'].sum()/variants['count'].sum()
        facts['cov_top10_vars'] = variants.head(10)['count'].sum()/variants['count'].sum()
        facts['cov_top20_vars'] = variants.head(20)['count'].sum()/variants['count'].sum()
        facts['cov_top50_vars'] = variants.head(50)['count'].sum()/variants['count'].sum()
        facts['average_levenshtein'] = distanceMatrix.mean()
        facts = pd.Series(facts)

        return distanceMatrix, signature, variants, facts


    def load_csv(self):
        df = pd.read_csv(self.path, nrows=None, low_memory=False)
        df[self.ts] = pd.to_datetime(df[self.ts])
        df.sort_values([self.ca, self.ts], inplace=True)
        df = df[[self.ca,self.ac]]
        df = df.loc[df.notna().all(axis=1),:]
        return df

    def extract_variants(self, df):
        variants = df.groupby(self.ca)[self.ac].agg(list)\
            .value_counts().reset_index().rename({'index':'seq', self.ac:'count'}, axis=1)
        variants['length'] = variants['seq'].str.len()
        return variants

    def distance_function(self, x1, x2):
        return editdistance.eval(x1, x2) / max([len(x1), len(x2)])

    def buildDistanceMatrix(self, seq):
        m = np.zeros([len(seq), len(seq)])
        for x, y in combinations(range(0,len(seq)), 2):
            d = self.distance_function(seq[x], seq[y])
            m[x,y] = d
            m[y,x] = d
        for x in range(len(seq)):
            m[x,x] = 0
        return m.astype(np.float64)

    def buildSignature(self, seq):
        cv = CountVectorizer(ngram_range=(1,2), tokenizer=lambda doc: doc, lowercase=False, max_features=1024)
        data = cv.fit_transform([['$$START$$']+x+['$$END$$'] for x in seq])
        data = TruncatedSVD(min(64, int(data.shape[1]/2)+1)).fit_transform(data).astype(np.float32)
        return data




