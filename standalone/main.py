
from standalone.CminSampler import CminSampler
seqs = [
    ['a','b','c'],
    ['a','b','c'],
    ['a','b','c'],
    ['a'],
    ['a'],
    ['a','b','b','c'],
]
sampler = CminSampler(3)
sampler.load_list(seqs)
print (sampler.sample())

import pandas as pd
df = pd.read_csv('../data/data_sample.csv', sep=';')
sampler = CminSampler(6)
sampler.load_df(df, 'Incident ID', 'IncidentActivity_Type')
print (sampler.sample())