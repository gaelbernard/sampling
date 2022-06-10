
# Sampling Event Logs
We aim to reduce the size of event logs (from Process Mining) to p-traces, while minimizing the earth movers' distance (EMD) from the unsampled original event log.

This Github complements our paper and is composed of two main parts: 
1. [Experiment](/experiment). Contains the code that was run to benchmark various approaches (results reported in the paper). It also contains some graphs that could not fit the paper.
1. [Standalone](/standalone). The standalone version offers a simple interface to apply our proposed iterative c-min sampler so that running our sampling technique is as easy as:
```python
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
```
Output:
```python
[0, 2, 4]
```
It is suggesting that the sequences with index 0, 2, and 4 are the 3 best traces that summarizes the event logs initially composed of 6 traces.


It is also possible to load a Dataframe as follow:
```python
from standalone.CminSampler import CminSampler
import pandas as pd
df = pd.read_csv('../data/data_sample.csv', sep=';')
sampler = CminSampler(6)
sampler.load_df(df, 'Incident ID', 'IncidentActivity_Type')

```

Output:
```python
['IM0000005', 'IM0000012', 'IM0000013', 'IM0000015', 'IM0000017', 'IM0000020']
```
Suggesting that these are the case IDs that best represent the original dataset
