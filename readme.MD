
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
sampler = CminSampler(seqs, 3)
print (sampler.sample())
```
Output:
```python
[('a', 'b', 'c'), ('a', 'b', 'c'), ('a',)]
```
It is suggesting that these are the 3 best traces that summarizes the event logs initially composed of 6 traces.