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