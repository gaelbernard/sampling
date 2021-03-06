# Sampling Event Logs
## Experiment
We couldn't show all the results in the paper due to space constraints. There are available in [plots/output/](plots/output/).

Here are the steps to reproduce the results: 
1. Download the datasets and update the paths in the file [datasets.py](datasets.py). These datasets are available at [https://data.4tu.nl/](https://data.4tu.nl/) (check the paper for the DOIs).
2. Run [main.py](main.py). It will take several days (tips: try setting the max nrows to 10000 in [Class/DataPrep.py](Class/DataPrep.py) to get quick but approximate results). The results will be stored in [results.csv](results.csv).
3. Run [plots/graph.py](plots/graph.py) to generate the graphs that are visible in the paper.

## Reproducibility
The experiment were run using the following config/libs:
- Computer: 16GB of RAM, 4~CPUs, and a processor speed of 2.8~GHz
- Python: Python 3.9
- PyEMD==0.5.1
- Faiss==1.7.1
- scikit-learn==0.24.2