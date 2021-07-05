from experiment.Class.DataPrep import DataPreparation
from experiment.Class.Sampler import *
from experiment.Class.Eval import KnolsBehavior, EMD
from experiment.datasets import datasets

def collect_result(sampler, data, distance_matrix_needed=False, signature_needed=False):
    '''
    Given a sampler and the data, prepare a dictionary
    to report the various metrics (event logs time, size, EMD)
    :param sampler: instance of one of the sampler
    :param data: instance of the data preparation
    :param distance_matrix_needed: is the distance matrix needed by the sampling algorithm?
    :param signature_needed: is the signature needed by the sampling algorithm?
    :return: a dictionary of results
    '''
    results = data.facts.copy()

    if not distance_matrix_needed:
        results['time_build_distance_matrix'] = 0
    if not signature_needed:
        results['time_to_build_signature'] = 0
    results['expectedOccReduction'] = sampler.expectedOccReduction
    results['p'] = sampler.p
    results['seed'] = sampler.seed
    results['technique'] = type(sampler).__name__
    results['time_sampling'] = sampler.time_sampling
    results['timeout'] = sampler.timeout
    results['SUM_time_preprocessing'] = results['time_load_csv'] + results['time_extract_variants'] + results['time_build_distance_matrix'] + results['time_to_build_signature']  + results['time_sampling']

    if not sampler.timeout:
        s = time.time()
        # KnolsBehavior
        behavior = KnolsBehavior(data.variants, sampler.subsample_count)
        results['trulysampled'] = behavior.trulysampled
        results['undersampled'] = behavior.undersampled
        results['oversampled'] = behavior.oversampled
        results['time_knols_eval'] = time.time()-s

        s = time.time()
        # Measure EMD
        emd = EMD(data.variants, sampler.subsample_count, data.distanceMatrix, sampler.seed)
        results['emd'] = emd.score
        results['time_emd_eval'] = time.time()-s

    return results


if __name__ == '__main__':
    results = []

    for seed in [0,1,2,3]:
        for ds in datasets:

            data = DataPreparation(forceReload=False, **ds)
            data.randomlyReOrder(seed)

            for p in [5,10,25,50,75,100,150,200]:

                print (ds['name'])

                sampler = RandomSampling(data.variants, p, seed=seed, expectedOccReduction=False)
                results.append(collect_result(sampler, data))

                sampler = RandomSampling(data.variants, p, seed=seed, expectedOccReduction=True)
                results.append(collect_result(sampler, data))

                sampler = BiasedSamplingVariant(data.variants, p, seed=seed, expectedOccReduction=False)
                results.append(collect_result(sampler, data))

                sampler = SimilarityBased(data.variants, p, seed=seed, expectedOccReduction=True, q=0.2)
                results.append(collect_result(sampler, data))

                sampler = LogRank(data.variants, p, seed=seed, expectedOccReduction=True, signature=data.signature, n_neighbors=200)
                results.append(collect_result(sampler, data, signature_needed=True))

                sampler = IterativeCentralityWithRedundancyCheck(data.variants, p, seed=seed, expectedOccReduction=True, distanceMatrix=data.distanceMatrix)
                results.append(collect_result(sampler, data, distance_matrix_needed=True))

                # IterativeCminSum
                sampler = IterativeCminSum(data.variants, p, seed=seed, distanceMatrix=data.distanceMatrix, expectedOccReduction=False)
                results.append(collect_result(sampler, data, distance_matrix_needed=True))
                del sampler

                sampler = IterativeCminSum(data.variants, p, seed=seed, distanceMatrix=data.distanceMatrix, expectedOccReduction=True)
                results.append(collect_result(sampler, data, distance_matrix_needed=True))
                del sampler

                sampler = IterativeCminSumEuclidean(data.variants, p, seed=seed, signature=data.signature, expectedOccReduction=True)
                results.append(collect_result(sampler, data, signature_needed=True))
                del sampler

            pd.DataFrame(results).to_csv('experiment/results.csv')
            del data

