from Class.DataPrep import DataPreparation
from Class.Sampler import *
from Class.Eval import KnolsBehavior, EMD
from datasets import datasets

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
        # KnolsBehavior
        behavior = KnolsBehavior(data.variants, sampler.subsample_count)
        results['trulysampled'] = behavior.trulysampled
        results['undersampled'] = behavior.undersampled
        results['oversampled'] = behavior.oversampled

        # Measure EMD
        emd = EMD(data.variants, sampler.subsample_count, data.distanceMatrix)
        results['emd'] = emd.score

    return results


if __name__ == '__main__':
    results = []


    for ds in datasets:
        data = DataPreparation(forceReload=True, **ds)
        continue
        for seed in range(5):
            for p in range(5,101,25):

                print (ds['name'])

                sampler = RandomSampling(data.variants, p, seed=5, expectedOccReduction=False)
                results.append(collect_result(sampler, data))

                sampler = IterativeCentralityWithRedundancyCheck(data.variants, p, seed=1, expectedOccReduction=True, distanceMatrix=data.distanceMatrix)
                results.append(collect_result(sampler, data))

                sampler = LogRank(data.variants, p, seed=1, expectedOccReduction=True, distanceMatrix=data.distanceMatrix)
                results.append(collect_result(sampler, data))

                sampler = SimilarityBased(data.variants, p, seed=1, expectedOccReduction=True)
                results.append(collect_result(sampler, data))

                sampler = VariantFrequency(data.variants, p, seed=1, expectedOccReduction=False)
                results.append(collect_result(sampler, data))

                sampler = RandomSampling(data.variants, p, seed=1, expectedOccReduction=True)
                results.append(collect_result(sampler, data))

                # IterativeCminSum
                sampler = IterativeCminSum(data.variants, p, seed=1, distanceMatrix=data.distanceMatrix, expectedOccReduction=False)
                results.append(collect_result(sampler, data, distance_matrix_needed=True))

                sampler = IterativeCminSum(data.variants, p, seed=1, distanceMatrix=data.distanceMatrix, expectedOccReduction=True)
                results.append(collect_result(sampler, data, distance_matrix_needed=True))

                sampler = IterativeCminSumEuclidean(data.variants, p, seed=1, signature=data.signature, expectedOccReduction=True)
                results.append(collect_result(sampler, data, signature_needed=True))

                pd.DataFrame(results).to_csv('results.csv')


