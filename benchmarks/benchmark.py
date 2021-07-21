import timeit
from functools import partial
import pickle
import os
from datetime import datetime
import json

import numpy as np
from fast_svm_predict import predict_fn
from sklearn.svm import SVC

from datasets import DATASETS

REPEATS = 10

STORE_MODELS = '_models_cache.pkl'
STORE_BENCHMARK = 'benchmark_{}'.format(str(datetime.now()))


# Create model and prediction functions

print("Creating models")
if os.path.exists(STORE_MODELS):
    with open(STORE_MODELS, 'rb') as fin:
        benchmark_tests = pickle.load(fin)

else:
    benchmark_tests = []
    for name, (X, y) in DATASETS:
        clf = SVC(probability=True, kernel='rbf').fit(X, y)        
        benchmark_tests.append((name, clf, (X, y)))

    with open(STORE_MODELS, 'wb') as fout:
        pickle.dump(benchmark_tests, fout)


# Running benchmark
def run(fn, X):
    fn(X)

data = []
test_sizes_porc = (0.001, 0.01, 0.1, 0.5, 1)

print("Running benchmark")

for name, clf, (X, _) in benchmark_tests:
    fast_predict = predict_fn(clf, output_type='class')
    size = X.shape[0]
    
    for porc in test_sizes_porc:
        if porc * size < 1:
            continue
        
        test_size = int(porc * size)
        
        X_frac = X[:test_size]
    
        original_time = timeit.Timer(partial(run, clf.predict, X_frac)).timeit(number=REPEATS) / REPEATS
        new_time = timeit.Timer(partial(run, fast_predict, X_frac)).timeit(number=REPEATS) / REPEATS
        
        if original_time > new_time:
            improvement_factor = original_time / new_time
        else:
            improvement_factor = - new_time / original_time
        
        improvement_factor = round(improvement_factor, 2)
        
        n_svs = len(np.ravel(clf.dual_coef_))
        record = {'name': name, 'size': X.shape[0], 'test_size': test_size, 'N. features': X.shape[1], 'N. support vectors': n_svs, 'Libsvm': original_time, 'fast_svm_predict': new_time, "improvement": improvement_factor}
        data.append(record)


with open(STORE_BENCHMARK, 'w') as fout:
    json.dump(data, fout)
