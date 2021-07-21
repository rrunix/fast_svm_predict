import timeit
from functools import partial
import pickle
import os

import numpy as np
import pandas as pd
from fast_svm_predict import predict_fn
from sklearn.svm import SVC

from datasets import DATASETS

REPEATS = 10

STORE_MODELS = '_models_cache.pkl'


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
print("Running benchmark")

for name, clf, (X, _) in benchmark_tests:
    fast_predict = predict_fn(clf, output_type='class')
    original_time = timeit.Timer(partial(run, clf.predict, X)).timeit(number=REPEATS) / REPEATS
    new_time = timeit.Timer(partial(run, fast_predict, X)).timeit(number=REPEATS) / REPEATS
    
    n_svs = len(np.ravel(clf.dual_coef_))
    
    record = {'name': name, 'size': X.shape[0], 'n_feats': X.shape[1], 'n_svs': n_svs, 'original_time': original_time, 'new_time': new_time}
    data.append(record)

data_df = pd.DataFrame(data)
print(data_df.to_markdown())
