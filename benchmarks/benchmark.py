import timeit
from functools import partial

from datasets import DATASETS
from sklearn.svm import SVC
from fast_svm_predict import predict_fn
import pandas as pd

REPEATS = 100

models = []

print("Creating models")

for name, (X, y) in DATASETS:
    clf = SVC(probability=True, kernel='rbf').fit(X, y)
    fast_predict = predict_fn(clf, output_type='class')
    
    models.append((name, clf, fast_predict, (X, y)))


def run(fn, X):
    fn(X)
    

data = []
print("Running benchmark")

for name, clf, fast_predict, (X, _) in models:
    original_time = timeit.Timer(partial(run, clf.predict, X)).timeit(number=REPEATS) / REPEATS
    new_time = timeit.Timer(partial(run, fast_predict, X)).timeit(number=REPEATS) / REPEATS
    
    data = {'name': name, 'size': X.shape[0], 'n_feats': X.shape[1], 'original_time': original_time, 'new_time': new_time}
    print(data)

data_df = pd.DataFrame(data)
print(data_df.to_markdown())