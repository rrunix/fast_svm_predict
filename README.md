Fast SVM predictions
====================

SVM prediction routines for a sklearn.svm.SVC classifier (previously trained). The prediction routines are implemented in JAX. Main features:
 * Supported kernels: Linear, polynomial and RBF.
 * Output modes: decision function, prediction (0 or 1) and probability (using the Plat Scaling method).
 * Multiclass is not currently supported.


Code example:

```python
import timeit

from sklearn.datasets import make_classification
from sklearn.svm import SVC

from fast_svm_predict import predict_fn

X, y = make_classification(10**4, 20)
clf = SVC(kernel='rbf', probability=True)

clf.fit(X, y)

# Create a fast prediction function for clf
# output_type can be class|proba|decision_function.
fast_predict = predict_fn(clf, output_type='class') 

test_function_fast = lambda: fast_predict(X)
clf_function = lambda: clf.predict(X)

repeats = 10
print("Fast", timeit.Timer(test_function_fast).timeit(number=repeats) / repeats)
print("Current", timeit.Timer(clf_function).timeit(number=repeats) / repeats)

```

Instalation
===========

First, clone the project:

```bash
git clone https://github.com/rrunix/fast_svm_predict.git
```

Alternatively, it can be downloaded using the "Download Zip" option from github.

Then, move inside the project folder:

```bash
cd fast_svm_predict
```

Finally, install the package:

```bash
python setup.py install
```

WIP: Upload the package to pypi.

Benchmarks
==========
In-progress....

Libsvm (scikit-learn implementation) vs fast_svm_predict benchmark. The datasets are generated using make_classification from sklearn.datasets. The SVM (rbf kernel) is trained with a dataset of size (size), and then, the prediction times are measure over a sample of size (test_size). The time measurements are the average over 10 runs. The improvement is calculated as the ratio libsvm / fast_svm_predict if libsvm > fast_svm_predict, and - fast_svm_predict / libsvm otherwise.


| Dataset              |   size |   test_size |   N. features |   N. support vectors |        Libsvm |   fast_svm_predict |   Improvement |
|:---------------------|-------:|------------:|--------------:|---------------------:|--------------:|-------------------:|--------------:|
| make_classification  |    100 |          10 |            20 |                   71 |   8.87895e-05 |         0.00725921 |        -81.76 |
| make_classification  |    100 |          50 |            20 |                   71 |   0.000225235 |         0.00931467 |        -41.36 |
| make_classification  |    100 |         100 |            20 |                   71 |   0.000403162 |         0.00983777 |        -24.4  |
| make_classification  |   1000 |          10 |            20 |                  347 |   0.000219602 |         0.00734786 |        -33.46 |
| make_classification  |   1000 |         100 |            20 |                  347 |   0.00171834  |         0.00744596 |         -4.33 |
| make_classification  |   1000 |         500 |            20 |                  347 |   0.00787228  |         0.00763076 |          1.03 |
| make_classification  |   1000 |        1000 |            20 |                  347 |   0.0157178   |         0.00769144 |          2.04 |
| make_classification  |  10000 |          10 |            20 |                 3068 |   0.00145492  |         0.00870538 |         -5.98 |
| make_classification  |  10000 |         100 |            20 |                 3068 |   0.0142145   |         0.0124236  |          1.14 |
| make_classification  |  10000 |        1000 |            20 |                 3068 |   0.138314    |         0.0155476  |          8.9  |
| make_classification  |  10000 |        5000 |            20 |                 3068 |   0.680432    |         0.0340907  |         19.96 |
| make_classification  |  10000 |       10000 |            20 |                 3068 |   1.36076     |         0.0564026  |         24.13 |
| make_classification  | 100000 |         100 |            20 |                34673 |   0.153987    |         0.0267248  |          5.76 |
| make_classification  | 100000 |        1000 |            20 |                34673 |   1.54036     |         0.0708721  |         21.73 |
| make_classification  | 100000 |       10000 |            20 |                34673 |  15.4438      |         0.566225   |         27.27 |
| make_classification  | 100000 |       50000 |            20 |                34673 |  76.8967      |         2.79502    |         27.51 |
| make_classification  | 100000 |      100000 |            20 |                34673 | 154.727       |         5.60326    |         27.61 |
| make_classification  |    100 |          10 |             4 |                   25 |   7.08533e-05 |         0.00421689 |        -59.52 |
| make_classification  |    100 |          50 |             4 |                   25 |   0.000106785 |         0.00427838 |        -40.07 |
| make_classification  |    100 |         100 |             4 |                   25 |   0.000156298 |         0.00425977 |        -27.25 |
| make_classification  |   1000 |          10 |             4 |                  318 |   0.000173481 |         0.00675876 |        -38.96 |
| make_classification  |   1000 |         100 |             4 |                  318 |   0.00119644  |         0.00691176 |         -5.78 |
| make_classification  |   1000 |         500 |             4 |                  318 |   0.0056378   |         0.00696773 |         -1.24 |
| make_classification  |   1000 |        1000 |             4 |                  318 |   0.0111269   |         0.00688652 |          1.62 |
| make_classification  |  10000 |          10 |             4 |                 1721 |   0.000646465 |         0.0077976  |        -12.06 |
| make_classification  |  10000 |         100 |             4 |                 1721 |   0.00602042  |         0.00947005 |         -1.57 |
| make_classification  |  10000 |        1000 |             4 |                 1721 |   0.0582433   |         0.010073   |          5.78 |
| make_classification  |  10000 |        5000 |             4 |                 1721 |   0.291171    |         0.0161469  |         18.03 |
| make_classification  |  10000 |       10000 |             4 |                 1721 |   0.57936     |         0.0228757  |         25.33 |
| make_classification  | 100000 |         100 |             4 |                28409 |   0.0952549   |         0.0152392  |          6.25 |
| make_classification  | 100000 |        1000 |             4 |                28409 |   0.944255    |         0.0351598  |         26.86 |
| make_classification  | 100000 |       10000 |             4 |                28409 |   9.45953     |         0.228862   |         41.33 |
| make_classification  | 100000 |       50000 |             4 |                28409 |  47.8352      |         1.11298    |         42.98 |
| make_classification  | 100000 |      100000 |             4 |                28409 |  94.6098      |         2.21929    |         42.63 |