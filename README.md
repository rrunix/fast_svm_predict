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

TODOs
=====

* Change API to ease create prediction functions that allow taking the model info as input
* Fallback to LIBSVM when the batch is small (threshold?)
* Write docs
* Add benchmark info

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


CPU

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

GPU

| name          |   size |   test_size |   N. features |   N. support vectors |        Libsvm |   fast_svm_predict |   improvement |
|:--------------|-------:|------------:|--------------:|---------------------:|--------------:|-------------------:|--------------:|
| VERY SMALL H  |    100 |           1 |            20 |                   71 |   6.24518e-05 |         0.0121213  |       -194.09 |
| VERY SMALL H  |    100 |          10 |            20 |                   71 |   9.55475e-05 |         0.0105539  |       -110.46 |
| VERY SMALL H  |    100 |          50 |            20 |                   71 |   0.000232559 |         0.00770049 |        -33.11 |
| VERY SMALL H  |    100 |         100 |            20 |                   71 |   0.000413187 |         0.00768424 |        -18.6  |
| SMALL H       |   1000 |           1 |            20 |                  347 |   7.76178e-05 |         0.0201447  |       -259.54 |
| SMALL H       |   1000 |          10 |            20 |                  347 |   0.000225349 |         0.0133659  |        -59.31 |
| SMALL H       |   1000 |         100 |            20 |                  347 |   0.00171771  |         0.00898836 |         -5.23 |
| SMALL H       |   1000 |         500 |            20 |                  347 |   0.00826019  |         0.00894821 |         -1.08 |
| SMALL H       |   1000 |        1000 |            20 |                  347 |   0.0159874   |         0.00888158 |          1.8  |
| MEDIUM H      |  10000 |          10 |            20 |                 3068 |   0.001438    |         0.0163638  |        -11.38 |
| MEDIUM H      |  10000 |         100 |            20 |                 3068 |   0.0139711   |         0.0114609  |          1.22 |
| MEDIUM H      |  10000 |        1000 |            20 |                 3068 |   0.13638     |         0.0112644  |         12.11 |
| MEDIUM H      |  10000 |        5000 |            20 |                 3068 |   0.687421    |         0.0115289  |         59.63 |
| MEDIUM H      |  10000 |       10000 |            20 |                 3068 |   1.37452     |         0.0118382  |        116.11 |
| BIG DATASET H | 100000 |         100 |            20 |                34673 |   0.156104    |         0.0172324  |          9.06 |
| BIG DATASET H | 100000 |        1000 |            20 |                34673 |   1.55555     |         0.0170809  |         91.07 |
| BIG DATASET H | 100000 |       10000 |            20 |                34673 |  15.638       |         0.0188169  |        831.06 |
| BIG DATASET H | 100000 |       50000 |            20 |                34673 |  76.8023      |         0.0209348  |       3668.65 |
| BIG DATASET H | 100000 |      100000 |            20 |                34673 | 153.534       |         0.0214762  |       7149.04 |
| VERY SMALL L  |    100 |           1 |             4 |                   25 |   6.25227e-05 |         0.00803405 |       -128.5  |
| VERY SMALL L  |    100 |          10 |             4 |                   25 |   7.00471e-05 |         0.00848009 |       -121.06 |
| VERY SMALL L  |    100 |          50 |             4 |                   25 |   0.000108885 |         0.00842376 |        -77.36 |
| VERY SMALL L  |    100 |         100 |             4 |                   25 |   0.000156031 |         0.00922553 |        -59.13 |
| SMALL L       |   1000 |           1 |             4 |                  318 |   7.26322e-05 |         0.0114562  |       -157.73 |
| SMALL L       |   1000 |          10 |             4 |                  318 |   0.00017734  |         0.00989645 |        -55.8  |
| SMALL L       |   1000 |         100 |             4 |                  318 |   0.00121239  |         0.00785264 |         -6.48 |
| SMALL L       |   1000 |         500 |             4 |                  318 |   0.0057493   |         0.0077374  |         -1.35 |
| SMALL L       |   1000 |        1000 |             4 |                  318 |   0.0111468   |         0.00762007 |          1.46 |
| MEDIUM L      |  10000 |          10 |             4 |                 1721 |   0.000679384 |         0.0101455  |        -14.93 |
| MEDIUM L      |  10000 |         100 |             4 |                 1721 |   0.00615546  |         0.00771747 |         -1.25 |
| MEDIUM L      |  10000 |        1000 |             4 |                 1721 |   0.0593397   |         0.00762058 |          7.79 |
| MEDIUM L      |  10000 |        5000 |             4 |                 1721 |   0.288604    |         0.00775758 |         37.2  |
| MEDIUM L      |  10000 |       10000 |             4 |                 1721 |   0.594485    |         0.00775156 |         76.69 |
| BIG DATASET L | 100000 |         100 |             4 |                28409 |   0.0953919   |         0.010376   |          9.19 |
| BIG DATASET L | 100000 |        1000 |             4 |                28409 |   0.944935    |         0.0103416  |         91.37 |
| BIG DATASET L | 100000 |       10000 |             4 |                28409 |   9.45899     |         0.0107094  |        883.24 |
| BIG DATASET L | 100000 |       50000 |             4 |                28409 |  47.8429      |         0.0108165  |       4423.13 |
| BIG DATASET L | 100000 |      100000 |             4 |                28409 |  94.4914      |         0.012065   |       7831.87 |
