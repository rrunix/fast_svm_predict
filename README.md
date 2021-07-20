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