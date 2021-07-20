Fast SVM predictions
====================

SVM prediction routines for a sklearn.svm.SVC classifier (previously trained). The prediction routines are implemented in JAX. It is currently implemented for the linear, polynomial and rbf kernels. The output can be the decision function value, the prediction or the probability (using the Plat Scaling method).


Code example:

```python
import timeit

from sklearn.datasets import make_classification
from sklearn.svm import SVC

from fast_svm_predict import predict_fn

X, y = make_classification(10**4, 20)
clf = SVC(kernel='rbf', probability=True)

clf.fit(X, y)

fast_predict = predict_fn(clf, output_type='class') # output_type can be class|proba|decision_function.

test_function_fast = lambda: fast_predict(X)
clf_function = lambda: clf.predict(X)

repeats = 10
print("Fast", timeit.Timer(test_function_fast).timeit(number=repeats) / repeats)
print("Current", timeit.Timer(clf_function).timeit(number=repeats) / repeats)

```

Benchmarks:
In-progress....