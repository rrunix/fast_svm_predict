from fast_svm_predict import predict_fn
from ._datasets import DATASETS
from sklearn.svm import SVC
import numpy as np

import unittest


MODELS = [
    (name, SVC(probability=True, kernel='rbf').fit(X, y), (X, y)) for name, (X, y) in DATASETS
]


class TestSVMPredict(unittest.TestCase):

    def test_proba(self):
        for name, clf, (X, y) in MODELS:
            original_predictions = clf.predict_proba(X)[:, 0]
            new_predictions = np.ravel(predict_fn(clf, output_type='proba')(X))
            self.assertTrue(np.allclose(original_predictions, new_predictions,
                            atol=1e-4), f"Dataset {name} predict proba test case")

    def test_decision_function(self):
        for name, clf, (X, y) in MODELS:
            original_predictions = clf.decision_function(X)
            new_predictions = np.ravel(predict_fn(
                clf, output_type='decision_function')(X))
            self.assertTrue(np.allclose(original_predictions, new_predictions,
                            atol=1e-4), f"Dataset {name} decision function test case")

    def test_class(self):
        for name, clf, (X, y) in MODELS:
            original_predictions = clf.predict(X)
            new_predictions = np.ravel(predict_fn(clf, output_type='class')(X))
            self.assertTrue(np.allclose(original_predictions, new_predictions,
                            atol=1e-4), f"Dataset {name} predict class test case")


if __name__ == '__main__':
    unittest.main()
