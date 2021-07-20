from scipy.sparse.construct import rand
from sklearn.datasets import make_classification


DATASET_BIG = make_classification(n_samples=10**5, n_features=20, n_informative=3, random_state=42)
DATASET_MEDIUM = make_classification(n_samples=10**4, n_features=20, n_informative=3, random_state=42)
DATASET_SMALL = make_classification(n_samples=10**3, n_features=20, n_informative=3, random_state=42)


DATASETS = (
    ("BIG SMALL", DATASET_SMALL),
    # ("BIG MEDIUM", DATASET_MEDIUM),
    # ("BIG DATASET", DATASET_BIG),
)
    