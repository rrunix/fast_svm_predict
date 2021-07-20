from scipy.sparse.construct import rand
from sklearn.datasets import make_classification


DATASET_BIG_H = make_classification(n_samples=10**5, n_features=20, n_informative=3, random_state=42)
DATASET_MEDIUM_H = make_classification(n_samples=10**4, n_features=20, n_informative=3, random_state=42)
DATASET_SMALL_H = make_classification(n_samples=10**3, n_features=20, n_informative=3, random_state=42)
DATASET_VERY_SMALL_H = make_classification(n_samples=10**2, n_features=20, n_informative=3, random_state=42)

DATASET_BIG_L = make_classification(n_samples=10**5, n_features=4, n_informative=2, random_state=42)
DATASET_MEDIUM_L = make_classification(n_samples=10**4, n_features=4, n_informative=2, random_state=42)
DATASET_SMALL_L = make_classification(n_samples=10**3, n_features=4, n_informative=2, random_state=42)
DATASET_VERY_SMALL_L = make_classification(n_samples=10**2, n_features=4, n_informative=2, random_state=42)


DATASETS = (
    ("VERY SMALL H", DATASET_VERY_SMALL_H),
    ("SMALL H", DATASET_SMALL_H),
    ("MEDIUM H", DATASET_MEDIUM_H),
    ("BIG DATASET H", DATASET_BIG_H),
    ("VERY SMALL L", DATASET_VERY_SMALL_L),
    ("SMALL L", DATASET_SMALL_L),
    ("MEDIUM L", DATASET_MEDIUM_L),
    ("BIG DATASET L", DATASET_BIG_L),
)
    