"""
This script is responsible for loading a test case from the datasets.
"""
import os
from datasets.dataset import Dataset

# labels
RANDOM = "random"
RANDOM_LARGE = "random-large"
RANDOM_HUGE = "random-huge"
USCA312 = "usca312"

# paths
ROOT_PATH = os.getcwd()
RANDOM_PATH = os.path.join(ROOT_PATH, "datasets", "random")
RANDOM_LARGE_PATH = ROOT_PATH + "datasets/random-large/"
RANDOM_HUGE_PATH = os.path.join(ROOT_PATH, "datasets", "random-huge")
USCA312_PATH = os.path.join(ROOT_PATH, "datasets", "usca312", "tests")


def load_dataset(label):
    if label == RANDOM:
        return Dataset(n_start=20,
                       n_end=100,
                       n_step_size=10,
                       k_start=2,
                       k_end=10,
                       k_step_size=1,
                       name=RANDOM,
                       path=RANDOM_PATH,
                       num_tests=100)
    elif label == RANDOM_LARGE:
        return Dataset(n_start=300,
                       n_end=500,
                       n_step_size=100,
                       k_start=10,
                       k_end=50,
                       k_step_size=10,
                       name=RANDOM_LARGE,
                       path=RANDOM_LARGE_PATH,
                       num_tests=10)
    elif label == RANDOM_HUGE:
        return Dataset(n_start=500,
                       n_end=800,
                       n_step_size=100,
                       k_start=20,
                       k_end=50,
                       k_step_size=10,
                       name=RANDOM_HUGE,
                       path=RANDOM_HUGE_PATH,
                       num_tests=3)
    elif label == USCA312:
        return Dataset(n_start=30,
                       n_end=300,
                       n_step_size=10,
                       k_start=2,
                       k_end=10,
                       k_step_size=1,
                       name=USCA312,
                       path=USCA312_PATH,
                       num_tests=100)
    raise FileNotFoundError
