"""
This script contains functions that will be used to consistently seed the random number generators.
"""

import numpy as np
import random
import torch


def seed_all(value=10):
    np.random.seed(value)
    random.seed(value)
    torch.manual_seed(value)
