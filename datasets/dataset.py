import json
import os

from utils.graph import CoordinateGraph


class Dataset:
    def __init__(self, n_start, n_end, n_step_size, k_start, k_end, k_step_size, name, path, num_tests):
        self._n_start = n_start
        self._n_end = n_end
        self._n_step_size = n_step_size
        self._k_start = k_start
        self._k_end = k_end
        self._k_step_size = k_step_size
        self._name = name
        self._path = path
        self._num_tests = num_tests

    def load_data(self, n, k, use_gpu):
        filename = f"n{n}k{k}.json"
        with open(os.path.join(self._path, filename)) as f:
            test_data = json.load(f)

            # replace the stored x and y arrays with graph objects
            for test in test_data:
                x = test.pop("x_values")
                y = test.pop("y_values")
                test["graph"] = CoordinateGraph(x, y, use_gpu)

            return test_data

    def load_single_test(self, n, k, test_number=0):
        # Simple check to make sure the test is within the bounds. May be better to check that it is in the range.
        assert self._n_start <= n <= self._n_end
        assert self._k_start <= k <= self._k_end
        assert 0 <= test_number < self._num_tests

        test_data = self.load_data(n, k)
        test = test_data[test_number]
        return test


    def n_range(self):
        return range(self._n_start, self._n_end + 1, self._n_step_size)

    def k_range(self):
        return range(self._k_start, self._k_end + 1, self._k_step_size)

    def print_information(self):
        print(f"{self._name} Dataset N={self._n_start}:{self._n_end} K={self._k_start}:{self._k_end} {self._num_tests} Tests")
