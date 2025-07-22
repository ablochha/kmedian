import torch


class GraphBase:
    def get_distance(self, i, j):
        raise NotImplementedError


class CoordinateGraph:
    def __init__(self, x, y, use_gpu):
        self._x = torch.tensor(x)
        self._y = torch.tensor(y)

        # Fill the distance cache using a matrix version of abs(Xi - Xj) + abs(Yi - Yj).
        n = len(self._x)
        self._distances = torch.abs(torch.reshape(self._x, (n, 1)) - self._x) + torch.abs(torch.reshape(self._y, (n, 1)) - self._y)

        # Create the normalized distance cache.
        max_value = torch.max(self._distances)
        min_value = torch.min(self._distances)
        self._normalized_distances = (self._distances - min_value) / (max_value - min_value)

        # Place the normalized cache onto the GPU.
        if use_gpu:
            self._gpu_normalized_distances = self._normalized_distances.to(device="cuda", copy=True)

    def get_standard_distance(self, i, j):
        return self._distances[i, j]

    def get_normalized_distance(self, i, j):
        return self._normalized_distances[i, j]


class DistanceGraph:
    """
    This graph is used when we have a distance matrix but no x and y coordinates.
    """
    def __init__(self, distances, use_gpu):
        self._distances = torch.tensor(distances)

        # Create the normalized distance cache.
        max_value = torch.max(self._distances)
        min_value = torch.min(self._distances)
        self._normalized_distances = (self._distances - min_value) / (max_value - min_value)

        # Place the normalized cache onto the GPU.
        if use_gpu:
            self._gpu_normalized_distances = self._normalized_distances.to(device="cuda", copy=True)

    def get_standard_distance(self, i, j):
        return self._distances[i, j]

    def get_normalized_distance(self, i, j):
        return self._normalized_distances[i, j]


class RandomGraph:
    def __init__(self, n, use_gpu):
        self._x = torch.rand(size=(n,))
        self._y = torch.rand(size=(n,))

        # Fill the distance cache using a matrix version of abs(Xi - Xj) + abs(Yi - Yj).
        n = len(self._x)
        self._distances = torch.abs(torch.reshape(self._x, (n, 1)) - self._x) + torch.abs(torch.reshape(self._y, (n, 1)) - self._y)

        # Create the normalized distance cache.
        # The normalization formula is: norm = (value - min) / (max - min) but we can simplify because our min value
        # will always be 0
        max_value = torch.max(self._distances)
        min_value = torch.min(self._distances)
        self._normalized_distances = (self._distances - min_value) / (max_value - min_value)

        # Place the normalized cache onto the GPU.
        if use_gpu:
            self._gpu_normalized_distances = self._normalized_distances.to(device="cuda", copy=True)

    def get_standard_distance(self, i, j):
        return self._distances[i, j]

    def get_normalized_distance(self, i, j):
        return self._normalized_distances[i, j]
