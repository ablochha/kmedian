import random

import torch

from problems.KMProblem import KMProblem
from solvers.brute_solver import calculate_distance
from solvers_alg.KMPSolver import KMPSolver


class HopfieldSecondParallelSolver(KMPSolver):

    def __init__(self, use_gpu, seed=None, alpha=0, beta=1.0, gamma=0, max_iterations=1000):
        self._name = "Hopfield Second Parallel Solver"
        self._solutionValue = None
        self._selectedFacilities = []

        self._n = None
        self._matrix_n = None
        self._k = None
        self._graph = None

        self._rng = random.Random(seed)

        self._num_rows = None
        self._num_cols = None
        self._size = None

        self._client_inner_values = None
        self._facility_inner_values = None
        self._client_activation_values = None
        self._facility_activation_values = None

        self._distance_values = None  # D (n,n)

        self._active_facility_list = []

        self._use_gpu = use_gpu
        if use_gpu:
            self._device = "cuda" if torch.cuda.is_available() else None
            assert self._device is not None
        else:
            self._device = "cpu"

        self._math_row_indices = None
        self._k_indices = None

        self._alpha = alpha
        self._beta = beta
        self._gamma = gamma
        self._max_iterations = max_iterations

    def set_seed(self, seed: int):
        self._rng = random.Random(seed)

    def getName(self):
        return self._name

    def getSolutionValue(self):
        return self._solutionValue

    def getSelectedFacilities(self):
        return self._selectedFacilities

    def setN(self, n):
        self._n = n

    def setK(self, k):
        self._k = k

    def setGraph(self, graph):
        self._graph = graph

    def initialize(self, problem:KMProblem):
        self._n = problem.getN()
        self._k = problem.getK()
        self._graph = problem.getGraph()
        if self._graph is None or self._n is None or self._k is None:
            raise ValueError("Graph, n, and k must be set before calling initialize().")

        # Initialize distance values
        if self._use_gpu:
            self._distance_values = self._graph._gpu_normalized_distances
        else:
            self._distance_values = self._graph._normalized_distances.clone().detach()

        num_clients, num_facility_candidates = self._distance_values.shape

        self._matrix_n = num_clients

        self._num_rows = self._matrix_n
        self._num_cols = self._k
        self._size = (self._num_rows, self._num_cols)

        self._math_row_indices = torch.arange(0, self._matrix_n, device=self._device)
        self._k_indices = torch.arange(0, self._k, device=self._device)

        self._facility_inner_values = torch.zeros(size=self._size, device=self._device)
        self._client_inner_values = torch.zeros(size=self._size, device=self._device)
        self._facility_activation_values = torch.zeros(size=self._size, dtype=torch.int, device=self._device)
        self._client_activation_values = torch.zeros(size=self._size, dtype=torch.int, device=self._device)

    def solve(self, runNum=None, starter_facilities=None):
        self._initialize_per_run_arrays(starter_facilities)

        iterations = 0
        stabilized = False

        while (not stabilized) and iterations < self._max_iterations:
            prev_C = self._client_activation_values.clone()
            prev_F = self._facility_activation_values.clone()

            self._calculate_client_values()
            self._update_client()

            self._calculate_facility_values()
            self._update_facility()

            stabilized = (
                torch.equal(prev_C, self._client_activation_values) and
                torch.equal(prev_F, self._facility_activation_values)
            )
            iterations += 1

            tmp_selected_facilities, tmp_solution_value = self._calculate_facilities_and_distance()
            print("Iteration", iterations, "distance:", tmp_solution_value)
            print("Selected facilities:", tmp_selected_facilities)

        self._selectedFacilities, self._solutionValue = self._calculate_facilities_and_distance()
        print(f"Distance: {self._solutionValue}")

    def _initialize_per_run_arrays(self, starter_facilities):
        self._client_activation_values = torch.zeros(self._size, dtype=torch.int, device=self._device)
        self._facility_activation_values = torch.zeros(self._size, dtype=torch.int, device=self._device)

        self._active_facility_list = []

        if starter_facilities is not None and len(starter_facilities) > 0:
            initial_facilities = list(starter_facilities)
        else:
            initial_facilities = self._rng.sample([i for i in range(self._matrix_n)], k=self._k)

        for q, facility in enumerate(initial_facilities):
            self._facility_activation_values[facility, q] = 1
            self._active_facility_list.append(facility)

        self._calculate_client_values()
        self._update_client()
        self._calculate_facility_values()

        tmp_selected_facilities, tmp_solution_value = self._calculate_facilities_and_distance()
        print("Initial distance:", tmp_solution_value)

    def _compute_distance_terms(self):
        d_ci_q = self._distance_values[:, self._active_facility_list]

        D_minus_q = torch.empty(
            (self._matrix_n, self._k),
            dtype=self._distance_values.dtype,
            device=self._device
        )

        for q in range(self._k):
            other_centers = [self._active_facility_list[r] for r in range(self._k) if r != q]

            if len(other_centers) == 0:
                D_minus_q[:, q] = float("inf")
            else:
                D_minus_q[:, q] = torch.min(self._distance_values[:, other_centers], dim=1).values

        return self._distance_values, d_ci_q, D_minus_q

    def _build_client_costs(self, d_ci_q, D_minus_q):
        return torch.minimum(
            D_minus_q,
            (1.0 + self._alpha) * d_ci_q
        )

    def _calculate_client_values(self):
        _, d_ci_q, D_minus_q = self._compute_distance_terms()
        client_costs = self._build_client_costs(d_ci_q, D_minus_q)
        self._client_inner_values = client_costs

    def _update_client(self):
        self._client_activation_values = torch.zeros(self._size, dtype=torch.int, device=self._device)
        max_indices = torch.argmin(self._client_inner_values, dim=1)
        self._client_activation_values[self._math_row_indices, max_indices] = 1

    def _calculate_facility_values(self):
        D, _, D_minus_q = self._compute_distance_terms()

        self._facility_inner_values = torch.empty(
            (self._matrix_n, self._k),
            dtype=D.dtype,
            device=self._device
        )

        for q in range(self._k):
            clipped = torch.minimum(D, D_minus_q[:, q].unsqueeze(1))
            base_cost = clipped.sum(dim=0)

            d_other_at_j = D_minus_q[:, q]

            threshold = self._beta * (self._matrix_n / self._k) * base_cost

            delta = torch.where(
                d_other_at_j > threshold,
                torch.zeros_like(base_cost),
                self._gamma * d_other_at_j
            )

            facility_cost = base_cost + delta
            self._facility_inner_values[:, q] = facility_cost

    def _update_facility(self):
        self._facility_activation_values = torch.zeros(
            self._size, dtype=torch.int, device=self._device
        )
        chosen = []
        used = torch.zeros(self._matrix_n, dtype=torch.bool, device=self._device)

        for q in range(self._k):
            scores = self._facility_inner_values[:, q].clone()

            # forbid already selected nodes
            scores[used] = float("inf")

            j = torch.argmin(scores).item()

            chosen.append(j)
            used[j] = True
            self._facility_activation_values[j, q] = 1

        self._active_facility_list = chosen

    def _calculate_facilities_and_distance(self):
        selected_facilities = list(self._active_facility_list)
        selected_distance = calculate_distance(self._graph, selected_facilities, self._n)
        return selected_facilities, selected_distance
