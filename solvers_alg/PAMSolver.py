import torch

from problems.KMProblem import KMProblem
from solvers.brute_solver import calculate_distance
from solvers_alg.KMPSolver import KMPSolver


class PAMSolver(KMPSolver):
    def __init__(self, use_gpu=False):
        self._name = "PAM Solver"
        self._solutionValue = None
        self._selectedFacilities = []

        self._n = None
        self._k = None
        self._graph = None
        self._distance_values = None

        self._use_gpu = use_gpu
        if use_gpu:
            self._device = "cuda" if torch.cuda.is_available() else None
            assert self._device is not None
        else:
            self._device = "cpu"

    def initialize(self, problem: KMProblem):
        self._n = problem.getN()
        self._k = problem.getK()
        self._graph = problem.getGraph()
        if self._graph is None or self._n is None or self._k is None:
            raise ValueError("Graph, n, and k must be set before calling initialize().")

        if self._use_gpu:
            self._distance_values = 1 - self._graph._gpu_normalized_distances
        else:
            self._distance_values = (1 - self._graph._normalized_distances).clone().detach()

        if self._distance_values.ndim != 2:
            raise ValueError(
                f"Expected a 2D distance matrix, got shape {tuple(self._distance_values.shape)}."
            )

        num_clients, num_facility_candidates = self._distance_values.shape
        if num_clients != num_facility_candidates:
            raise ValueError(
                "PAMSolver expects a square distance matrix for k-median. "
                f"Got shape {tuple(self._distance_values.shape)}."
            )
        if self._k > num_facility_candidates:
            raise ValueError(
                f"k={self._k} cannot exceed number of facility candidates ({num_facility_candidates})."
            )

        if self._n != num_clients:
            print(
                "Warning: problem n does not match distance matrix size. "
                f"Using matrix size n={num_clients} for tensor ops "
                f"(problem reported n={self._n})."
            )
            self._n = num_clients

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

    def solve(self, runNum=None):
        self._selectedFacilities = self._warm_start_facilities_greedy_deterministic()
        self._solutionValue = calculate_distance(
            self._graph,
            self._selectedFacilities,
            self._n,
        )

    # Greedy BUILD initializer
    def _warm_start_facilities_greedy_deterministic(self) -> list[int]:
        # Work on CPU for simple deterministic indexing and masking.
        D = self._distance_values.detach().to("cpu")
        num_candidates = D.shape[1]

        # D stores similarity (1 - normalized distance), so we maximize totals.
        # First facility: argmax_f sum_i D[i, f]
        col_sums = D.sum(dim=0)
        first_facility = torch.argmax(col_sums).item()

        selected = [first_facility]
        selected_mask = torch.zeros(num_candidates, dtype=torch.bool)
        selected_mask[first_facility] = True

        # Maintain max similarity to selected set for each client i.
        current_max_sim = D[:, first_facility].clone()

        for _ in range(1, self._k):
            # For each candidate f, compute sum_i max(current_max_sim[i], D[i, f]).
            candidate_scores = torch.maximum(current_max_sim.unsqueeze(1), D).sum(dim=0)
            candidate_scores[selected_mask] = float("-inf")

            next_facility = torch.argmax(candidate_scores).item()
            selected.append(next_facility)
            selected_mask[next_facility] = True

            # Update maintained maximum similarities.
            current_max_sim = torch.maximum(current_max_sim, D[:, next_facility])

        return selected
