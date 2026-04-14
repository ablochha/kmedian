import torch
import math

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

    def solve(self, runNum=None, use_fastpam1=True):
        medoids = self._warm_start_facilities_greedy_deterministic()
        if use_fastpam1:
            medoids, cost = self._pam_swap_fastpam1(medoids)
        else:
            medoids, cost = self._pam_swap(medoids)
        self._selectedFacilities = medoids
        self._solutionValue = cost

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
    
    def _get_original_distance_matrix_cpu(self):
        if self._use_gpu and hasattr(self._graph, "_gpu_distances"):
            return self._graph._gpu_distances.detach().to("cpu", dtype=torch.float64)
        return self._graph._distances.detach().to("cpu", dtype=torch.float64)

    def _compute_nearest_and_second_nearest(self, medoids):
        D = self._get_original_distance_matrix_cpu()
        medoid_tensor = torch.tensor(medoids, dtype=torch.long, device=D.device)
        medoid_distances = D[:, medoid_tensor]  # (n, k)

        nearest_vals, nearest_pos = torch.min(medoid_distances, dim=1)

        second_vals = torch.full_like(nearest_vals, float("inf"), dtype=D.dtype)
        if len(medoids) > 1:
            temp = medoid_distances.clone()
            row_idx = torch.arange(self._n, device=D.device)
            temp[row_idx, nearest_pos] = float("inf")
            second_vals = torch.min(temp, dim=1).values

        nearest_medoids = medoid_tensor[nearest_pos]
        return D, nearest_vals, second_vals, nearest_medoids

    def _fastpam1_best_swap(self, medoids, current_cost):
        D, nearest_vals, second_vals, nearest_medoids = self._compute_nearest_and_second_nearest(medoids)

        best_delta = 0.0
        best_swap = None

        medoid_to_pos = {m: idx for idx, m in enumerate(medoids)}
        medoid_set = set(medoids)
        non_medoids = [i for i in range(self._n) if i not in medoid_set]

        for o in non_medoids:
            d_to_o = D[:, o]
            delta_by_pos = torch.zeros(len(medoids), dtype=D.dtype, device=D.device)

            for i in range(self._n):
                current_nearest = nearest_vals[i].item()
                current_second = second_vals[i].item()
                nearest_medoid = int(nearest_medoids[i].item())
                d_io = d_to_o[i].item()

                if d_io < current_nearest:
                    common_delta = d_io - current_nearest
                    delta_by_pos += common_delta

                    nearest_pos = medoid_to_pos[nearest_medoid]
                    delta_by_pos[nearest_pos] += min(d_io, current_second) - d_io
                else:
                    nearest_pos = medoid_to_pos[nearest_medoid]
                    delta_by_pos[nearest_pos] += min(d_io, current_second) - current_nearest

            candidate_pos = torch.argmin(delta_by_pos).item()
            candidate_delta = delta_by_pos[candidate_pos].item()

            if candidate_delta < best_delta:
                best_delta = candidate_delta
                best_swap = (medoids[candidate_pos], o)

        if best_swap is None:
            return None, current_cost

        m, o = best_swap
        new_cost = current_cost + best_delta
        return best_swap, new_cost
    
    def _pam_swap(self, medoids):
        current_cost = calculate_distance(self._graph, medoids, self._n)

        while True:
            best_cost = current_cost
            best_swap = None

            medoid_set = set(medoids)
            non_medoids = [i for i in range(self._n) if i not in medoid_set]

            for m in medoids:
                for o in non_medoids:
                    candidate = medoids.copy()
                    candidate[candidate.index(m)] = o

                    cost = calculate_distance(self._graph, candidate, self._n)
                    if cost < best_cost:
                        best_cost = cost
                        best_swap = (m, o)

            if best_swap is None:
                break

            m, o = best_swap
            medoids[medoids.index(m)] = o
            current_cost = best_cost

        return medoids, current_cost

    def _pam_swap_fastpam1(self, medoids):
        medoids = medoids.copy()
        current_cost = calculate_distance(self._graph, medoids, self._n)

        while True:
            best_swap, estimated_cost = self._fastpam1_best_swap(medoids, current_cost)
            if best_swap is None:
                break

            m, o = best_swap
            medoids[medoids.index(m)] = o

            # Recompute exact cost after applying the swap for safety.
            current_cost = calculate_distance(self._graph, medoids, self._n)

            # Numerical guard: if estimated cost and exact cost differ slightly, trust exact cost.
            if not math.isclose(current_cost, estimated_cost, rel_tol=1e-9, abs_tol=1e-9):
                current_cost = calculate_distance(self._graph, medoids, self._n)

        return medoids, current_cost
