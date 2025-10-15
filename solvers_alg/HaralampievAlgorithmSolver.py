import random

from algorithms.haralampiev import (HaralampievNetwork,
                                    calculate_temperature_decay)
from solvers.brute_solver import calculate_distance, get_facilities
from solvers_alg.KMPSolver import KMPSolver


# I noticed it was only used by this solver, so I assumed it is best fit here
class RandomChoiceAlgorithm(KMPSolver):
    """
    This class exists for two purposes:
    1. See if an approximation is doing well or just appearing to do well.
    2. This class runs quickly, so it can be used to test that datasets have been constructed properly.
    """

    def name(self):
        return "Random Choice"

    def run(self, graph, n, k, solution=None):
        selection = random.sample([i for i in range(n)], k=k)
        return selection

class HaralampievAlgorithmSolver(KMPSolver):
    def __init__(self, temperature, epoch_length, decay_interval, runs, graph, n, k, solution=None):
        self._name = "Haralampiev Network"
        self._solutionValue = 0
        self._selectedFacilities = []
        self._temperature = temperature
        self._epoch_length = epoch_length  # Set to none in order to dynamically get n at runtime
        self._alpha = calculate_temperature_decay(self.temperature, decay_interval)
        self._runs = runs
        self._instance = HaralampievNetwork(n, k, graph)

    def getName(self):
        return self._name
    
    def solve(self, graph, n, k):
        warm_instance = RandomChoiceAlgorithm()
        warm_solution = warm_instance.run(graph, n, k)
        self._instance.warm_start(warm_solution)
        # Epoch length can be variable so we allow a None assignment in order to signal that we should use n
        if self._epoch_length is None:
            epoch_length = n
        else:
            epoch_length = self._epoch_length

        best_facilities = None
        best_distance = None
        for _ in range(self._runs):
            self._instance.run(temperature=self._temperature, epoch_length=epoch_length, alpha=self._alpha)
            facilities = get_facilities(self._instance, n, k)

            current_distance = calculate_distance(graph, facilities, n)
            if best_distance is None or current_distance < best_distance:
                best_distance = current_distance
                best_facilities = facilities

            self._instance.reset()
            self._instance.warm_start(best_facilities)

        self._selectedFacilities = best_facilities
        self._solutionValue = best_distance
    
    def getSolutionValue(self):
        return self._solutionValue
    
    def getSelectedFacilites(self):
        return self._selectedFacilities