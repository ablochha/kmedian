from algorithms.hopfield import Hopfield
from solvers_alg.KMPSolver import KMPSolver


class HopfieldAlgorithmSolver(KMPSolver):
    def __init__(self, runs, use_gpu, graph, n, k, solutions=None):
        self._name = "Hopfield"
        self._solutionValue = 0
        self._selectedFacilities = []
        self._runs = runs
        self._use_gpu = use_gpu
        self._instance = Hopfield(graph, n, k, use_gpu)

    def getName(self):
        return self._name

    def solve(self, graph, n, k, solutions=None):
        result = self._instance.run(self._runs, solutions)

        self.selectedFacilities = result[0]
        self.solutionValue = result[1]

    def getSolutionValue(self):
        return self._solutionValue

    def getSelectedFacilites(self):
        return self._selectedFacilities