from algorithms.hopfield import Hopfield
from solvers_alg.KMPSolver import KMPSolver


class HopfieldAlgorithmSolver(KMPSolver):
    def __init__(self, runs, use_gpu):
        self.name = "Hopfield"
        self.solutionValue = 0
        self.selectedFacilities = []
        self.runs = runs
        self.use_gpu = use_gpu

    def getName(self):
        return self.name

    def solve(self, graph, n, k, solutions=None):
        instance = Hopfield(n, k, graph, self._use_gpu)
        result = instance.run(self._runs, solutions)

        self.selectedFacilities = result[0]
        self.solutionValue = result[1]

    def getSolutionValue(self):
        return self.solutionValue

    def getSelectedFacilites(self):
        return self.selectedFacilities