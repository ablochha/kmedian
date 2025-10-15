from solvers_alg.solver import Solver


class KMPSolver(Solver):
    def __init__(self):
        self.selectedFacilities = []

    def getName(self):
        raise NotImplementedError

    def solve(self, graph, n, k):
        raise NotImplementedError

    def getSolutionValue(self):
        raise NotImplementedError

    def getSelectedFacilites(self):
        raise NotImplementedError