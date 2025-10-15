from solvers_alg.solver import Solver


class FLSolver(Solver):
    def __init__(self):
        self.name = None
        self.solutionValue = 0
        self.selectedFacilities = []

    def getName(self):
        return self.name

    def solve(self, graph, n, k):
        raise NotImplementedError

    def getSolutionValue(self):
        return self.solutionValue

    def getSelectedFacilites(self):
        return self.selectedFacilities