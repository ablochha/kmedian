from problems.Problem import Problem


class FLProblem(Problem):
    
    def __init__(self, graph, n, openCost, optimal):
        self._name = "Facility Location Problem"
        self._graph = graph
        self._n = n
        self._openCost = openCost
        self._optimal = optimal

    def getName(self):
        return self._name

    def getOptimal(self):
        return self._optimal

    def setOptimal(self, optimal):
        self._optimal = optimal

    def getN(self):
        return self._n

    def setN(self, n):
        self._n = n

    def getOpenCost(self):
        return self._openCost
    
    def setOpenCost(self, openCost):
        self._openCost = openCost

    def getGraph(self):
        return self._graph
    
    def setGraph(self, graph):
        self._graph = graph