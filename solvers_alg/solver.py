# Interface for different solver algorithms
from abc import ABC, abstractmethod


class Solver(ABC):
    @abstractmethod
    def getName(self):
        pass

    @abstractmethod
    def solve(self, problem):
        pass

    @abstractmethod
    def getSolutionValue(self):
        pass

    @abstractmethod
    def getSelectedFacilites(self):
        pass