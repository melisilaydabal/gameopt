from abc import ABC, abstractmethod
import time


class Solver(ABC):
    def __init__(self):
        self.start_time = time.time_ns()
        self.end_time = time.time_ns()

    @abstractmethod
    def solve(self, arglist, data, players, players_sites, strategies, alg, alg_parameter, utility_model=None):
        pass

    def compute_time(self):
        self.end_time = time.time_ns()
        return self.end_time - self.start_time

    def line_search(self, arglist, data, players, strategies, alg, parameter_values):
        search_results = {}
        for value in parameter_values:
            search_results[value] = self.find_equilibria(arglist, data, players, strategies, alg, value)
        return search_results



