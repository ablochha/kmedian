import csv
import os
import time

import numpy as np

from problems.KMProblem import KMProblem
from solvers.brute_solver import calculate_distance
from solvers_alg.AryaMultiSolver import AryaMultiSolver
from solvers_alg.CohenAddadMultiSolver import CohenAddadMultiSolver
from solvers_alg.CohenAddadSolver import CohenAddadSolver
from solvers_alg.DominguezAlgorithmSolver import DominguezAlgorithmSolver
from solvers_alg.HaralampievAlgorithmSolver import HaralampievAlgorithmSolver
from solvers_alg.HopfieldAlgorithmSolver import HopfieldAlgorithmSolver
from solvers_alg.HopfieldBestHalfMultiAlgorithmSolver import \
    HopfieldBestHalfMultiSolver
from solvers_alg.HopfieldBestHalfSecondClosestAlgorithmSolver import \
    HopfieldBestHalfSecondClosestAlgorithmSolver
from solvers_alg.HopfieldBestHalfSingleSolver import \
    HopfieldBestHalfSingleSolver
from solvers_alg.HopfieldExhaustiveAlgorithmSolver import \
    HopfieldExhaustiveAlgorithmSolver
from solvers_alg.HopfieldOriginal2nkSolver import HopfieldOriginalSolver
from solvers_alg.InterchangeAlgorithmSolver import InterchangeAlgorithmSolver
from solvers_alg.LocalSearchSolver import LocalSearchSolver
from solvers_alg.ZhuAlgorithmSolver import ZhuAlgorithmSolver


class ExperimentManager():
    def __init__(self, problems, solver, num_runs=None):
        self._problems = problems
        self._solver = solver
        if(num_runs == None):
            self._num_runs = 10
        else:
            self._num_runs = num_runs

    def run(self):
        if isinstance(self._problems, list):
            # Case 1: list of test sets
            results = self._run_from_list()
        elif isinstance(self._problems, dict):
            # Case 2: dict of directories (e.g., {"pmed": tests_object})
            results = self._run_from_directory_dict()
        else:
            raise TypeError("Did not pass a list of problems to Experiment Manage")

        # Save results
        self._save_results_to_csv(results)

    def _run_from_list(self):
        results = []
        for problem in self._problems:
            optimal_distance = problem.getOptimal()

            ratios = []
            minRatio = 100.000
            maxRatio = 0.000
            
            times = []
            minTime = 99999.999
            maxTime = 0.000

            self._solver.initialize(problem)

            for iteration in range(self._num_runs):
                start_time = time.time()
                self._solver.solve(iteration)
                facilities = self._solver.getSelectedFacilities()
                end_time = time.time()
                elapsedTime = end_time - start_time
                times.append(elapsedTime)

                if elapsedTime > maxTime:
                    maxTime = elapsedTime
                if elapsedTime < minTime:
                    minTime = elapsedTime

                distance = calculate_distance(problem.getGraph(), facilities, problem.getN())
                approximationRatio = distance / optimal_distance
                ratios.append(approximationRatio)
                
                if approximationRatio > maxRatio:
                    maxRatio = approximationRatio
                if approximationRatio < minRatio:
                    minRatio = approximationRatio

            average_ratio = sum(ratios) / len(ratios)
            average_time = sum(times) / len(times)
            standard_deviation_ratio = np.std(ratios, dtype=np.float64)
            standard_deviation_time = np.std(times, dtype=np.float64)
            test_name = problem.getName()

            results.append((test_name, problem.getN(), problem.getK(), minRatio, average_ratio, maxRatio, minTime, average_time, maxTime, standard_deviation_ratio, standard_deviation_time))    
            #results.append((test_name, n, k, ratio, total_time, distance))
            print(f"n={problem.getN()} k={problem.getK()} Completed")

        return results
    

    def _run_from_directory_dict(self):
        results = []

        for key, problem_list in self._problems.items():

            for problem in problem_list:
                optimal_distance = problem.getOptimal()

                key_ratios = []
                minRatio = 100.000
                maxRatio = 0.000

                key_times = []
                minTime = 99999.999
                maxTime = 0.000

                self._solver.initialize(problem)

                # Run the solver multiple times for stability
                for iteration in range(self._num_runs):
                    start_time = time.time()
                    self._solver.solve(iteration)
                    facilities = self._solver.getSelectedFacilities()
                    end_time = time.time()

                    elapsedTime = end_time - start_time
                    key_times.append(elapsedTime)

                    if elapsedTime > maxTime:
                        maxTime = elapsedTime
                    if elapsedTime < minTime:
                        minTime = elapsedTime

                    distance = calculate_distance(problem.getGraph(), facilities, problem.getN())
                    approximationRatio = distance / optimal_distance
                    key_ratios.append(approximationRatio)

                    if approximationRatio > maxRatio:
                        maxRatio = approximationRatio
                    if approximationRatio < minRatio:
                        minRatio = approximationRatio

            average_ratio = sum(key_ratios) / len(key_ratios)
            average_time = sum(key_times) / len(key_times)
            standard_deviation_ratio = np.std(key_ratios, dtype=np.float64)
            standard_deviation_time = np.std(key_times, dtype=np.float64)
            results.append((problem.getN(), problem.getK(), minRatio, average_ratio, maxRatio, minTime, average_time, maxTime, standard_deviation_ratio, standard_deviation_time))

            print(f"n={problem.getN()} k={problem.getK()} Completed")

        return results

    def _save_results_to_csv(self, results):
        # Detect if results contain names (list case) or not (directory case)
        # If first element has 11 values → includes name; if 10 → no name
        has_name = len(results[0]) == 11 if results else False

        if has_name:
            # Use first problem's name as part of filename
            first_name = results[0][0].replace(" ", "_")  # replace spaces with _
            output_filename = f"results_{first_name}.csv"
        else:
            # Use timestamp if no names
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_filename = f"results_{timestamp}.csv"

        # Get the directory where this file (ExperimentManager) is located
        base_dir = os.path.dirname(os.path.abspath(__file__))
        output_path = os.path.join(base_dir, output_filename)

        # Build appropriate header
        if has_name:
            header = [
                "Instance", "N", "K", "Min Ratio", "Average Ratio", "Max Ratio",
                "Min Time", "Average Time", "Max Time",
                "Standard Deviation (Ratio)", "Standard Deviation (Time)"
            ]
        else:
            header = [
                "N", "K", "Min Ratio", "Average Ratio", "Max Ratio",
                "Min Time", "Average Time", "Max Time",
                "Standard Deviation (Ratio)", "Standard Deviation (Time)"
            ]

        results.sort(key=lambda r: ((r[1], r[2]) if len(r) == 11 else (r[0], r[1])))

        # Write the CSV
        with open(output_path, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)

            for r in results:
                if has_name:
                    name, n, k, min_ratio, avg_ratio, max_ratio, min_time, avg_time, max_time, std_ratio, std_time = r
                    writer.writerow([
                        name, n, k,
                        f"{min_ratio:.3f}", f"{avg_ratio:.3f}", f"{max_ratio:.3f}",
                        f"{min_time:.3f}", f"{avg_time:.3f}", f"{max_time:.3f}",
                        f"{std_ratio:.3f}", f"{std_time:.3f}"
                    ])
                else:
                    n, k, min_ratio, avg_ratio, max_ratio, min_time, avg_time, max_time, std_ratio, std_time = r
                    writer.writerow([
                        n, k,
                        f"{min_ratio:.3f}", f"{avg_ratio:.3f}", f"{max_ratio:.3f}",
                        f"{min_time:.3f}", f"{avg_time:.3f}", f"{max_time:.3f}",
                        f"{std_ratio:.3f}", f"{std_time:.3f}"
                    ])

        print(f"\n✅ Results saved to {output_path}")