import csv
import os
import re
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

    def run(self, dataset_key):
        if dataset_key in ["4", "5"]:
            # Case 1: list of test sets
            results = self._run_from_list()
        elif dataset_key in ["1", "2", "3"]:
            # Case 2: dict of directories (e.g., {"pmed": tests_object})
            results = self._run_from_directory_dict()
        elif dataset_key == "6":
            results = self._run_special()
        else:
            raise TypeError("Did not pass a list of problems to Experiment Manage")

        # Save results
        self._save_results_to_csv(results, dataset_key)

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
    
    def _run_special(self):
        results = []

        for problem in self._problems:
            optimal_distance = problem.getOptimal()
            self._solver.initialize(problem)

            start_time = time.time()
            self._solver.solve()
            facilities = self._solver.getSelectedFacilities()
            end_time = time.time()
            total_time = end_time - start_time

            distance = calculate_distance(problem.getGraph(), facilities, problem.getN())
            ratio = distance / optimal_distance
            test_name = problem.getName()
            results.append((test_name, problem.getN(), problem.getK(), ratio, total_time, distance))
            print(f"n={problem.getN()} k={problem.getK()} Completed")

        return results


    def _save_results_to_csv(self, results, dataset_key):
        if dataset_key in ["4", "5", "6"]:
            # Original value
            full_name = results[0][0]
            # Keep only the letters at the start
            first_name = re.match(r"[a-zA-Z]+", full_name).group(0)
            # Replace spaces just in case (optional)
            first_name = first_name.replace(" ", "_")
            output_filename = f"results_{first_name}.csv"
            has_name = True
        elif dataset_key in ["1", "2", "3"]:
            # Use timestamp if no names
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_filename = f"results_{timestamp}.csv"
            has_name = False

        # Get the directory where this file (ExperimentManager) is located
        base_dir = os.path.dirname(os.path.abspath(__file__))
        output_path = os.path.join(base_dir, output_filename)

        # Build appropriate header
        if has_name and dataset_key in ["4", "5"]:
            header = [
                "Instance", "N", "K", "Min Ratio", "Average Ratio", "Max Ratio",
                "Min Time", "Average Time", "Max Time",
                "Standard Deviation (Ratio)", "Standard Deviation (Time)"
            ]
        elif dataset_key == "6":
            header = [
                "Instance", "N", "K", "Ratio", "Total Time", "Distance"
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
                if dataset_key in ["4", "5"]:
                    name, n, k, min_ratio, avg_ratio, max_ratio, min_time, avg_time, max_time, std_ratio, std_time = r
                    writer.writerow([
                        name, n, k,
                        f"{min_ratio:.3f}", f"{avg_ratio:.3f}", f"{max_ratio:.3f}",
                        f"{min_time:.3f}", f"{avg_time:.3f}", f"{max_time:.3f}",
                        f"{std_ratio:.3f}", f"{std_time:.3f}"
                    ])
                elif dataset_key == "6":
                    writer.writerow([
                        r[0], r[1], r[2], f"{r[3]:.3f}", f"{r[4]:.3f}", f"{r[5]:.3f}"
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