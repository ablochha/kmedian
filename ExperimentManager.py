import csv
import os
import re
import time

import numpy as np

from solvers.brute_solver import calculate_distance, calculate_radius
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
from solvers_alg.HopfieldOriginal2nkCKMPSolver import \
    HopfieldOriginal2nkCKMPSolver
from solvers_alg.HopfieldOriginal2nkSolver import HopfieldOriginalSolver
from solvers_alg.HopfieldParallelCKMSolver import HopfieldParallelCKMSolver
from solvers_alg.HopfieldParallelKCPSolver import HopfieldParallelKCPSolver
from solvers_alg.InterchangeAlgorithmSolver import InterchangeAlgorithmSolver
from solvers_alg.LocalSearchSolver import LocalSearchSolver
from solvers_alg.LocalSearchSolverKCenter import LocalSearchSolverKCenter
from solvers_alg.ZhuAlgorithmSolver import ZhuAlgorithmSolver


class ExperimentManager():
    def __init__(self, problems, solver, problem_family, num_runs=None):
        self._problems = problems
        self._solver = solver
        self._problem_family = problem_family
        self._latest_results = []
        if(num_runs == None):
            self._num_runs = 10
        else:
            self._num_runs = num_runs

    def run(self, dataset_key):
        self._latest_results = []

        if dataset_key in ["4", "5"]:
            # Case 1: list of test sets
            self._run_from_list(dataset_key)
        elif dataset_key in ["1", "2", "3"]:
            # Case 2: dict of directories (e.g., {"pmed": tests_object})
            self._run_from_directory_dict(dataset_key)
        elif dataset_key == "6":
            self._run_special(dataset_key)
        else:
            raise TypeError("Did not pass a list of problems to Experiment Manage")

        # Save results
        self._save_results_to_csv(self._latest_results, dataset_key)

    def _run_single_test(self, problem):
        """Helper function to run a single problem instance multiple times."""
        optimal_distance = problem.getOptimal()
        
        ratios = []
        minRatio = 100.000
        maxRatio = 0.000
        
        times = []
        minTime = 99999.999
        maxTime = 0.000
        
        # Initialize the solver for the *current* problem instance
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

            if self._problem_family == "1" or self._problem_family == "3":
                distance = calculate_distance(problem.getGraph(), facilities, problem.getN())
            elif self._problem_family == "2":
                distance = calculate_radius(problem.getGraph(), facilities)
            else:
                raise ValueError("Unknown problem family")
            
            approximationRatio = distance / optimal_distance
            ratios.append(approximationRatio)

            if approximationRatio > maxRatio:
                maxRatio = approximationRatio
            if approximationRatio < minRatio:
                minRatio = approximationRatio
                
        # Calculate stats for the run
        average_ratio = sum(ratios) / len(ratios)
        average_time = sum(times) / len(times)
        standard_deviation_ratio = np.std(ratios, dtype=np.float64)
        standard_deviation_time = np.std(times, dtype=np.float64)
        
        return (problem.getName(), problem.getN(), problem.getK(), 
                minRatio, average_ratio, maxRatio, minTime, average_time, 
                maxTime, standard_deviation_ratio, standard_deviation_time)


    def _run_from_list(self, dataset_key):
        for problem in self._problems:
            test_name = problem.getName()
            n = problem.getN()
            k = problem.getK()
            try:
                # 1. Run the test and get the results tuple
                # This calls the solver initialization once per problem instance
                result_tuple = self._run_single_test(problem)
                
                # 2. Append the result
                self._latest_results.append(result_tuple)
                
                print(f"n={n} k={k} Completed")
            except Exception as e:
                print(f"⚠️ Skipped {test_name} (n={n}, k={k}): {type(e).__name__}: {e}")
            finally:
                # 3. Explicitly delete the local reference to the problem object
                # This frees up the memory for the large graph/tensors sooner.
                del problem
    

    def _run_from_directory_dict(self, dataset_key):
        for key, problem_list in self._problems.items():
            for problem in problem_list:
                test_name = problem.getName()
                n = problem.getN()
                k = problem.getK()
                try:
                    # 1. Run the test and get the results tuple
                    # Unpack the result, ignoring the name since it's not needed for this format
                    _, n, k, minR, avgR, maxR, minT, avgT, maxT, stdR, stdT = self._run_single_test(problem)
                    
                    # 2. Append the result (excluding the name which is not used here)
                    self._latest_results.append((n, k, minR, avgR, maxR, minT, avgT, maxT, stdR, stdT))
                    
                    print(f"n={n} k={k} Completed")
                except Exception as e:
                    print(f"⚠️ Skipped {test_name} (n={n}, k={k}): {type(e).__name__}: {e}")
                finally:
                    # 3. Explicitly delete the local reference to the problem object
                    del problem
    
    def _run_special(self, dataset_key):
        for problem in self._problems:
            test_name = problem.getName()
            n = problem.getN()
            k = problem.getK()
            try:
                optimal_distance = problem.getOptimal()
                self._solver.initialize(problem)

                start_time = time.time()
                self._solver.solve()
                facilities = self._solver.getSelectedFacilities()
                end_time = time.time()
                total_time = end_time - start_time

                if self._problem_family == "1" or self._problem_family == "3":
                    distance = calculate_distance(problem.getGraph(), facilities, problem.getN())
                elif self._problem_family == "2":
                    distance = calculate_radius(problem.getGraph(), facilities)
                else:
                    raise ValueError("Unknown problem family")
                
                ratio = distance / optimal_distance
                self._latest_results.append((test_name, n, k, ratio, total_time, distance))
                print(f"n={n} k={k} Completed")
            except Exception as e:
                print(f"⚠️ Skipped {test_name} (n={n}, k={k}): {type(e).__name__}: {e}")
            finally:
                # Add explicit deletion here as well for consistency
                del problem


    def _save_results_to_csv(self, results, dataset_key, output_path=None, announce=True):
        if len(results) == 0:
            return

        if dataset_key in ["4", "5", "6"]:
            # Original value
            full_name = results[0][0]
            # Keep only the letters at the start
            match = re.match(r"[a-zA-Z]+", full_name)
            first_name = match.group(0) if match else "dataset"
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
        if output_path is None:
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

        if announce:
            print(f"\n✅ Results saved to {output_path}")
