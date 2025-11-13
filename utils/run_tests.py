import os
import time

import numpy as np
from tabulate import tabulate

from datasets import loader
from datasets.pmed import pmed_utils
from datasets.special import special_utils
from datasets.tsplib import tsplib_utils
from solvers.brute_solver import calculate_distance


def run_tests(algorithm, dataset, use_gpu):

    # Random-small
    if dataset == "1":
        print(f"\nTesting algorithm {algorithm.getName()} on dataset Random-Small")
        tests = loader.load_dataset(loader.RANDOM)
        return run_dataset_tests(algorithm, tests, use_gpu)
    elif dataset == "2":
        print(f"\nTesting algorithm {algorithm.getName()} on dataset Random-Large")
        tests = loader.load_dataset(loader.RANDOM_HUGE)
        return run_dataset_tests(algorithm, tests, use_gpu)
    elif dataset == "3":
        print(f"\nTesting algorithm {algorithm.getName()} on dataset USCA312")
        tests = loader.load_dataset(loader.USCA312)
        return run_dataset_tests(algorithm, tests, use_gpu)
    elif dataset == "4":
        print(f"\nTesting algorithm {algorithm.getName()} on dataset P-Median")
        # get current directory and then add the dataset to the path
        path = os.getcwd()
        pmed_path = os.path.join(path, "datasets")
        return pmed_utils.run_tests(pmed_path, algorithm, use_gpu)
    elif dataset == "5":
        print(f"\nTesting algorithm {algorithm.getName()} on dataset TSPLib")
        # get current directory and then add the dataset to the path
        path = os.getcwd()
        pmed_path = os.path.join(path, "datasets")
        return tsplib_utils.run_tests(pmed_path, algorithm, use_gpu)
    elif dataset == "6":
    	print(f"\nTesting algorithm {algorithm.getName()} on dataset Special")
    	path = os.getcwd()
    	special_path = os.path.join(path, "datasets")
    	return special_utils.run_tests(special_path, algorithm, use_gpu)

def run_dataset_tests(algorithm_instance, tests, use_gpu):
    results = []

    for n in tests.n_range():
        #i = 1
        for k in tests.k_range():
            #j = 1
            #k = 10
            test_list = tests.load_data(n, k, use_gpu)

            ratios = []
            minRatio = 100.000
            maxRatio = 0.000
            
            times = []
            minTime = 99999.999
            maxTime = 0.000
            
            for test_number, test_data in enumerate(test_list):
                #x = 1
                #x = x + 1
                #if x != 1:
                #    continue
                graph = test_data["graph"]
                optimal_distance = test_data["distance"]

                start_time = time.time()
                algorithm_instance.setN(n)
                algorithm_instance.setK(k)
                algorithm_instance.setGraph(graph)
                algorithm_instance.initialize()
                algorithm_instance.solve()
                facilities = algorithm_instance.getSelectedFacilities()
                end_time = time.time()
                elapsedTime = end_time - start_time
                times.append(elapsedTime)
         
                if elapsedTime > maxTime:
                    maxTime = elapsedTime
                if elapsedTime < minTime:
                    minTime = elapsedTime

                distance = calculate_distance(graph, facilities, n)
                approximationRatio = distance / optimal_distance
                ratios.append(distance / optimal_distance)
                
                if approximationRatio > maxRatio:
                    maxRatio = approximationRatio
                if approximationRatio < minRatio:
                    minRatio = approximationRatio

            average_ratio = sum(ratios) / len(ratios)
            average_time = sum(times) / len(times)
            standard_deviation_ratio = np.std(ratios, dtype=np.float64)
            standard_deviation_time = np.std(times, dtype=np.float64)
            results.append((n, k, minRatio, average_ratio, maxRatio, minTime, average_time, maxTime, standard_deviation_ratio, standard_deviation_time))

            print(f"n={n} k={k} Completed")
            
            #j = j + 1
            #if j != 1:
            #    continue
            #break
        #i = i + 1
        #if i == 1:
        #    break
        #break

    return results

def print_results(dataset, results):
    tabular_format = 'simple'  # change to 'github' for markdown
    #standard_headers = ["Test", "N", "K", "Ratio", "Total Time", "Total Distance"]
    standard_headers = ["N", "K", "Min Ratio", "Average Ratio", "Max Ratio", "Min Time", "Average Time", "Max Time", "Standard Deviation (Ratio)", "Standard Deviation (Time)"]
    average_headers = ["N", "K", "Min Ratio", "Average Ratio", "Max Ratio", "Min Time", "Average Time", "Max Time", "Standard Deviation (Ratio)", "Standard Deviation (Time)"]

    if dataset == "1" or dataset == "2" or dataset == "3":
        print(tabulate(results, average_headers, tablefmt=tabular_format, floatfmt=".3f"))
    else:
        print(tabulate(results, standard_headers, tablefmt=tabular_format, floatfmt=".3f"))

