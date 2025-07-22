import json
import os
import time

from solvers.brute_solver import calculate_distance
from utils.graph import DistanceGraph


def run_tests(dataset_folder_path, algorithm_instance, use_gpu):
    results = []
    # 40 test files in the pmed dataset
    for i in range(1, 3):
        filepath = os.path.join(dataset_folder_path, "special", "tests", f"special{i}.json")
        with open(filepath, 'r') as f:
            test_data = json.loads(f.read())

        n = test_data['n']
        k = test_data['k']
        distances = test_data['distances']
        graph = DistanceGraph(distances, use_gpu)
        optimal_distance = test_data['distance']

        start_time = time.time()
        facilities = algorithm_instance.run(graph, n, k)
        end_time = time.time()
        total_time = end_time - start_time

        distance = calculate_distance(graph, facilities, n)
        ratio = distance / optimal_distance
        test_name = f"special{i}"
        results.append((test_name, n, k, ratio, total_time, distance))
        print(f"n={n} k={k} Completed")

    return results
