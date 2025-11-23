import os
import time

from ExperimentManager import ExperimentManager
from reader.KMPJSONCoordinateReader import KMPJSONCoordinateReader
from reader.KMPJSONDistanceReader import KMPJSONDistanceReader
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
from utils.user_input import get_input_arguments

# -------------------------------
# 🔹 Utility functions
# -------------------------------

def get_data_path(key):
    """Return the full path to a dataset folder based on its key, using paths.txt mappings."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    resources_dir = os.path.join(base_dir, 'resources')
    paths_file = os.path.join(resources_dir, 'paths.txt')

    # Read mappings from paths.txt
    paths = {}
    with open(paths_file, 'r') as f:
        for line in f:
            if '=' in line:
                k, v = line.strip().split('=', 1)
                paths[k.strip()] = v.strip()

    # Get the relative path for the given key
    relative_path = paths.get(str(key))
    if not relative_path:
        raise ValueError(f"No dataset path found for key {key}")

    return os.path.join(resources_dir, relative_path)

def load_problems(dataset_path, dataset_key, use_gpu):
    """
    Loads problem files for a given dataset key.
    - Nested directories for case 1,2,3 → use KMPJSONCoordinateReader
    - Flat files for case 4,5,6 → use KMPJSONDistanceReader
    """
    if dataset_key in ["1", "2", "3"]:
        # Nested coordinate datasets
        problems = {}

        for subdir in os.listdir(dataset_path):
            subdir_path = os.path.join(dataset_path, subdir)
            if not os.path.isdir(subdir_path):
                continue

            problems[subdir] = []
            for filename in os.listdir(subdir_path):
                if not filename.endswith(".json"):
                    continue
                file_path = os.path.join(subdir_path, filename)
                try:
                    reader = KMPJSONCoordinateReader()
                    if reader.canRead(file_path):
                        problem = reader.parse(file_path, use_gpu)
                        problems[subdir].append(problem)
                    else:
                        print(f"⚠️ Skipped {file_path}: cannot read")
                except Exception as e:
                    print(f"⚠️ Failed to read {file_path}: {e}")
        return problems
    elif dataset_key in ["4", "6"]:
        # Flat distance datasets
        problems = []

        for filename in os.listdir(dataset_path):
            if not filename.endswith(".json"):
                continue
            file_path = os.path.join(dataset_path, filename)
            try:
                reader = KMPJSONDistanceReader()
                if reader.canRead(file_path):
                    problem = reader.parse(file_path, use_gpu)
                    problems.append(problem)
                else:
                    print(f"⚠️ Skipped {file_path}: cannot read")
            except Exception as e:
                print(f"⚠️ Failed to read {file_path}: {e}")
        return problems
    elif dataset_key in ["5"]:
        # Flat coordinate datasets
        problems = []

        for filename in os.listdir(dataset_path):
            if not filename.endswith(".json"):
                continue
            file_path = os.path.join(dataset_path, filename)
            try:
                reader = KMPJSONCoordinateReader()
                if reader.canRead(file_path):
                    problem = reader.parse(file_path, use_gpu)
                    problems.append(problem)
                else:
                    print(f"⚠️ Skipped {file_path}: cannot read")
            except Exception as e:
                print(f"⚠️ Failed to read {file_path}: {e}")
        return problems

# -------------------------------
# 🔹 Main script
# -------------------------------


if __name__ == '__main__':

    # Get input arguments
    args = get_input_arguments()
    solver = None

    match args['algorithm']:
        case "1":
            solver = HopfieldAlgorithmSolver(use_gpu=args["use_gpu"])

        case "2":
            solver = HaralampievAlgorithmSolver(temperature=args["parameters"]["temperature"], epoch_length=args["parameters"]["epoch"], decay_interval=args["parameters"]["decay"])

        case "3":
            solver = LocalSearchSolver(max_time=args["parameters"]["max_time"])

        case "4":
            solver = ZhuAlgorithmSolver(max_time=args["parameters"]["max_time"])

        case "5":
            solver = AryaMultiSolver(max_time=args["parameters"]["max_time"])

        case "6":
            solver = CohenAddadSolver(max_time=args["parameters"]["max_time"])

        case "7":
            solver = CohenAddadMultiSolver(max_time=args["parameters"]["max_time"])

        case "8":
            solver = HopfieldOriginalSolver(use_gpu=args["use_gpu"])

        case "9":
            solver = HopfieldBestHalfSingleSolver(use_gpu=args["use_gpu"])

        case "10":
            solver = HopfieldBestHalfMultiSolver(use_gpu=args["use_gpu"])

        case "11":
            solver = HopfieldBestHalfSecondClosestAlgorithmSolver(use_gpu=args["use_gpu"])

        case "12":
            solver = HopfieldExhaustiveAlgorithmSolver(use_gpu=args["use_gpu"])

        case "13":
            solver = InterchangeAlgorithmSolver(use_gpu=args["use_gpu"])

        case "14":
            solver = DominguezAlgorithmSolver(use_gpu=args["use_gpu"])

    dataset_key = args["dataset"]
    dataset_path = get_data_path(dataset_key)

    match dataset_key:
        case "1": print(f"\nTesting algorithm {solver.getName()} on dataset Random-Small")

        case "2": print(f"\nTesting algorithm {solver.getName()} on dataset Random-Large")

        case "3": print(f"\nTesting algorithm {solver.getName()} on dataset USCA312")

        case "4": print(f"\nTesting algorithm {solver.getName()} on dataset P-Median")

        case "5": print(f"\nTesting algorithm {solver.getName()} on dataset TSPLib")

        case "6": print(f"\nTesting algorithm {solver.getName()} on dataset Special")

    problems = load_problems(dataset_path, dataset_key, args["use_gpu"])

    manager = ExperimentManager(problems, solver, args.get("runs", None))
    manager.run(dataset_key)
