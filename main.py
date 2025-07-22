from utils.alg import arguments_to_algorithm
from utils.run_tests import run_tests, print_results
from utils.seeding import seed_all
from utils.user_input import get_input_arguments

import time

if __name__ == '__main__':
    # Set the RNG seeds to the same value
    #seed_all(10)
    start_time = time.time()

    # Get input arguments
    args = get_input_arguments()
    algorithm = arguments_to_algorithm(args)

    # Run tests
    results = run_tests(algorithm, args["dataset"], args["use_gpu"])
    print_results(args["dataset"], results)
    
    print(time.time() - start_time)

