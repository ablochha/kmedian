def get_input_arguments():
    args = {}

    args["algorithm"] = get_algorithm()
    args["parameters"] = get_parameters(args["algorithm"])
    args["dataset"] = get_datasets()
    args["use_gpu"] = get_gpu_setting()

    return args


def get_algorithm():
    print("Select Algorithm:")
    print("\t1: Modified Hopfield Algorithm")
    print("\t2: Haralampiev's Algorithm")
    print("\t3: Local Search Algorithm")
    print("\t4: Zhu's algorithm MRA")
    print("\t5: Arya Multi Swap")
    print("\t6: Cohen-Addad Local Search")
    print("\t7: Cohen-Addad's Multi Swap")
    print("\t8: 2nk Original Single Hopfield")
    print("\t9: 2nk Best Half Single Hopfield")
    print("\t10: 2nk Best Half Multi Hopfield")
    print("\t11: 2nk Best Half Second Closest Hopfield")
    print("\t12: 2nk Exhaustive Hopfield")
    print("\t13: Fast Interchange")
    print("\t14: Dominguez NAL")
    

    return input("Enter a value 1-15: ")


def get_parameters(algorithm_selection):
    parameters = {}
    
    # n + n^2 Hopfield parameters
    if algorithm_selection == "1":
        input_runs = input("\nEnter Number of Runs (Default is 1): ")
        if len(input_runs) == 0:
            print("Using default: 1 run")
            parameters["runs"] = 1
        else:
            parameters["runs"] = int(input_runs)
        return parameters
    
    # Haralampiev Parameters
    elif algorithm_selection == "2":
        input_runs = input("\nEnter Number of Runs (Default is 1): ")
        if len(input_runs) == 0:
            print("Using default: 1 run")
            parameters["runs"] = 1
        else:
            parameters["runs"] = int(input_runs)
        input_temp = input("\nEnter Starting Temperature (Default is 1.0): ")
        if len(input_temp) == 0:
            print("Using default: 1.0")
            parameters["temperature"] = 1.0
        else:
            parameters["temperature"] = float(input_temp)
        input_epoch = input("\nEnter Epoch Length integer. Leave empty to use the dataset n values: ")
        if len(input_epoch) == 0:
            print("Using default: n")
            parameters["epoch"] = None
        else:
            parameters["epoch"] = int(input_epoch)
        input_decay = input("\nEnter Decay Interval (Default is 3): ")
        if len(input_decay) == 0:
            print("Using default: 3")
            parameters["decay"] = 3
        else:
            parameters["decay"] = int(input_decay)

        return parameters
    
    # Local Search Parameters
    elif algorithm_selection == "3":
        max_time = input("\nEnter Maximum runtime in seconds (Default is 5): ")
        if len(max_time) == 0:
            print("Using default: 5")
            parameters["max_time"] = 5
        else:
            parameters["max_time"] = int(max_time)
        return parameters
    
    # Zhu's algorith MRA
    elif algorithm_selection == "4":
        max_time = input("\nEnter Maximum runtime in seconds (Default is 5): ")
        if len(max_time) == 0:
            print("Using default: 5")
            parameters["max_time"] = 5
        else:
            parameters["max_time"] = int(max_time)
        return parameters
        
    # Arya Multi Swap
    elif algorithm_selection == "5":
        max_time = input("\nEnter Maximum runtime in seconds (Default is 5): ")
        if len(max_time) == 0:
            print("Using default: 5")
            parameters["max_time"] = 5
        else:
            parameters["max_time"] = int(max_time)
        return parameters
        
    # Cohen-Addad Local Search
    elif algorithm_selection == "6":
        max_time = input("\nEnter Maximum runtime in seconds (Default is 5): ")
        if len(max_time) == 0:
            print("Using default: 5")
            parameters["max_time"] = 5
        else:
            parameters["max_time"] = int(max_time)
        return parameters
        
    # Cohen-Addad Multi Swap
    elif algorithm_selection == "7":
        max_time = input("\nEnter Maximum runtime in seconds (Default is 5): ")
        if len(max_time) == 0:
            print("Using default: 5")
            parameters["max_time"] = 5
        else:
            parameters["max_time"] = int(max_time)
        return parameters
        
    # 2nk Original Single Hopfield parameters
    if algorithm_selection == "8":
        input_runs = input("\nEnter Number of Runs (Default is 1): ")
        if len(input_runs) == 0:
            print("Using default: 1 run")
            parameters["runs"] = 1
        else:
            parameters["runs"] = int(input_runs)
        return parameters
        
    # 2nk Best Half Single Hopfield parameters
    if algorithm_selection == "9":
        input_runs = input("\nEnter Number of Runs (Default is 1): ")
        if len(input_runs) == 0:
            print("Using default: 1 run")
            parameters["runs"] = 1
        else:
            parameters["runs"] = int(input_runs)
        return parameters
        
    # 2nk Best Half Multi Hopfield parameters
    if algorithm_selection == "10":
        input_runs = input("\nEnter Number of Runs (Default is 1): ")
        if len(input_runs) == 0:
            print("Using default: 1 run")
            parameters["runs"] = 1
        else:
            parameters["runs"] = int(input_runs)
        return parameters
        
    # 2nk Best Half Single Second Closest Hopfield parameters
    if algorithm_selection == "11":
        input_runs = input("\nEnter Number of Runs (Default is 1): ")
        if len(input_runs) == 0:
            print("Using default: 1 run")
            parameters["runs"] = 1
        else:
            parameters["runs"] = int(input_runs)
        return parameters
        
    # 2nk Exhaustive Hopfield parameters
    if algorithm_selection == "12":
        input_runs = input("\nEnter Number of Runs (Default is 1): ")
        if len(input_runs) == 0:
            print("Using default: 1 run")
            parameters["runs"] = 1
        else:
            parameters["runs"] = int(input_runs)
        return parameters
        
    # Fast Interchange
    if algorithm_selection == "13":
        input_runs = input("\nEnter Number of Runs (Default is 1): ")
        if len(input_runs) == 0:
            print("Using default: 1 run")
            parameters["runs"] = 1
        else:
            parameters["runs"] = int(input_runs)
        return parameters
        
    # Dominguez NAL
    if algorithm_selection == "14":
        input_runs = input("\nEnter Number of Runs (Default is 1): ")
        if len(input_runs) == 0:
            print("Using default: 1 run")
            parameters["runs"] = 1
        else:
            parameters["runs"] = int(input_runs)
        return parameters

    return None


def get_datasets():
    print("\nSelect Dataset:")
    print("\t1: Random-Small")
    print("\t2: Random-Large")
    print("\t3: USCA312")
    print("\t4: OR-Library P-Median")
    print("\t5: TSPLib")
    print("\t6: Special")

    return input("Enter a value 1-6: ")


def get_gpu_setting():
    print("\nUse the GPU (requires CUDA)?")
    selection = input("Enter a value y/n: ")

    if selection.lower() == "y":
        return True
    else:
        return False
