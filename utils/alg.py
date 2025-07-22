from algorithms import algorithm_interface


def arguments_to_algorithm(args):
    # switch on input
    if args["algorithm"] == "1":
        runs = args["parameters"]["runs"]
        return algorithm_interface.HopfieldAlgorithm(runs, args["use_gpu"])
        
    elif args["algorithm"] == "2":
        runs = args["parameters"]["runs"]
        temperature = args["parameters"]["temperature"]
        decay = args["parameters"]["decay"]
        epoch = args["parameters"]["epoch"]
        return algorithm_interface.HaralampievAlgorithm(temperature, epoch, decay, runs)
        
    elif args["algorithm"] == "3":
        max_time = args["parameters"]["max_time"]
        return algorithm_interface.LocalSearchAlgorithm(max_time)
        
    elif args["algorithm"] == "4":
        max_time = args["parameters"]["max_time"]
        return algorithm_interface.ZhuAlgorithm(max_time)
        
    elif args["algorithm"] == "5":
        max_time = args["parameters"]["max_time"]
        return algorithm_interface.AryaMultiAlgorithm(max_time)
        
    elif args["algorithm"] == "6":
        max_time = args["parameters"]["max_time"]
        return algorithm_interface.CohenAddadAlgorithm(max_time)
        
    elif args["algorithm"] == "7":
        max_time = args["parameters"]["max_time"]
        return algorithm_interface.CohenAddadMultiAlgorithm(max_time)
        
    elif args["algorithm"] == "8":
        runs = args["parameters"]["runs"]
        return algorithm_interface.HopfieldOriginalAlgorithm(runs, args["use_gpu"])
        
    elif args["algorithm"] == "9":
        runs = args["parameters"]["runs"]
        return algorithm_interface.HopfieldBestHalfSingleAlgorithm(runs, args["use_gpu"])
        
    elif args["algorithm"] == "10":
        runs = args["parameters"]["runs"]
        return algorithm_interface.HopfieldBestHalfMultiAlgorithm(runs, args["use_gpu"])
        
    elif args["algorithm"] == "11":
        runs = args["parameters"]["runs"]
        return algorithm_interface.HopfieldBestHalfSecondClosestAlgorithm(runs, args["use_gpu"])
        
    elif args["algorithm"] == "12":
        runs = args["parameters"]["runs"]
        return algorithm_interface.HopfieldExhaustiveAlgorithm(runs, args["use_gpu"])
        
    elif args["algorithm"] == "13":
        runs = args["parameters"]["runs"]
        return algorithm_interface.InterchangeAlgorithm(runs, args["use_gpu"])
        
    elif args["algorithm"] == "14":
        runs = args["parameters"]["runs"]
        return algorithm_interface.DominguezAlgorithm(runs, args["use_gpu"])

    return None
