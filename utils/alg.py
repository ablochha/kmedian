from solvers_alg import (AryaMultiSolver, CohenAddadMultiSolver,
                         CohenAddadSolver, DominguezAlgorithmSolver,
                         HaralampievAlgorithmSolver, HopfieldAlgorithmSolver,
                         HopfieldBestHalfMultiAlgorithmSolver,
                         HopfieldBestHalfSecondClosestAlgorithmSolver,
                         HopfieldBestHalfSingleSolver,
                         HopfieldExhaustiveAlgorithmSolver,
                         HopfieldOriginal2nkSolver, ILPAlgorithmSolver,
                         InterchangeAlgorithmSolver, LocalSearchSolver,
                         ZhuAlgorithmSolver)


def arguments_to_algorithm(args):
    # switch on input
    if args["algorithm"] == "1":
        runs = args["parameters"]["runs"]
        return HopfieldAlgorithmSolver(use_gpu=args["use_gpu"])
        
    elif args["algorithm"] == "2":
        temperature = args["parameters"]["temperature"]
        decay = args["parameters"]["decay"]
        epoch = args["parameters"]["epoch"]
        return HaralampievAlgorithmSolver(temperature=temperature, epoch_length=epoch, decay_interval=decay)
        
    elif args["algorithm"] == "3":
        max_time = args["parameters"]["max_time"]
        return LocalSearchSolver(max_time=max_time)
        
    elif args["algorithm"] == "4":
        max_time = args["parameters"]["max_time"]
        return ZhuAlgorithmSolver(max_time=max_time)
        
    elif args["algorithm"] == "5":
        max_time = args["parameters"]["max_time"]
        return AryaMultiSolver(max_time=max_time)
        
    elif args["algorithm"] == "6":
        max_time = args["parameters"]["max_time"]
        return CohenAddadSolver(max_time=max_time)
        
    elif args["algorithm"] == "7":
        max_time = args["parameters"]["max_time"]
        return CohenAddadMultiSolver(max_time=max_time)
        
    elif args["algorithm"] == "8":
        runs = args["parameters"]["runs"]
        return HopfieldOriginal2nkSolver(use_gpu=args["use_gpu"])
        
    elif args["algorithm"] == "9":
        runs = args["parameters"]["runs"]
        return HopfieldBestHalfSingleSolver(use_gpu=args["use_gpu"])
        
    elif args["algorithm"] == "10":
        runs = args["parameters"]["runs"]
        return HopfieldBestHalfMultiAlgorithmSolver(use_gpu=args["use_gpu"])
        
    elif args["algorithm"] == "11":
        runs = args["parameters"]["runs"]
        return HopfieldBestHalfSecondClosestAlgorithmSolver(use_gpu=args["use_gpu"])
        
    elif args["algorithm"] == "12":
        runs = args["parameters"]["runs"]
        return HopfieldExhaustiveAlgorithmSolver(use_gpu=args["use_gpu"])
        
    elif args["algorithm"] == "13":
        runs = args["parameters"]["runs"]
        return InterchangeAlgorithmSolver(use_gpu=args["use_gpu"])
        
    elif args["algorithm"] == "14":
        runs = args["parameters"]["runs"]
        return DominguezAlgorithmSolver(use_gpu=args["use_gpu"])

    return None
