import argparse
import sys
import os
import numpy as np

"""
Parses command-line arguments for running the GRASP algorithm
"""
def get_args():

    parser = argparse.ArgumentParser(description='GRASP algorithm')

    parser.add_argument('--dataset', type=str, help='dataset path (CSV file required)')

    parser.add_argument('--constraints', type=str, help='constraints path (JSON file required)')

    parser.add_argument('--K', type=int, help='number of clusters', default=None)

    parser.add_argument('--seed', type=int, help='seed for the pseudo-random number generator', default=None)

    parser.add_argument('--verbose', help='Activate verbose', action='store_true', default=False)

    ####################################################
    ### GRASPClust Parameters ###
    ####################################################

    parser.add_argument('--assignment', type=str, help='assignment option', default='greedy', choices=['greedy', 'greedy_rand', 'greedy_rand_penalty'])

    parser.add_argument('--Nmax', type=int, help='maximum number of consecutive iterations without improvement of the best solution', default=5)

    parser.add_argument('--max_iter', type=int, help='maximum number of iterations', default=50)

    parser.add_argument('--Nmax_ls', type=int, help='maximum number of consecutive iterations without improvement of the best solution for the local search', default=1)

    parser.add_argument('--max_iter_ls', type=int, help='maximum number of iterations for the local search', default=np.inf)

    parser.add_argument('--tol_sol', type=float, help='tolerance to choose if a solution is better than an another one', default=1e-6)

    return parser.parse_args(sys.argv[1:])

"""
Validates input arguments to ensure they meet required conditions before execution.
"""
def check_args(args):
    
    assert os.path.exists(args.dataset)
    assert os.path.exists(args.constraints)

    assert args.K > 0

    if args.seed is not None:
        assert args.seed > 0

    assert args.Nmax > 0
    assert args.max_iter > 0
    assert args.Nmax_ls > 0
    assert args.max_iter_ls > 0
    assert args.tol_sol > 0

"""
Creates and writes a CSV file with the parsed command-line arguments.
"""
def args_file_creation(res_path, args):
    args_file = open(os.path.join(res_path, 'params.csv'), 'w')
    
    for key in args.__dict__.keys():
        if type(args.__dict__[key]) == float:
            args_file.write('{};{}\n'.format(key, str(round(args.__dict__[key], 10)).replace('.', ',')))
        else:
            args_file.write('{};{}\n'.format(key, args.__dict__[key]))
    
    args_file.close()
