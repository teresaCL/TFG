# This file has been modified by Teresa Córdoba Lillo in 2025.
# It is based on original work licensed under the Apache License, Version 2.0.
# Original source: https://github.com/pierlumanzu/s_mdeclust

import argparse
import sys
import os
import numpy as np

"""
Parses command-line arguments for running the S-MDEClust algorithm
"""
def get_args():

    parser = argparse.ArgumentParser(description='memetic and genetic algorithms for Global Multi-Objective Optimization')

    parser.add_argument('--dataset', type=str, help='dataset path (CSV file required)')

    parser.add_argument('--constraints', type=str, help='constraints path (JSON file required)')

    parser.add_argument('--K', type=int, help='number of clusters')

    parser.add_argument('--seed', type=int, help='seed for the pseudo-random number generator', default=None)

    parser.add_argument('--verbose', help='activate verbose', action='store_true', default=False)

    parser.add_argument('--title', type=str, help='directory title with results', default="")


    ####################################################
    ### S-MDEClust Parameters ###
    ####################################################

    parser.add_argument('--assignment', type=str, help='assignment option', default='exact', choices=['exact', 'greedy', 'greedy_rand', 'greedy_rand_penalty'])

    parser.add_argument('--mutation', help='use mutation operator', action='store_true', default=False)
    
    parser.add_argument('--P', type=int, help='size of the population', default=20)

    parser.add_argument('--Nmax', type=int, help='maximum number of consecutive evaluations without improvement of the best solution', default=5000)

    parser.add_argument('--max_iter', type=int, help='maximum number of iterations', default=np.inf)

    parser.add_argument('--tol_pop', type=float, help='population tolerance', default=1e-4)

    parser.add_argument('--Nmax_ls', type=int, help='maximum number of consecutive iterations without improvement of the best solution for the local search', default=1)

    parser.add_argument('--max_iter_ls', type=int, help='maximum number of iterations for the local search', default=np.inf)

    parser.add_argument('--tol_sol', type=float, help='tolerance to choose if a solution is better than an another one', default=1e-6)

    parser.add_argument('--F', help='F paramater used for the crossover operator', default='mdeclust')  # Available options: random, mdeclust, half_mdeclust, float/integer value in (0, 2)

    parser.add_argument('--alpha', type=float, help='alpha paramater used for the mutation operator', default=0.5)

    parser.add_argument('--solis', type=str, help='solis wets algorithm options', default='no', choices=['no', 'w_penalty', 'wo_penalty'])
    
    parser.add_argument('--apply_LS_all', help='apply or not local search to all individuals in the population', action='store_true', default=False)

    parser.add_argument('--restart', type=int, help='number of restarts', default=0)

    parser.add_argument('--decrease_pop_size_reset', help='decrease or not population size at restarts', action='store_true', default=False)

    parser.add_argument('--crossover', type=str, help='crossover option', default='original', choices=['original', 'pbest_v1', 'pbest_v2'])

    parser.add_argument('--shade', help='use SHADE', action='store_true', default=False)

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

    assert args.P > 0
    assert args.Nmax > 0
    assert args.max_iter > 0
    assert args.tol_pop > 0
    assert args.Nmax_ls > 0
    assert args.max_iter_ls > 0
    assert args.tol_sol > 0
    assert args.restart >= 0

    if type(args.F) == str:
        assert args.F in ['random', 'mdeclust', 'half_mdeclust']
    elif type(args.F) == float or type(args.F) == int:
        assert args.F > 0
    else:
        raise AssertionError
    
    assert 0 <= args.alpha <= 1

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
