import json
import os
import pandas as pd
import pickle as pkl
from datetime import datetime

from args_utils_grasp import get_args, check_args, args_file_creation
from utils import create_groups
from s_graspclust import S_GRASPClust

# Retrieve and verify arguments
args = get_args()
check_args(args)

# Create a path for the results using the current date
date = datetime.now().strftime('%Y-%m-%d')
res_path = os.path.join('Results', date, args.dataset.split('/')[-1].split('.')[0], args.constraints.split('/')[-1].split('.')[0])
os.makedirs(res_path, exist_ok=True)

# File with the parameters used for the execution
args_file_creation(res_path, args)

# Read dataset and set the number of clusters
dataset = pd.read_csv(args.dataset, sep='\s+', engine='python', index_col=False, header=None, skiprows=1)
D = dataset.values
K = args.K

# Load constraints and create ML and CL groups
f_json = json.load(open(args.constraints, 'r'))
ML = [tuple(i) for i in f_json['ml']]
CL = [tuple(i) for i in f_json['cl']]
ML_groups, CL_groups = create_groups(D, ML, CL)

# Run the algorithm
s_grasp = S_GRASPClust(
    args.assignment,
    args.Nmax, args.max_iter,
    args.Nmax_ls, args.max_iter_ls,
    args.tol_sol, args.verbose
)
labels, centers, score, n_iter, n_iter_ls, elapsed_time, ended_no_improvement = s_grasp.run(D, K, ML, CL, args.seed, ML_groups, CL_groups)

# Save algorithm results
pkl.dump({
    'labels': labels, 'centers': centers, 'score': score,
    'n_iter': n_iter, 'n_iter_ls': n_iter_ls, 'elapsed_time': elapsed_time, 'ended_no_improvement': ended_no_improvement
}, open(os.path.join(res_path, '{}.pkl'.format(args.constraints.split('/')[-1].split('.')[0])), 'wb'))
