# This file has been modified by Teresa Córdoba Lillo in 2025.
# It is based on original work licensed under the Apache License, Version 2.0.
# Original source: https://github.com/pierlumanzu/s_mdeclust

import json
import os
import pandas as pd
import pickle as pkl
from datetime import datetime
from args_utils_smdeclust import get_args, check_args, args_file_creation
from utils import create_groups
from s_mdeclust import S_MDEClust
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import os

# Retrieve and verify arguments
args = get_args()
check_args(args)

# Create a path for the results using the current date and the provided title
date = datetime.now().strftime('%Y-%m-%d')
res_path = os.path.join('Results', f"{date}-{args.title}", args.dataset.split('/')[-1].split('.')[0], args.constraints.split('/')[-1].split('.')[0])
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

# Run the S-MDEClust algorithm
s_mde = S_MDEClust(
    args.assignment, args.mutation,
    args.P, args.Nmax, args.max_iter, args.tol_pop,
    args.Nmax_ls, args.max_iter_ls,
    args.tol_sol,
    args.F, args.crossover, args.alpha,
    args.solis,
    args.apply_LS_all,
    args.restart,
    args.decrease_pop_size_reset,
    args.shade,
    args.verbose
)
labels, centers, score, n_iter, n_ls, n_iter_ls, elapsed_time, is_pop_collapsed, best_scores, pop_diversity, worst_scores, mean_scores = s_mde.run(D, K, ML, CL, args.seed, ML_groups, CL_groups)

# Save algorithm results
pkl.dump({
    'labels': labels, 'centers': centers, 'score': score,
    'n_iter': n_iter, 'n_ls': n_ls, 'n_iter_ls': n_iter_ls, 'elapsed_time': elapsed_time,
    'is_pop_collapsed': is_pop_collapsed
}, open(os.path.join(res_path, '{}.pkl'.format(args.constraints.split('/')[-1].split('.')[0])), 'wb'))


# Generate plots showing the score and diversity evolution
iterations = list(range(len(best_scores))) # X-axis

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))

ax1.plot(iterations, best_scores, label='Best', color='blue')
ax1.plot(iterations, mean_scores, label='Average', color='orange')
ax1.plot(iterations, worst_scores, label='Worst', color='red')
ax1.set_xlabel('Iterations')
ax1.set_ylabel('Score')
ax1.set_title('Evolution of the score: best, average, and worst in the population')
ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x)}'))
ax1.legend()

ax2.plot(iterations, pop_diversity, color='green')
ax2.set_ylabel('Population diversity')
ax2.set_title('Evolution of population diversity')
ax2.set_xlabel('Iterations')
ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:.2f}'))
plt.tight_layout()

# Save plots
output_path = os.path.join(res_path, 'plots.jpg')
plt.savefig(output_path)

# Save plots data
output_path = os.path.join(res_path, 'scores.csv')
data = {
    'Best': best_scores,
    'Mean': mean_scores,
    'Worst': worst_scores,
    'Diversity': pop_diversity
}
df = pd.DataFrame(data)
df.to_csv(output_path, index=False)

