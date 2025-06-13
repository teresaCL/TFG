import json
import os
import pandas as pd
import pickle as pkl
from datetime import datetime

from args_utils_grasp import get_args, check_args, args_file_creation
from utils import create_groups
from s_graspclust import S_GRASPClust

# Obtener y comprobar argumentos
args = get_args()
check_args(args)

# Crear carpeta de resultados y archivo de parámetros
date = datetime.now().strftime('%Y-%m-%d')
res_path = os.path.join('Results', date, args.dataset.split('/')[-1].split('.')[0], args.constraints.split('/')[-1].split('.')[0])
os.makedirs(res_path, exist_ok=True)
args_file_creation(res_path, args)

# Leer el dataset
dataset = pd.read_csv(args.dataset, sep='\s+', engine='python', index_col=False, header=None, skiprows=1)
D = dataset.values

# Guardar el número de clusters
if args.K is not None:
    K = args.K
else:
    K = int(max(dataset.values[:, -1]) + 1)

# Guardar restricciones y crear grupos de restricciones ML y CL
f_json = json.load(open(args.constraints, 'r'))
ML = [tuple(i) for i in f_json['ml']]
CL = [tuple(i) for i in f_json['cl']]

ML_groups, CL_groups = create_groups(D, ML, CL)

# Ejecutar algoritmo
s_grasp = S_GRASPClust(
    args.assignment,
    args.Nmax, args.max_iter,
    args.Nmax_ls, args.max_iter_ls,
    args.tol_sol, args.verbose
)
labels, centers, score, n_iter, n_iter_ls, elapsed_time, ended_no_improvement = s_grasp.run(D, K, ML, CL, args.seed, ML_groups, CL_groups)

# Guardar resultados
pkl.dump({
    'labels': labels, 'centers': centers, 'score': score,
    'n_iter': n_iter, 'n_iter_ls': n_iter_ls, 'elapsed_time': elapsed_time, 'ended_no_improvement': ended_no_improvement
}, open(os.path.join(res_path, '{}.pkl'.format(args.constraints.split('/')[-1].split('.')[0])), 'wb'))
