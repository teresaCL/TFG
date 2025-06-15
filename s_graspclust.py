import time
import random
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from gurobipy import Model, GRB

from blpkm import BLPKM

"""
Implements a GRASP algorithm for constrained clustering.
"""
class S_GRASPClust:
    """
    Initializes the model parameters.

    Parameters:
    -----------
        assignment (str): Type of assignment to use.
        Nmax (int): Maximum number of consecutive iterations without improvement.
        max_iter (int): Maximum number of iterations.
        Nmax_ls (int): Maximum number of consecutive local search iterations without improvement.
        max_iter_ls (int): Maximum number of local search iterations.
        tol_sol (float): Solution tolerance threshold. Indicates the required improvement in a solution's score for it to be replaced.
        verbose (bool): If True, displays results during the process.

    Returns:
    --------
        None
    """
    def __init__(self, assignment,
                 Nmax, max_iter,
                 Nmax_ls, max_iter_ls, 
                 tol_sol,verbose):
        
        self.__assignment = assignment 
        self.__Nmax = Nmax 
        self.__max_iter = max_iter  
        self.__tol_sol = tol_sol    
        self.__verbose = verbose    
        self.__ls = BLPKM(Nmax_ls, max_iter_ls)

    """
    Assigns each point in D to a cluster center from 'centers', while attempting to respect ML and CL constraints.

    Parameters:
    -----------
        D (np.ndarray): Data matrix of shape (n, d), where each row represents a point in d dimensions.
        distances (np.ndarray): Symmetric matrix of shape (n, n) containing squared distances between every pair of points in D.
        idx_centers (np.ndarray): Array of shape (k,) containing indices of the points in D that serve as centroids.
        ML_groups (list of sets): Each set represents a group of points that must be in the same cluster.  
        CL_groups (list of sets): The set at the i-th position contains the indices of the ML groups that cannot be in the same cluster as group i.

    Returns:
    --------
        labels (np.ndarray): Array containing the cluster label assigned to each object in D, with shape (n,), where n is the number of samples.  
    """
    def assign_objects(self, D, distances, idx_centers, ML_groups, CL_groups):
        # Initialize labels for each point to -1 (unassigned)
        labels = -1 * np.ones(len(D), dtype=int)
        
        # Initialize list that will store, for each centroid, the indices of the groups assigned to that centroid
        groups_to_centers = [[] for _ in range(len(idx_centers))]
        
        # Iterate through each ML group
        for idx_ml_gr, ml_gr in enumerate(ML_groups):
            ml_gr = np.array(list(ml_gr))   # indices of the points in D that are part of the group

            if self.__assignment=='greedy':
                # Indices of the clusters to which the points in the ml_gr group can be assigned
                possible_ks = np.array([idx_k for idx_k in range(len(idx_centers)) if not (set(groups_to_centers[idx_k]) & CL_groups[idx_ml_gr])])
                if len(possible_ks) == 0:
                    possible_ks = np.arange(len(idx_centers))
                
                # Compute the sum of squared distances from the group points to each possible centroid
                dist_sums = np.array([distances[ml_gr, idx_centers[pk]].sum() for pk in possible_ks])
                
                # Select the centroid with the lowest sum of distances
                ml_gr_k = possible_ks[np.argmin(dist_sums)] # idx in [0, len(idx_centers)-1]
                

            elif self.__assignment=='greedy_rand_penalty':
                # All centroids have a probability of being selected
                possible_ks = np.arange(len(idx_centers))

                # Compute the sum of squared distances from the group points to each possible centroid adding a penalty
                dist_sums = np.array([
                    distances[ml_gr, idx_centers[pk]].sum() +
                    (self.__greedy_penalty if len(set(groups_to_centers[pk]) & CL_groups[idx_ml_gr]) != 0 else 0)
                    for pk in possible_ks
                ])

                # Invert distances and normalize so they sum to 1 to obtain selection probabilities and select a k
                dist_sums[dist_sums == 0] = 1e-10   # avoid division by zero
                probs = 1 / dist_sums 
                probs /= probs.sum() 
                ml_gr_k = possible_ks[np.random.choice(len(dist_sums), p=probs)]

            elif self.__assignment=='greedy_rand':
                # Indices of the clusters to which the points in the ml_gr group can be assigned
                possible_ks = np.array([idx_k for idx_k in range(len(idx_centers)) if not (set(groups_to_centers[idx_k]) & CL_groups[idx_ml_gr])])
                if len(possible_ks) == 0:
                    possible_ks = np.arange(len(idx_centers))
                
                # Compute the sum of squared distances from the group points to each possible centroid
                dist_sums = np.array([distances[ml_gr, idx_centers[pk]].sum() for pk in possible_ks])

                # Invert distances and normalize so they sum to 1 to obtain selection probabilities and select a k
                dist_sums[dist_sums == 0] = 1e-10   # evitar división entre 0
                probs = 1 / dist_sums 
                probs /= probs.sum() 
                ml_gr_k = possible_ks[np.random.choice(len(dist_sums), p=probs)]

            # Assign the group to the corresponding centroid
            groups_to_centers[ml_gr_k].append(idx_ml_gr)
            
            # Update labels of the points in the ML group
            labels[ml_gr] = ml_gr_k
        
        return labels
        
    """
    Constructs a solution using a randomized greedy approach while attempting to respect constraints.

    Parameters:
    -----------
        D (np.ndarray): Data matrix of shape (n, d), where each row represents a point in d dimensions.
        distances (np.ndarray): Symmetric matrix of shape (n, n) containing squared distances between every pair of points in D.
        K (int): Number of clusters.
        ML_groups (list of sets): Each set represents a group of points that must be in the same cluster.  
        CL_groups (list of sets): The set at the i-th position contains the indices of the ML groups that cannot be in the same cluster as group i.

    Returns:
    --------
        centers (np.ndarray): Matrix of shape (K, d) representing the cluster centroids.
    """
    def greedy_randomized_sol(self, D, distances, K, ML_groups, CL_groups):
        idxs_points = list(range(D.shape[0])) # indices of the data points

        # List of indices of the data points chosen as centroids of the solution
        idx_centers = []

        # Select the K centroids
        for c in range(K):
            # List of candidates: tuples (i, j) where i is the sum of squared distances considering the point with index j as the new centroid
            candidate_list = []

            for i in idxs_points:
                if i not in idx_centers:
                    idx_centers.append(i)
                    labels = self.assign_objects(D, distances, idx_centers, ML_groups, CL_groups) # indices of idx_centers

                    # Calculate the sum of squared distances
                    total_squared_distance = 0
                    for idx_point, idx_c in enumerate(labels):
                        total_squared_distance += distances[idx_point,idx_centers[idx_c]]

                    candidate_list.append((total_squared_distance,i))
                    idx_centers.pop()

            # Randomly select a new centroid from the top 10% of the best candidates
            candidate_list.sort()
            alpha = max(2, int(len(candidate_list) * 0.1))  
            new_center = random.choice(candidate_list[:alpha])[1]
            idx_centers.append(new_center)

        return D[idx_centers]
    
    """
    Executes the algorithm.

    Parameters:
    -----------
        D (np.ndarray): Data matrix of shape (n, d), where each row represents a point in d dimensions.
        K (int): Number of clusters.
        ML: List of tuples (i, i'). Must-link constraints: points i and i' must belong to the same cluster.
        CL: List of tuples (i, i'). Cannot-link constraints: points i and i' cannot belong to the same cluster.
        seed (int): Random seed.
        ML_groups (list of sets): Each set represents a group of points that must be in the same cluster.  
        CL_groups (list of sets): The set at the i-th position contains the indices of the ML groups that cannot be in the same cluster as group i.

    Returns:
    --------
        best_labels (np.ndarray): Array of shape (n,) containing the cluster assignments for the best solution.
        best_centers (np.ndarray): Matrix of shape (K, d) representing the final cluster centroids.
        best_score (float): Score of the best solution found.
        n_iter (int): Number of algorithm iterations.
        n_iter_ls (int): Number of local search iterations.
        time (float): Elapsed time.
        bool: Indicates whether execution stopped due to reaching the maximum number of consecutive iterations without improvement.
    """
    def run(self, D, K, ML, CL, seed, ML_groups, CL_groups):
        start_time = time.time()

        # Compute penalty for greedy assignment: average of the differences between the maximum and minimum value of each attribute × number of attributes × 0.5
        if self.__assignment == 'greedy_rand_penalty':
            self.__greedy_penalty = np.mean(np.max(D, axis=0) - np.min(D, axis=0)) * D.shape[1] * 0.5

        if self.__verbose:
            print('||' + 'N°iter'.rjust(20) + ' |' + 'f*'.rjust(20) + ' |' + 'N°w/oImprBest'.rjust(20) + ' |' + 'N°iter_ls'.rjust(20) + ' |' + 'time'.rjust(20) + ' ||')
    
        np.random.seed(seed)    # set random seed

        # Initialize counters for total number of iterations, consecutive iterations without improvement, and local search iterations
        n_iter = 0
        n_cons_it_wo_impr = 0
        n_iter_ls = 0

        # Compute squared distances between each pair of points
        distances = cdist(D, D, metric='sqeuclidean')

        # Objective function value of the best solution initialized to infinity
        best_score = np.inf

        while n_cons_it_wo_impr < self.__Nmax and n_iter < self.__max_iter:
            # Construct randomized greedy solution
            centers = self.greedy_randomized_sol(D, distances, K, ML_groups, CL_groups)

            # Apply local search to the solution
            labels, centers, score, add_n_iter_ls = self.__ls.run(D, centers, K, ML, CL)
            n_iter_ls += add_n_iter_ls

           # Update best solution
            if best_score - score >= self.__tol_sol:
                best_labels = labels
                best_centers = centers
                best_score = score
                n_cons_it_wo_impr = 0
            else:
                n_cons_it_wo_impr += 1

            if self.__verbose:
                print('||' + str(n_iter).rjust(20) +  ' |' + str(round(best_score, 3)).rjust(20) + ' |' + str(n_cons_it_wo_impr).rjust(20) + ' |' + str(n_iter_ls).rjust(20) + ' |' + str(round(time.time() - start_time, 3)).rjust(20) + ' ||')

            n_iter += 1

        return best_labels, best_centers, best_score, n_iter, n_iter_ls, time.time() - start_time, n_cons_it_wo_impr==self.__Nmax
