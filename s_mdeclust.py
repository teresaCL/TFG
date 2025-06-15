# This file has been modified by Teresa Córdoba Lillo in 2025.
# It is based on original work licensed under the Apache License, Version 2.0.
# Original source: https://github.com/pierlumanzu/s_mdeclust

import time
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from gurobipy import Model, GRB
from blpkm import BLPKM

"""
Implements the S-MDEClust algorithm for constrained clustering.
"""
class S_MDEClust:
    """
    Initializes the model parameters.

    Parameters:
    -----------
        assignment (str): Type of assignment to use.
        mutation (bool): Boolean value indicating whether mutation should be performed or not.
        P (int): Population size.
        Nmax (int): Maximum number of consecutive evaluations without improvement.
        max_iter (int): Maximum number of iterations.
        tol_pop (float): Diversity threshold for the population (stopping criterion).
        Nmax_ls (int): Maximum number of consecutive local search iterations without improvement.
        max_iter_ls (int): Maximum number of local search iterations.
        tol_sol (float): Solution tolerance threshold. Indicates the required improvement in a solution's score for it to be replaced.
        F (float or str): Crossover parameter.
        crossover: (str): Type of crossover to use.
        alpha (float): Parameter for the mutation operator to calculate probabilities, where 0 is equiprobable and 1 is proportional to the distance.
        solis (str): Indicates whether the Solis-Wets algorithm should not be used, or if it should be applied with a penalized or non-penalized objective function.
        apply_LS_all (bool): Boolean value indicating whether the local search should be applied to all solutions.
        restart (int): Number of population restarts.
        decrease_pop_size_reset (bool):
        shade (bool): Boolean value indicating whether SHADE should be applied.
        verbose (bool): If True, displays results during the process.

    Returns:
    --------
        None
    """
    def __init__(self, 
                 assignment, mutation, 
                 P, Nmax, max_iter, tol_pop, 
                 Nmax_ls, max_iter_ls, 
                 tol_sol, F, crossover, alpha,
                 solis,
                 apply_LS_all,
                 restart,
                 decrease_pop_size_reset,
                 shade,
                 verbose):
        
        self.__assignment = assignment
        self.__mutation = mutation  
        self.__P = P    
        self.__Nmax = Nmax  
        self.__max_iter = max_iter  
        self.__tol_pop = tol_pop           
        self.__tol_sol = tol_sol  
        self.__F = F
        self.__crossover = crossover
        self.__alpha = alpha  
        self.__solis = solis 
        self.__apply_LS_all = apply_LS_all
        self.__restart = restart
        self.__decrease_pop_size_reset = decrease_pop_size_reset
        self.__shade = shade  

        if self.__decrease_pop_size_reset:
            self.__P *= 2
            self.__Nmax *= 2

        self.__verbose = verbose 
        self.__ls = BLPKM(Nmax_ls, max_iter_ls) # local search


    """
    Initializes the population.

    Parameters:
    -----------
        D (np.ndarray): Data matrix of dimension (N, d), where N is the number of samples and d is the feature dimension.
        K (int): Number of clusters.
        ML: List of tuples (i, i'). Must-link constraints: points i and i' must belong to the same cluster.
        CL: List of tuples (i, i'). Cannot-link constraints: points i and i' cannot belong to the same cluster.
        start_time (float): Timestamp indicating when the initialization process started.
        ML_groups: list of sets. Each set represents a group of points that must be in the same cluster.  
        CL_groups: list of sets. The set at the i-th position contains the indices of the ML groups that cannot be in the same cluster as group i.
    
    Returns:
    -----------
        phi (np.ndarray): Array representing the cluster assignments for each individual in the population, with dimensions (P x N), where P is the population size and N is the number of individuals.  
        psi (np.ndarray): Array containing the cluster centroids for each individual, with dimensions (P x K x d), where P is the population size, K is the number of clusters, and d is the feature dimension.  
        scores (np.ndarray): Scores of each individual in the population.
        best_idx (int): Index of the best individual in the population (lowest score).
        n_iter_ls (int): Total number of local search iterations.
    """
    def initialize_population(self, D, K, ML, CL, start_time, ML_groups, CL_groups):
        N, d = D.shape   # number of instances (N) and number of features (d)

        # Initialize the membership vector and the centroid matrix for each individual, and set each individual's scores to infinity.
        phi = np.empty((self.__P, N), dtype=int)     
        psi = np.empty((self.__P, K, d), dtype=float) 
        scores = np.inf * np.ones(self.__P, dtype=float)  
        n_iter_ls = 0   # counter for local search iterations

        if not self.__apply_LS_all:
            # Select which individuals to apply local search to
            idx_LS = np.random.choice(np.arange(self.__P), size=int(0.2*self.__P), replace=False)

        # Generation of population individuals
        for n_p in range(self.__P):
            # Select K points from the dataset without replacement
            center_ids = np.random.choice(np.arange(D.shape[0]), size=K, replace=False)
            centers = D[center_ids, :]

            # Execute local search and update the local search iteration counter or repair solutions
            if self.__apply_LS_all or (n_p in idx_LS):
                phi[n_p, :], psi[n_p, :, :], scores[n_p], n_iter = self.__ls.run(D, centers, K, ML, CL)
                n_iter_ls += n_iter
            else:
                phi[n_p, :] = self.assign_objects(D, psi[n_p], ML, CL, ML_groups, CL_groups, True)  # exact assignment step
                scores[n_p] = self.calculate_fitness(D, psi[n_p], phi[n_p], ML, CL)

            if self.__verbose:
                print('||' + str(-1).rjust(20) + ' |' + str(n_p+1).rjust(20) + ' |' + str(round(np.min(scores), 3)).rjust(20) + ' |' + 'N/A'.rjust(20) + ' |' + 'N/A'.rjust(20) + ' |' + str(n_p+1).rjust(20) + ' |' + str(n_iter_ls).rjust(20) + ' |' + str(round(time.time() - start_time, 3)).rjust(20) + ' ||')

        return phi, psi, scores, np.argmin(scores), n_iter_ls
    
    """
    Computes the population diversity as the sum of score differences between each pair of individuals in the population.

    Parameters:
    -----------
        scores (np.ndarray): Array of population scores.

    Returns:
    -----------
        div (float): population diversity.
    """
    @staticmethod
    def population_diversity(scores):
        div = 0
        for i in range(len(scores)):
            for j in range(i+1, len(scores)):
                div = div + abs(scores[i] - scores[j])
        return div

    """
    Performs an optimal assignment between the centers of two solutions.

    Parameters:
    -----------
        psi1 (np.ndarray): Center matrix of the first solution, with size (K, d), where K is the number of clusters and d is the feature dimension.
        psi2 (np.ndarray): Center matrix of the second solution, with size (K, d), where K is the number of clusters and d is the feature dimension.

    Returns:
    -----------
        new_psi1 (np.ndarray): A new version of psi1 with centers reordered according to the best assignment.
    """
    @staticmethod
    def exact_matching(psi1, psi2):
        # Compute the cost matrix W based on the squared Euclidean distance between each pair of centers
        W = np.linalg.norm(psi1[:, None, :]-psi2[None, :, :], axis=2)**2
       
        # Solve the optimal assignment problem
        row_indices, col_indices = linear_sum_assignment(W)

        # Reorder the centers of psi1 according to the best assignment found
        new_psi1 = np.zeros(psi1.shape)
        new_psi1[col_indices] = psi1[row_indices]

        return new_psi1
    
    """
    Assigns each point in D to a cluster center in 'centers', trying to respect must-link (ML) and cannot-link (CL) constraints.

    Parameters:
    -----------
        D (np.ndarray): Data matrix of dimension (N, d), where N is the number of samples and d is the feature dimension.
        centers (np.ndarray): Matrix containing the cluster centers, with shape (K, d), where K is the number of clusters and d is the feature dimension.         
        ML: List of tuples (i, i'). Must-link constraints: points i and i' must belong to the same cluster.
        CL: List of tuples (i, i'). Cannot-link constraints: points i and i' cannot belong to the same cluster.
        ML_groups (list of sets): Each set represents a group of points that must be in the same cluster.  
        CL_groups (list of sets): The set at the i-th position contains the indices of the ML groups that cannot be in the same cluster as group i.
        exact(bool): Is True, force to use the exact assignment.

    Returns:
    --------
        labels (np.ndarray): Array containing the cluster label assigned to each object in D, with shape (n,), where n is the number of samples.  
    """
    def assign_objects(self, D, centers, ML, CL, ML_groups, CL_groups, exact=False):
        # EXACT ASSIGNMENT
        if self.__assignment == 'exact' or exact:
            n = D.shape[0]   # number of data points     
            k = centers.shape[0]   # number of clusters

            # Compute the squared Euclidean distance between each point and each cluster centroid
            distances = cdist(D, centers, metric='sqeuclidean')
            assignments = {(i, j): distances[i, j] for i in range(n) for j in range(k)}

            # Create the optimization model
            m = Model()
            m.setParam("OutputFlag", False)

            # Binary variables: y[i, j] = 1 if point i is assigned to cluster j
            y = m.addVars(assignments, obj=assignments, vtype=GRB.BINARY)

            # Constraint 1: Each point must be assigned to exactly one cluster
            m.addConstrs(y.sum(i, '*') == 1 for i in range(n))

            # Constraint 2: Each cluster must have at least one assigned point
            m.addConstrs(y.sum('*', j) >= 1 for j in range(k))

            # Constraint 3: Enforce Must-Link (ML) constraints
            m.addConstrs(y[i, j] == y[i_, j] for j in range(k) for i, i_ in ML)
            
            # Constraint 4: Enforce Cannot-Link (CL) constraints
            m.addConstrs(y[i, j] + y[i_, j] <= 1 for j in range(k) for i, i_ in CL)

            # Run the optimizer
            m.optimize()

            # Retrieve cluster labels for each point
            labels = np.array([j for i, j in y.keys() if y[i, j].X > 0.5])

            return labels
        # GREEDY ASSIGNMENT AND VARIANTS
        else:
            # Initialize labels for each point to -1 (unassigned)
            labels = -1 * np.ones(len(D), dtype=int)

            # Initialize list that will store, for each centroid, the indices of the groups assigned to that centroid
            groups_to_centers = []
            for k in range(len(centers)):
                groups_to_centers.append([])
            
            # Iterate through each ML group
            for idx_ml_gr, ml_gr in enumerate(ML_groups):
                if self.__assignment == 'greedy':
                    # Indices of the clusters where the points in the ML group ml_gr can be assigned
                    possible_ks = np.array([k for k in range(len(centers)) if len(set(groups_to_centers[k]) & CL_groups[idx_ml_gr]) == 0])
                    if len(possible_ks) == 0:
                        possible_ks = np.arange(len(centers))
                    
                    # Index of the cluster among the possible ones that minimizes the sum of distances from the group's points to the cluster centroid
                    ml_gr_k = np.argmin(np.array([np.sum(np.linalg.norm(D[np.array(list(ml_gr))] - centers[pk], axis=1)**2) for pk in possible_ks]))

                elif self.__assignment == 'greedy_rand_penalty': 
                    # All centroids have a probability of being selected
                    possible_ks = np.arange(len(centers))

                    # Compute distances for each centroid with penalty
                    distances = [
                        np.sum(np.linalg.norm(D[np.array(list(ml_gr))] - centers[k], axis=1) ** 2) +
                        (self.__greedy_penalty if len(set(groups_to_centers[k]) & CL_groups[idx_ml_gr]) != 0 else 0)
                        for k in range(len(centers))
                    ]
                    # Invert distances and normalize so they sum to 1 to obtain selection probabilities and select a cluster k
                    distances = np.array(distances, dtype=np.float64)
                    distances[distances == 0] = 1e-10   
                    probs = 1 / distances 
                    probs /= probs.sum() 
                    ml_gr_k = np.random.choice(len(distances), p=probs)

                elif self.__assignment == 'greedy_rand': 
                    # Indices of the clusters where the points in the ML group ml_gr can be assigned
                    possible_ks = np.array([k for k in range(len(centers)) if len(set(groups_to_centers[k]) & CL_groups[idx_ml_gr]) == 0])
                    if len(possible_ks) == 0:
                        possible_ks = np.arange(len(centers))

                    # Sum of squared distances from the group's points to each centroid
                    distances_k = np.array([np.sum(np.linalg.norm(D[np.array(list(ml_gr))] - centers[pk], axis=1)**2) for pk in possible_ks])
                    
                    # Invert distances and normalize so they sum to 1 to obtain selection probabilities and select a cluster k
                    distances_k[distances_k == 0] = 1e-10
                    probs = 1 / distances_k
                    probs /= probs.sum()
                    ml_gr_k = np.random.choice(len(distances_k), p=probs)

                # Add the index of the ML group ml_gr to the cluster list ml_gr_k
                groups_to_centers[possible_ks[ml_gr_k]].append(idx_ml_gr)

                # Update the label of the group's points
                labels[np.array(list(ml_gr))] = possible_ks[ml_gr_k]

            return labels
        

    """
    Applies the crossover operator
    
    Parameters:
    -----------
        D (np.ndarray): Data matrix of dimension (N, d), where N is the number of samples and d is the feature dimension.
        psis (list of np.ndarray): List with the matrices of centroids that will be combined.
        ML: List of tuples (i, i'). Must-link constraints: points i and i' must belong to the same cluster.
        CL: List of tuples (i, i'). Cannot-link constraints: points i and i' cannot belong to the same cluster.
        ML_groups (list of sets): Each set represents a group of points that must be in the same cluster.  
        CL_groups (list of sets): The set at the i-th position contains the indices of the ML groups that cannot be in the same cluster as group i.

    Returns:
    -----------
        labels (np.ndarray): Array of shape (n,) containing the cluster label assigned to each object in D.
        psiO (np.ndarray): Matrix of centroids resulting from the crossover.
    """
    def crossover(self, D, psis, ML, CL, ML_groups, CL_groups):
        # Select F  
        if self.__F == 'random':    # F in (0,2)
            F = (1e-7 + np.random.rand() * (2-2e-7))        
        elif self.__F == 'mdeclust':    # F in [0.5,0.8)
            F = (0.5 + np.random.rand() * 0.3)    
        elif self.__F == 'half_mdeclust':   # F in [0.25,0.4)
            F = (0.25 + np.random.rand() * 0.15)    
        elif type(self.__F) == float or type(self.__F) == int:
            F = self.__F        
        else:
            raise AssertionError

        # Aplies crossover
        if self.__crossover == 'original':
            psiO = psis[0] + F * (psis[1] - psis[2])
        else:
            psi_i, psib, psi1, psi2 = psis
            psiO = psi_i + F * (psib - psi_i + psi1 - psi2)
        
        return self.assign_objects(D, psiO, ML, CL, ML_groups, CL_groups), psiO


    """
    Assigns each point in D to a cluster center in 'centers', trying to respect must-link (ML) and cannot-link (CL) constraints while excluding a centroid.

    Parameters:
    -----------
        D (np.ndarray): Data matrix of dimension (N, d), where N is the number of samples and d is the feature dimension.
        labels (np.ndarray): Array of shape (n,), containing the cluster label assigned to each object in D.
        centers (np.ndarray): Matrix containing the cluster centers, with shape (K, d), where K is the number of clusters and d is the feature dimension.         
        ML: List of tuples (i, i'). Must-link constraints: points i and i' must belong to the same cluster.
        CL: List of tuples (i, i'). Cannot-link constraints: points i and i' cannot belong to the same cluster.
        idx_c (int): Index of the removed cluster.
        ML_groups (list of sets): Each set represents a group of points that must be in the same cluster.  
        CL_groups (list of sets): The set at the i-th position contains the indices of the ML groups that cannot be in the same cluster as group i.

    Returns:
    --------
        labels (np.ndarray): Array of shape (n,), containing the cluster label assigned to each object in D.
        successful (bool): Boolean indicating whether or not the assignment was successful
    """
    def assign_objects_excluding_c(self, D, labels, centers, ML, CL, idx_c, ML_groups, CL_groups):
        # EXACT ASSIGNMENT
        if self.__assignment == 'exact':
            n = D.shape[0]  # number of data points  
            k = centers.shape[0]   # number of clusters 

            # Compute the squared Euclidean distance between each point and each cluster
            distances = cdist(D, centers, metric='sqeuclidean')
            assignments = {(i, j): distances[i, j] for i in range(n) for j in range(k)}

            # Create the optimization model
            m = Model()
            m.setParam("OutputFlag", False)

            # Binary variables: y[i, j] = 1 if point i is assigned to cluster j
            y = m.addVars(assignments, obj=assignments, vtype=GRB.BINARY)

            # Constraint 1: Each point must be assigned to exactly one cluster
            m.addConstrs(y.sum(i, '*') == 1 for i in range(n))

            # Constraint 2: Each cluster must have at least one assigned point (except for cluster idx_c, which must not have any assigned points)
            m.addConstrs(y.sum('*', j) >= 1 if j != idx_c else y.sum('*', j) == 0 for j in range(k))
            
            # Constraint 3: Enforce Must-Link (ML) constraints
            m.addConstrs(y[i, j] == y[i_, j] for j in range(k) for i, i_ in ML)
            
            # Constraint 4: Enforce Cannot-Link (CL) constraints
            m.addConstrs(y[i, j] + y[i_, j] <= 1 for j in range(k) for i, i_ in CL)

            # Run the optimizer
            m.optimize()

            # Retrieve cluster labels for each point
            if m.Status == GRB.OPTIMAL:
                labels = np.array([j for i, j in y.keys() if y[i, j].X > 0.5])
                return labels, True
            else:
                return np.empty(0), False
        
        # GREEDY ASSIGNMENT AND VARIANTS 
        else:
            groups_to_centers = []  # List where position i contains the indices of the groups assigned to the centroid at index i
            groups_to_modify = [] # List with the indices of the groups that need to be reassigned because they are assigned to cluster idx_c

            for k in range(len(centers)):
                groups_to_centers.append([])   # Initialize with empty lists

                # Indices of the points assigned to cluster k
                assigned_points = np.where(labels == k)[0]

                for idx_ml_gr, ml_gr in enumerate(ML_groups):
                    # Check if the group ml_gr is assigned to cluster k and update the lists groups_to_centers or groups_to_modify
                    if len(ml_gr & set(assigned_points)) != 0:  
                        if k != idx_c:
                            groups_to_centers[-1].append(idx_ml_gr)
                        else:
                            groups_to_modify.append(idx_ml_gr)

            # Iterate over the groups that need to be modified because they are assigned to cluster idx_c
            for idx_ml_gr in groups_to_modify:
                if self.__assignment == 'greedy':
                    # Indices of the clusters where the points in the ML group ml_gr can be assigned
                    possible_ks = np.array([k for k in range(len(centers)) if (k != idx_c) and (len(set(groups_to_centers[k]) & CL_groups[idx_ml_gr]) == 0)])
                    if len(possible_ks) == 0:
                        possible_ks = np.array([k for k in range(len(centers)) if k != idx_c])
                    
                    # Index of the cluster among the possible ones that minimizes the sum of distances from the group's points to the cluster centroid
                    if len(possible_ks) > 1:
                        ml_gr_k = np.argmin(np.array([np.sum(np.linalg.norm(D[np.array(list(ML_groups[idx_ml_gr]))] - centers[pk], axis=1)**2) for pk in possible_ks]))
                    else:
                        ml_gr_k = 0

                elif self.__assignment == 'greedy_rand_penalty': 
                    # Indices of the clusters where the points in the ML group ml_gr can be assigned
                    possible_ks = np.array([k for k in range(len(centers)) if k != idx_c])

                    # Compute distances for each centroid with penalty
                    distances = [
                        np.sum(np.linalg.norm(D[np.array(list(ml_gr))] - centers[k], axis=1) ** 2) +
                        (self.__greedy_penalty if len(set(groups_to_centers[k]) & CL_groups[idx_ml_gr]) != 0 else 0)
                        for k in range(len(centers)) if k != idx_c
                    ]
                    # Invert distances and normalize so they sum to 1 to obtain selection probabilities and select a cluster k
                    distances = np.array(distances, dtype=np.float64)
                    distances[distances == 0] = 1e-10 
                    probs = 1 / distances 
                    probs /= probs.sum() 
                    ml_gr_k = np.random.choice(len(distances), p=probs)

                elif self.__assignment == 'greedy_rand': 
                    # Indices of the clusters where the points in the ML group ml_gr can be assigned
                    possible_ks = np.array([k for k in range(len(centers)) if (k != idx_c) and (len(set(groups_to_centers[k]) & CL_groups[idx_ml_gr]) == 0)])
                    if len(possible_ks) == 0:
                        possible_ks = np.arange(len(centers))

                    # Compute distances for each centroid 
                    distances_k = np.array([np.sum(np.linalg.norm(D[np.array(list(ml_gr))] - centers[pk], axis=1)**2) for pk in possible_ks])
                    
                    # Invert distances and normalize so they sum to 1 to obtain selection probabilities and select a cluster k
                    distances_k[distances_k == 0] = 1e-10
                    probs = 1 / distances_k
                    probs /= probs.sum()
                    ml_gr_k = np.random.choice(len(distances_k), p=probs)

                # Add the index of the ML group ml_gr to the cluster list ml_gr_k
                groups_to_centers[possible_ks[ml_gr_k]].append(idx_ml_gr)

                # Update the label of the group's points
                labels[np.array(list(ML_groups[idx_ml_gr]))] = possible_ks[ml_gr_k]

            return labels, True

    """
    Computes the assignment probabilities of instances as centroids based on the distance from the instance to its assigned centroid and a weighting parameter alpha.

    Parameters:
    -----------
        D (np.ndarray): Data matrix of dimension (N, d), where N is the number of samples and d is the feature dimension.
        phi (np.ndarray): Membership vector with dimensions (N,), where N is the number of individuals.  
        psi (np.ndarray): Matrix of centroids with dimensions (K x d), where K is the number of clusters and d is the feature dimension.  
        alpha (float): Parameter that controls the balance between a uniform assignment and a distance-based assignment.

    Returns:
    --------
        np.ndarray: Array of shape (n,) containing the assignment probabilities for each point in D.
    """
    @staticmethod
    def calculate_probs(D, phi, psi, alpha):
        if alpha != 0:
            # Compute the distances between data points and their assigned centers
            d = np.linalg.norm(D - psi[phi], axis=1)

            # Compute distance-weighted probabilities
            return (((1 - alpha) / len(D)) * np.ones(len(D))) + (alpha * (d / np.sum(d)))
        else:
            # If alpha is 0 -> equal probability
            return (1 / len(D)) * np.ones(len(D))
    
    """
    Applies the mutation operator

    Parameters:
    -----------
        D (np.ndarray): Data matrix of dimension (N, d), where N is the number of samples and d is the feature dimension.
        phiO (np.ndarray): Membership vector of the original solution, with dimensions (N,), where N is the number of individuals.  
        psiO (np.ndarray): Matrix of centroids of the original solution, with dimensions (K x d), where K is the number of clusters and d is the feature dimension.  
        ML: List of tuples (i, i'). Must-link constraints: points i and i' must belong to the same cluster.
        CL: List of tuples (i, i'). Cannot-link constraints: points i and i' cannot belong to the same cluster.
        ML_groups (list of sets): Each set represents a group of points that must be in the same cluster.  
        CL_groups (list of sets): The set at the i-th position contains the indices of the ML groups that cannot be in the same cluster as group i.

    Returns:
    --------
        np.ndarray: New matrix of centroids after mutation. 
    """
    def mutation(self, D, phiO, psiO, ML, CL, ML_groups, CL_groups):
        # Randomly select an index of a centroid in psiO to be removed
        idx_removed_c = np.random.randint(len(psiO))
        
        # Assign points to clusters, excluding the removed centroid
        tmp_phi, success = self.assign_objects_excluding_c(D, phiO, psiO, ML, CL, idx_removed_c, ML_groups, CL_groups)
        
        # Compute selection probabilities for new centroids
        probs = self.calculate_probs(D, tmp_phi, psiO, self.__alpha if success else 0)

        # Select a new centroid based on the computed probabilities
        idx_new_center = np.random.choice(np.arange(len(D)), p=probs)

        # Replace the removed centroid with the newly selected center
        psiO[idx_removed_c] = np.copy(D[idx_new_center])
        
        return psiO
   
    """
    Computes the proportion of constraints violated by the solution.

    Parameters:
    -----------
        labels (np.ndarray): Array of cluster assignments for each point.
        ML: List of tuples (i, i'). Must-link constraints: points i and i' must belong to the same cluster.
        CL: List of tuples (i, i'). Cannot-link constraints: points i and i' cannot belong to the same cluster.

    Returns:
    --------
        float: proportion of violated constraints in the solution.
    """
    @staticmethod
    def calculate_infeasibility(labels, ML, CL):
        total = len(ML) + len(CL)
        num = 0
        for i, j in ML:
            if labels[i] != labels[j]:
                num += 1

        for i, j in CL:
            if labels[i] == labels[j]:
                num += 1

        return num/total

    """
    Computes the objective function value of a solution.

    Parameters:
    -----------
        D (np.ndarray): Data matrix of dimension (N, d), where N is the number of samples and d is the feature dimension.
        centers (np.ndarray): Matrix of centroids of the solution, with dimensions (K x d), where K is the number of clusters and d is the feature dimension.  
        labels (np.ndarray): Membership vector of the solution, with dimensions (N,), where N is the number of individuals.  
        ML: List of tuples (i, i'). Must-link constraints: points i and i' must belong to the same cluster.
        CL: List of tuples (i, i'). Cannot-link constraints: points i and i' cannot belong to the same cluster.

    Returns:
    --------
        float: Objective function value of the solution.
    """
    @staticmethod
    def calculate_fitness(D, centers, labels, ML, CL):
        return np.sum((D - centers[labels]) ** 2)
    
    
    """
    Computes the objective function value of a solution, adding a penalty for constraint violations.

    Parameters:
    -----------
        D (np.ndarray): Data matrix of dimension (N, d), where N is the number of samples and d is the feature dimension.
        centers (np.ndarray): Matrix of centroids of the solution, with dimensions (K x d), where K is the number of clusters and d is the feature dimension.  
        labels (np.ndarray): Membership vector of the solution, with dimensions (N,), where N is the number of individuals.  
        ML: List of tuples (i, i'). Must-link constraints: points i and i' must belong to the same cluster.
        CL: List of tuples (i, i'). Cannot-link constraints: points i and i' cannot belong to the same cluster.

    Returns:
    --------
        float: Objective function value of the solution, penalized based on constraint violations.
    """
    @staticmethod
    def calculate_fitness_with_penalty(D, centers, labels, ML, CL):
        fun_obj_wo_penalty = S_MDEClust.calculate_fitness(D, centers, labels, ML, CL)
        infeasibility = S_MDEClust.calculate_infeasibility(labels, ML, CL)
        return fun_obj_wo_penalty * (1 + 10 * infeasibility)
    
    """
    Applies the Solis-Wets algorithm to a solution.

    Parameters:
    -----------
        centers (np.ndarray): Matrix of centroids of the solution, with dimensions (K x d), where K is the number of clusters and d is the feature dimension.  
        maxevals (int): Maximum number of evaluations allowed.
        rho (float): Step size parameter controlling the search radius.
        D (np.ndarray): Data matrix of dimension (N, d), where N is the number of samples and d is the feature dimension.
        ML: List of tuples (i, i'). Must-link constraints: points i and i' must belong to the same cluster.
        CL: List of tuples (i, i'). Cannot-link constraints: points i and i' cannot belong to the same cluster.
        ML_groups (list of sets): Each set represents a group of points that must be in the same cluster.  
        CL_groups (list of sets): The set at the i-th position contains the indices of the ML groups that cannot be in the same cluster as group i.

    Returns:
    --------
        labels (np.ndarray): Array of cluster assignments for each point.
        centers (np.ndarray): Updated array of centroids.
        evals (int): Total number of evaluations performed.
    """
    def soliswets(self, centers, maxevals, rho, D, ML, CL, ML_groups, CL_groups):
        # Select the objective function based on whether constraint violations are penalized
        obj_func = S_MDEClust.calculate_fitness_with_penalty if self.__solis=='w_penalty' else S_MDEClust.calculate_fitness

        # Evaluate the initial solution
        labels = self.assign_objects(D, centers, ML, CL, ML_groups, CL_groups)
        fitness_sol = obj_func(D, centers, labels, ML, CL)

        # Initialize bias (direction vector for searching solutions) to zero
        bias = np.zeros_like(centers)

        # Counters
        evals = 0       # total number of evaluations performed
        failures = 0    # number of consecutive failed attempts
        successes = 0   # number of consecutive successful attempts

        while evals < maxevals and rho > 1e-5:
            # Generate a random normal difference (mean = bias, standard deviation = rho)
            dif = np.random.normal(loc=bias, scale=rho, size=centers.shape)

            # Test a new solution by adding the difference
            newcenters = centers + dif
            newlabels = self.assign_objects(D, newcenters, ML, CL, ML_groups, CL_groups)
            fitness_new_sol = obj_func(D, newcenters, newlabels, ML, CL)
            evals += 1

            # If the new solution improves, update bias toward the direction that improved
            if fitness_sol - fitness_new_sol > self.__tol_sol:
                failures = 0
                successes += 1

                bias = 0.4 * dif + 0.2 * bias

                centers = newcenters
                labels = newlabels
                fitness_sol = fitness_new_sol

            # If it does not improve, try in the opposite direction (subtracting the difference)
            elif evals < maxevals:
                newcenters = centers - dif
                newlabels = self.assign_objects(D, newcenters, ML, CL, ML_groups, CL_groups)
                fitness_new_sol = obj_func(D, newcenters, newlabels, ML, CL)
                evals += 1

                if fitness_sol - fitness_new_sol > self.__tol_sol:
                    # If it improves in the opposite direction, update as before but with a negative bias adjustment
                    failures = 0
                    successes += 1
                    bias = bias - 0.4 * dif
                    centers = newcenters
                    labels = newlabels
                    fitness_sol = fitness_new_sol
                # If it still does not improve, reduce bias by half
                else:
                    failures += 1
                    successes = 0
                    bias *= 0.5

            # Adjust rho: if there are 5 consecutive improvements, double it; if there are 3 consecutive failures, halve it
            if successes >= 5:
                successes = 0
                rho *= 2
            elif failures >= 3:
                failures = 0
                rho /= 2
        return labels, centers, evals


    """
    Updates the memory values for mutation (M_F) and crossover rate (M_CR) based on the improvements observed in recent solutions.

    Parameters:
    -----------
        S_CR (list[float]): List of crossover rates associated with successful solutions.
        S_F (list[float]): List of scaling factors associated with successful solutions.
        improvements (list[float]): List of improvement values, used to weigh the updates.

    Returns:
    --------
        M_F_new (float): Updated scaling factor, constrained within the range [0,1].
        M_CR_new (float): Updated crossover rate, constrained within the range [0,1].
    """
    def update_Ms(self, S_CR, S_F, improvements):
        # Compute the total improvement
        total = sum(improvements)

        # Compute the weights
        weights = [improvement / total for improvement in improvements]

        # Compute M_F_new
        weighted_S_F = [w * f for w, f in zip(weights, S_F)]
        weighted_S_F_squared = [wf * f for wf, f in zip(weighted_S_F, S_F)]

        M_F_new = sum(weighted_S_F_squared) / sum(weighted_S_F)
        M_F_new = max(0, min(1, M_F_new))

        # Compute M_CR_new
        weighted_S_CR = [w * cr for w, cr in zip(weights, S_CR)]
        M_CR_new = sum(weighted_S_CR)
        M_CR_new = max(0, min(1, M_CR_new))

        return M_F_new, M_CR_new
    
    """
    Executes the S-MDEClust algorithm.

    Parameters:
    -----------
        D (np.ndarray): Data matrix of dimension (N, d), where N is the number of samples and d is the feature dimension.
        K (int): Number of clusters.
        ML: List of tuples (i, i'). Must-link constraints: points i and i' must belong to the same cluster.
        CL: List of tuples (i, i'). Cannot-link constraints: points i and i' cannot belong to the same cluster.
        seed (int): random seed
        ML_groups (list of sets): Each set represents a group of points that must be in the same cluster.  
        CL_groups (list of sets): The set at the i-th position contains the indices of the ML groups that cannot be in the same cluster as group i.

    Returns:
    --------
        phi (np.ndarray): Membership vector of the best solution, with dimensions (N,), where N is the number of individuals.  
        psi (np.ndarray): Matrix of centroids of the best solution, with dimensions (K x d), where K is the number of clusters and d is the feature dimension.  
        score (float): Score of the best solution in the population.
        total_it (int): Total number of iterations performed by the algorithm.
        total_ls (int): Total number of local search executions.
        total_it_ls (int): Total number of iterations within the local search process.
        time(float): Elapsed time.
        pop_collapsed (bool): Indicates whether the algorithm terminated due to population collapse into a single solution.
        best_scores (float): Score of the best individual in the population.
        pop_diversity (float): Measure of the population's diversity, indicating variation in solutions.
        worst_scores (float): Score of the worst individual in the population.
        mean_scores (float): Average score of all individuals in the population.
    """
    def run(self, D, K, ML, CL, seed, ML_groups, CL_groups):
        start_time = time.time()

        if self.__verbose:
            print('||' + 'N°iter'.rjust(20) + ' |' + 'Sol'.rjust(20) + ' |' + 'f*'.rjust(20) + ' |' + 'N°w/oImprBest'.rjust(20) + ' |' + 'Pop_tol'.rjust(20) + ' |' + 'N°ls'.rjust(20) + ' |' + 'N°iter_ls'.rjust(20) + ' |' + 'time'.rjust(20) + ' ||')
    
        # Compute the penalty for greedy assignment: average of the differences between the maximum and minimum value of each attribute × number of attributes × 0.5
        if self.__assignment == 'greedy_rand_penalty':
            self.__greedy_penalty = np.mean(np.max(D, axis=0) - np.min(D, axis=0)) * D.shape[1] * 0.5

        # Compute de inital rho for the Solis Wets algorithm
        if self.__solis != "no":
            RHO = np.mean(np.max(D, axis=0) - np.min(D, axis=0)) * 0.1 

        # SHADE parameters
        if self.__shade:
            H = self.__P  # Memory size equal to population size
            M_CR = np.ones(H) * 0.5  # CR memory
            M_F = np.ones(H) * 0.5   # F memory
            sigma = np.sqrt(0.1)  # Standard normal deviation
        
        # Set random seed
        np.random.seed(seed)    

        # Data for the plots
        best_scores = []    # best score for each iteration    
        worst_scores = []   # worst score for each iteration    
        mean_scores = []    # average score for each iteration     
        pop_diversity = []  # population diversity for each iteration    

        # Counters
        total_it_ls = 0 # total number of local searchs iterations
        total_it = 0    # total number of iterations
        total_ls = 0    # total number of local searchs 
        num_restart = 0 # restart counter

        # Best solution from the previous restart
        best_centers = None
        best_labels = None
        best_fitness = None

        while num_restart <= self.__restart:
            # Initialize the population
            phi, psi, scores, best_s_idx, n_iter_ls = self.initialize_population(D, K, ML, CL, start_time, ML_groups, CL_groups)

            # Replace the worst solution with the best from the previous restart
            if num_restart > 0:
                idx_worst = np.argmax(scores)
                phi[idx_worst] = best_labels
                psi[idx_worst] = best_centers
                scores[idx_worst] = best_fitness
                if best_fitness+self.__tol_pop < scores[best_s_idx]:
                    best_s_idx = idx_worst

            # Initialize counters for the number of iterations and the number of consecutive iterations without improvement
            n_iter = 0
            n_cons_it_wo_impr = 0

            if self.__verbose:
                print('||' + str(n_iter).rjust(20) + ' |' + str(0).rjust(20) + ' |' + str(round(scores[best_s_idx], 3)).rjust(20) + ' |' + str(n_cons_it_wo_impr).rjust(20) + ' |' + str(round(self.population_diversity(scores), 3)).rjust(20) + ' |' + str(len(scores)).rjust(20) + ' |' + str(n_iter_ls).rjust(20) + ' |' + str(round(time.time() - start_time, 3)).rjust(20) + ' ||')

            max_it_reached = True  # indicates whether the current stage (the period between two restarts) has ended due to reaching the maximum number of iterations
            
            if self.__shade:
                k = 0 # index in the memories to be updated

            # While the maximum number of iterations has not been reached
            while n_cons_it_wo_impr < self.__Nmax and n_iter < self.__max_iter:
                diversity = self.population_diversity(scores)
                best_scores.append(scores[best_s_idx])
                worst_scores.append(max(scores))
                mean_scores.append(sum(scores)/len(scores))
                pop_diversity.append(diversity)

                # Check if the diversity threshold is met
                if diversity < self.__tol_pop:
                    max_it_reached = False
                    num_restart += 1

                    # Save best solution
                    best_centers = psi[best_s_idx]
                    best_labels = phi[best_s_idx]
                    best_fitness = scores[best_s_idx]

                    total_it_ls += n_iter_ls
                    total_it += n_iter

                    if self.__decrease_pop_size_reset:
                        self.__P = int(self.__P / 2)
                        self.__Nmax = int(self.__Nmax / 2)
                    break

                # Index of the solution to which Solis Wets local search is applied
                idx_solis = best_s_idx

                # Select which individuals to apply local search to: 10% of the best plus 10% random
                if not self.__apply_LS_all:
                    idx_sorted = np.argsort(scores)  
                    n_10pct = int(0.1 * self.__P) 

                    first_10pct = idx_sorted[:n_10pct]
                    idx_LS = list(first_10pct) 

                    remaining_indices = idx_sorted[n_10pct:]
                    second_10pct = np.random.choice(remaining_indices, size=n_10pct, replace=False)
                    idx_LS.extend(second_10pct)
                
                if self.__shade:
                    S_CR = []   # list of successful CR parameters in this iteration
                    S_F = []    # list of successful F parameters in this iteration
                    improvements = []   # list of improvements in this iteration

                # For each individual in the population
                for s in range(self.__P):
                    if self.__shade:
                        # Choose CR y F
                        idx_H = np.random.randint(0, H)
                        Fi = 0.1 * np.tan(np.pi * (np.random.rand() - 0.5)) + M_F[idx_H]    # cauchy distribution mean M_F[idx_H] y scale 0.1
                        CRi = np.random.normal(M_CR[idx_H], sigma)

                        # Truncate CR and F to ensure they are within the range [0,1]
                        Fi = max(0, min(1, Fi))
                        self.__F = Fi
                        CRi = max(0, min(1, CRi))
                    else:
                        CRi = 1 # crossover is applied with 100% probability

                    # Crossover
                    if (np.random.rand() < CRi):
                        if self.__crossover == "original":
                            s1, s2, s3 = np.random.choice(np.arange(self.__P), size=3, replace=False, p=np.array([1/(self.__P - 1) if i != s else 0 for i in range(self.__P)]))
                            psi1 = self.exact_matching(psi[s1], psi[s3])
                            psi2 = self.exact_matching(psi[s2], psi[s3])
                            phiO, psiO = self.crossover(D, [psi1, psi2, psi[s3]], ML, CL, ML_groups, CL_groups)
                        else:
                            if self.__crossover == 'pbest_v1':
                                s1, s2 = np.random.choice(np.arange(self.__P), size=2, replace=False, p=np.array([1/(self.__P - 1) if i != s else 0 for i in range(self.__P)]))
                                idx_scores_sorted = np.argsort(scores)
                                best_idxs = idx_scores_sorted[:int(len(scores)*0.25)]
                                sb = np.random.choice(best_idxs)
                            
                            elif self.__crossover == 'pbest_v2':
                                s1, s2 = np.random.choice(np.arange(self.__P), size=2, replace=False, p=np.array([1/(self.__P - 1) if i != s else 0 for i in range(self.__P)]))
                                random_idxs = np.random.choice(len(scores), size=int(len(scores)*0.25), replace=False)
                                sb = random_idxs[np.argmin(scores[random_idxs])]

                            psi1 = self.exact_matching(psi[s1], psi[s])
                            psi2 = self.exact_matching(psi[s2], psi[s])
                            psib = self.exact_matching(psi[sb], psi[s])
                            phiO, psiO = self.crossover(D, [psi[s], psib, psi1, psi2], ML, CL, ML_groups, CL_groups)
                    else:
                        psiO = psi[s].copy()
                        phiO = phi[s].copy()

                    # Mutation
                    if self.__mutation and np.random.rand() < 1/(n_iter + 1):
                        psiO = self.mutation(D, phiO, psiO, ML, CL, ML_groups, CL_groups)

                    # Solis Wets
                    if self.__solis!='no' and s==idx_solis:
                        phiO, psiO, add_n_iter_ls = self.soliswets(psiO, 10, RHO, D, ML, CL, ML_groups, CL_groups)
                        n_iter_ls += add_n_iter_ls
                        total_ls += 1

                    # Local Search
                    if self.__apply_LS_all or (s in idx_LS):
                        phiO, psiO, scoreO, add_n_iter_ls = self.__ls.run(D, psiO, K, ML, CL)
                        n_iter_ls += add_n_iter_ls
                        total_ls += 1
                    else:
                        phiO = self.assign_objects(D, psiO, ML, CL, ML_groups, CL_groups, True) # exact assignment step
                        scoreO = self.calculate_fitness(D, psiO, phiO, ML, CL)


                    # Check if there has been an improvement in the solution
                    if scores[s] - scoreO >= self.__tol_sol:
                        if self.__shade:
                            # Add Fi and CRi to S
                            S_CR.append(CRi)
                            S_F.append(Fi)
                            improvements.append(scores[s] - scoreO)

                        # Update solution and its score
                        phi[s] = phiO
                        psi[s] = psiO
                        scores[s] = scoreO
                        
                        # Check if the best solution has been improved
                        if scores[best_s_idx] - scores[s] >= self.__tol_sol:
                            best_s_idx = s
                            n_cons_it_wo_impr = 0
                        else:
                            n_cons_it_wo_impr += 1
                    else:
                        n_cons_it_wo_impr += 1

                    if self.__verbose:
                        print('||' + str(n_iter).rjust(20) + ' |' + str(s+1).rjust(20) + ' |' + str(round(scores[best_s_idx], 3)).rjust(20) + ' |' + str(n_cons_it_wo_impr).rjust(20) + ' |' + str(round(self.population_diversity(scores), 3)).rjust(20) + ' |' + str(n_iter * len(scores) + len(scores) + s + 1).rjust(20) + ' |' + str(n_iter_ls).rjust(20) + ' |' + str(round(time.time() - start_time, 3)).rjust(20) + ' ||')

                n_iter += 1

                # Update M_CR and M_F
                if self.__shade and len(S_CR)!=0:
                    M_F_new, M_CR_new = self.update_Ms(S_CR, S_F, improvements)
                    M_F[k] = M_F_new
                    M_CR[k] = M_CR_new
                    k = (k+1) % H
            
            if max_it_reached:
                num_restart += 1

                # Save best solution
                best_centers = psi[best_s_idx]
                best_labels = phi[best_s_idx]
                best_fitness = scores[best_s_idx]

                total_it_ls += n_iter_ls
                total_it += n_iter

                if self.__decrease_pop_size_reset:
                    self.__P = int(self.__P / 2)
                    self.__Nmax = int(self.__Nmax / 2)

        return phi[best_s_idx], psi[best_s_idx], scores[best_s_idx], total_it, total_ls, total_it_ls, time.time() - start_time, self.population_diversity(scores) < self.__tol_pop, best_scores, pop_diversity, worst_scores, mean_scores
