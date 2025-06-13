# Code adapted from the original version in https://github.com/phil85/BLPKM-CC
# © 2021, Universität Bern, Chair of Quantitative Methods, Philipp Baumann


from scipy.spatial.distance import cdist
import gurobipy as gb
import numpy as np

"""
Implements the BLPKM algorithm.
"""
class BLPKM:

    """
    Initializes the model with stopping criteria parameters.
        
        Nmax: Maximum number of consecutive iterations without improvement allowed.
        max_iter: Maximum number of iterations for the algorithm.
    """
    def __init__(self, Nmax, max_iter):
        self.__Nmax = Nmax  
        self.__max_iter = max_iter


    """
    Assigns each point in D to a cluster center in 'centers', respecting must-link (ML) and cannot-link (CL) constraints.

    Parameters:
    -----------
        D (np.ndarray): Data matrix of dimension (N, d), where N is the number of samples and d is the feature dimension.
        centers (np.ndarray): Matrix containing the cluster centers, with shape (K, d), where K is the number of clusters and d is the feature dimension.         
        ML: List of tuples (i, i'). Must-link constraints: points i and i' must belong to the same cluster.
        CL: List of tuples (i, i'). Cannot-link constraints: points i and i' cannot belong to the same cluster.

    Returns:
    --------
        labels (np.ndarray): Array containing the cluster label assigned to each object in D, with shape (n,), where n is the number of samples.  
    """
    @staticmethod
    def assign_objects(D, centers, ML, CL):
        n = D.shape[0]  # number of data points     
        k = centers.shape[0]    # number of clusters

        # Compute the squared Euclidean distance between each point and each cluster
        distances = cdist(D, centers, metric='sqeuclidean')
        assignments = {(i, j): distances[i, j] for i in range(n) for j in range(k)}

        # Create the optimization model
        m = gb.Model()
        m.setParam("OutputFlag", False)

        # Binary variables: y[i, j] = 1 if point i is assigned to cluster j
        y = m.addVars(assignments, obj=assignments, vtype=gb.GRB.BINARY)

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
    

    """
    Recomputes cluster centers as the mean of the points assigned to each cluster.
    
    Parameters:
    -----------
        D (np.ndarray): Data matrix of dimension (N, d), where N is the number of samples and d is the feature dimension.
        centers (np.ndarray): Matrix containing the cluster centers, with shape (K, d), where K is the number of clusters and d is the feature dimension.         
        K (int): Number of clusters.
        labels (np.ndarray): Array containing the cluster label assigned to each object in D, with shape (n,), where n is the number of samples.  

    Returns:
    --------
        ndarray of shape (K, d). Updated matrix of centroids.
    """
    @staticmethod
    def update_centers(D, centers, K, labels):
        for i in range(K):
            centers[i] = D[labels == i, :].mean(axis=0)
        return centers
    
    """
    Computes the total squared distance between data points and their assigned centers.

    Parameters:
    -----------
        D (np.ndarray): Data matrix of dimension (N, d), where N is the number of samples and d is the feature dimension.
        centers (np.ndarray): Matrix containing the cluster centers, with shape (K, d), where K is the number of clusters and d is the feature dimension.         
        labels (np.ndarray): Array containing the cluster label assigned to each object in D, with shape (n,), where n is the number of samples.  

    Returns:
    --------
        (float): Total squared distance.
    """
    @staticmethod
    def get_total_distance(D, centers, labels):
        return np.sum((D - centers[labels]) ** 2)


    """
    Executes the BLPKM algorithm.

    Parameters:
    -----------
        D (np.ndarray): Data matrix of dimension (N, d), where N is the number of samples and d is the feature dimension.
        centers (np.ndarray): Matrix containing the cluster centers, with shape (K, d), where K is the number of clusters and d is the feature dimension.
        K (int): Number of clusters.
        ML: List of tuples (i, i'). Must-link constraints: points i and i' must belong to the same cluster.
        CL: List of tuples (i, i'). Cannot-link constraints: points i and i' cannot belong to the same cluster.

    Returns:
    --------
        best_labels (np.ndarray): Array of final labels for each data point.
        best_centers (np.ndarray): Updated matrix of final cluster centers.
        best_total_distance (float): Total squared distance between data points and their assigned clusters.
        n_iter (int): Number of iterations until convergence.
    """
    def run(self, D, centers, K, ML, CL):
        labels = self.assign_objects(D, centers, ML, CL)
        best_labels = labels

        centers = self.update_centers(D, centers, K, labels)
        best_centers = centers

        best_total_distance = self.get_total_distance(D, centers, labels)

        n_iter = 1
        n_cons_it_wo_impr = 0

        while n_iter < self.__max_iter:
            labels = self.assign_objects(D, centers, ML, CL)

            centers = self.update_centers(D, centers, K, labels)

            total_distance = self.get_total_distance(D, centers, labels)

            if total_distance >= best_total_distance:
                n_cons_it_wo_impr += 1
                if n_cons_it_wo_impr >= self.__Nmax:
                    n_iter += 1
                    break
            else:
                n_cons_it_wo_impr = 0

                best_labels = labels
                best_centers = centers
                best_total_distance = total_distance

            n_iter += 1

        return best_labels, best_centers, best_total_distance, n_iter
