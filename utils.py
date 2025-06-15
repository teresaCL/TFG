import networkx as nx

"""
    Constructs must-link and cannot-link groups for constrained clustering.

    Parameters:
    -----------
        D (np.ndarray): Data matrix of dimension (N, d), where N is the number of samples and d is the feature dimension.
        ML: List of tuples (i, i'). Must-link constraints: points i and i' must belong to the same cluster.
        CL: List of tuples (i, i'). Cannot-link constraints: points i and i' cannot belong to the same cluster.

    Returns:
    --------
        ML_groups (list of sets): Each set represents a group of points that must be in the same cluster.  
        CL_groups (list of sets): The set at the i-th position contains the indices of the ML groups that cannot be in the same cluster as group i.
"""
def create_groups(D, ML, CL):
    # Groups nodes into connected components from the ML edge list.
    ML_groups = [concom for concom in nx.connected_components(nx.from_edgelist(ML))]

    # Adds unconnected indices as individual groups to ensure full coverage.
    for idx_p in range(len(D)):
        already_added = False
        for ml_gr in ML_groups:
            if idx_p in ml_gr:
                already_added = True
                break
        
        if not already_added:
            ML_groups.append({idx_p})

    # Initialize the CL groups
    CL_groups = []
    for _ in ML_groups:
        CL_groups.append([])

    # For each cannot-link pair, find their group indices in ML_groups
    for cl in CL:
        found_0 = False
        found_1 = False

        for idx_ml_gr in range(len(ML_groups)):
            # Check if first element of the pair is in the current group
            if cl[0] in ML_groups[idx_ml_gr]:
                found_0 = True
                idx_0 = idx_ml_gr
            
            # Check if second element of the pair is in the current group
            if cl[1] in ML_groups[idx_ml_gr]:
                found_1 = True
                idx_1 = idx_ml_gr

            # Stop searching once both indices are found
            if found_0 and found_1:
                break

        assert idx_0 != idx_1

        # Register the cannot-link constraint between groups
        CL_groups[idx_0].append(idx_1)
        CL_groups[idx_1].append(idx_0)

    # Convert CL group lists to sets to remove duplicates
    for idx_cl_gr in range(len(CL_groups)):
        CL_groups[idx_cl_gr] = set(CL_groups[idx_cl_gr])

    return ML_groups, CL_groups