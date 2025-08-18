import pulp as pl
import numpy as np
from typing import Callable

def inverse_class_count(n_objects: int,
                        n_classes: int,
                        class_counts_per_object: np.ndarray | dict[int,list[int]]) -> dict[int,int]:
    
    I = list(range(n_objects)) # object indexes
    C = list(range(n_classes)) # classes
    w = {(i,c): class_counts_per_object[i][c] for i in I for c in C}  # class counts
    T = {c: sum(w[i,c] for i in I) for c in C} # number of counts in each class
    gamma = {c: 1.0 / max(1, T[c]) for c in C} # nomralization
    return gamma

def stratified_k_way_split_respecting_partition(n_objects: int,
                                  n_classes: int,
                                  class_counts_per_object: np.ndarray | dict[int,list[int]],
                                  split_ratios: list[float],
                                  lamda_card = 1.0,
                                  lamda_cls = 10.0,
                                  normalization_strategy: Callable = inverse_class_count,
                                  enforce_object_in_each_partition = False,
    ):
    """
    Solve a stratified k-way partitioning problem using Mixed Integer Linear Programming (MILP).

    This algorithm partitions a set of objects into `k` disjoint subsets (partitions) such that:

    - The number of objects assigned to each partition approximately respects the desired
      `split_ratios`.
    - The class distribution of objects in each partition is approximately proportional to the 
      global class distribution across all objects.

    The problem is formulated as an MILP with binary assignment variables, slack variables for 
    deviations in cardinality and class proportions, and a weighted objective that balances 
    between enforcing exact partition sizes and stratified class distributions.

    Args:
        n_objects (int): 
            Total number of objects to partition.
        n_classes (int): 
            Total number of classes across the dataset.
        class_counts_per_object (np.ndarray | dict[int, list[int]]): 
            Class counts per object, indexed either as a NumPy array of shape (n_objects, n_classes) 
            or a dictionary mapping object index → list of class counts.
        split_ratios (list[float]): 
            Desired proportions of the split. Must sum to 1.0. 
            For example, [0.8, 0.2] splits into train/test with an 80/20 ratio.
        lamda_card (float, optional): 
            Weight in the objective for enforcing cardinality balance across partitions.
            Larger values enforce partition sizes more strictly. Default is 1.0.
        lamda_cls (float, optional): 
            Weight in the objective for enforcing stratified class distribution. 
            Larger values enforce proportional class balance more strictly. Default is 10.0.
        normalization_strategy (Callable, optional): 
            Function that defines normalization weights for class deviation penalties. 
            Default is `inverse_class_count`, which downweights frequent classes.
        enforce_object_in_each_partition (bool, optional):
            Decide to enforce at least one object type in each partition. Not all
            datasets can satisfy this requirement (default False)

    Returns:
        dict[int, int]: 
            A mapping from object index → assigned partition index (0 to k-1).

    Raises:
        ValueError: If `split_ratios` do not sum to 1.0.

    Notes:
        - The optimization objective is:
          
          minimize   λ_card * Σ_j (u_plus[j] + u_minus[j]) 
                   + λ_cls  * Σ_j Σ_c γ[c] * (e_plus[j][c] + e_minus[j][c])

          where:
            * u_plus, u_minus are slack variables for partition size deviations
            * e_plus, e_minus are slack variables for class distribution deviations
            * γ[c] are normalization weights for each class
        - The solution is exact up to solver tolerance but may be sensitive to large datasets 
          due to MILP complexity.
    """    
    sum_ratios = sum([ratio for ratio in split_ratios])
    if sum_ratios != 1.0:
        raise ValueError("Ratios must sum to one")
    num_partitions = len(split_ratios)
    # Inputs
    I = list(range(n_objects)) # object indexes
    C = list(range(n_classes)) # classes
    K = list(range(num_partitions)) # partition indexes
    w = {(i,c): class_counts_per_object[i][c] for i in I for c in C}  # class counts
    r = {j: split_ratios[j] for j in K}                     # partition weights (sum to 1)
    N = len(I) # number of objects
    T = {c: sum(w[i,c] for i in I) for c in C} # number of counts in each class
    gamma = normalization_strategy(n_objects,n_classes,class_counts_per_object)
    # tunable relative weights 
    # higher lambda_cls means more focus on 
    # enforcing classification, whereas higher lambda_card
    # means enforce the k-way split condition more strongly.
    lamda_card, lamda_cls = 1.0, 10.0                  

    # Model instantiation
    m = pl.LpProblem("k_way_vector_partition_with_ratios", pl.LpMinimize)

    # Vars
    # assignement x_ij = 1 if object i is assigned to partition j and zero otherwise
    x = pl.LpVariable.dicts("x", (I, K), 0, 1, cat=pl.LpBinary) 

    # partition cardinality slack variables
    u_p = pl.LpVariable.dicts("u_plus", K, lowBound=0)
    u_m = pl.LpVariable.dicts("u_minus", K, lowBound=0)

    # class count slack variables
    e_p = pl.LpVariable.dicts("e_plus", (K, C), lowBound=0)
    e_m = pl.LpVariable.dicts("e_minus", (K, C), lowBound=0)

    # Assignment constraint (all objects need to be assigned a partition)
    for i in I:
        m += pl.lpSum(x[i][j] for j in K) == 1

    # Cardinality deviations 
    # (u_p-u_m constraint is a common absolute value reduction technique)
    for j in K:
        m += (pl.lpSum(x[i][j] for i in I) - r[j]*N) == (u_p[j] - u_m[j])

    if enforce_object_in_each_partition:
        for j in K:
            for c in C:
                if T[c] > 0:                       # skip empty classes
                    m += pl.lpSum(w[i,c] * x[i][j] for i in I) >= 1 # number of objects of class c in partition j

    # Class deviations
    for j in K:
        for c in C:
            m += (pl.lpSum(w[i,c]*x[i][j] for i in I) - r[j]*T[c]) == (e_p[j][c] - e_m[j][c])

    # Objective
    m += lamda_card * pl.lpSum(u_p[j] + u_m[j] for j in K) \
    + lamda_cls  * pl.lpSum(gamma[c]*(e_p[j][c] + e_m[j][c]) for j in K for c in C)

    m.solve(pl.PULP_CBC_CMD(msg=False))
    assignments = {i: max(K, key=lambda j: pl.value(x[i][j])) for i in I}
    return assignments
