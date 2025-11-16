# classical.py
"""
Spectral clustering + direct optimization
"""

from tac_data_loader import TACDataLoader
from sklearn.cluster import SpectralClustering
import numpy as np


def spectral_clustering(data_loader: TACDataLoader, sc_assign_labels, random_state=42):
    """
    Spectral clustering using precomputed affinity matrix built from
    alpha * w + beta * m + gamma * adjacency.
    """
    N = data_loader.N
    K = data_loader.K

    # Safely get hyperparameters with defaults
    alpha = float(data_loader.constraints.get("alpha", 1.0))
    beta = float(data_loader.constraints.get("beta", 1.0))
    gamma = float(data_loader.constraints.get("gamma", 1.0))

    W_similarity = (
        alpha * data_loader.w_ij +
        beta  * data_loader.m_ij +
        gamma * data_loader.a_ij
    )

    # Ensure non-negative and symmetric
    if W_similarity.min() < 0:
        W_similarity = W_similarity - W_similarity.min()
    W_similarity = (W_similarity + W_similarity.T) / 2.0
    np.fill_diagonal(W_similarity, 0.0)

    sc = SpectralClustering(
        n_clusters=K,
        affinity='precomputed',
        random_state=random_state,
        assign_labels=sc_assign_labels,
        n_init=10
    )

    labels = sc.fit_predict(W_similarity)

    # Evaluate
    breakdown = data_loader.compute_cost(labels)
    score = breakdown["total"]
    print(f"Spectral init: total={score:.2f} paging={breakdown['paging']:.2f} mobility={breakdown['mobility']:.2f}")

    return labels


def direct_optimization(data_loader: TACDataLoader, sc_assign_labels='kmeans', random_state=42, max_iters=100):
    """
    Simple greedy local search initialized by spectral clustering.
    Note: still O(N^2) compute_cost calls; for large N you should implement
    delta-cost updates (local updates) or use approximate evaluation.
    """
    np.random.seed(random_state)
    N = data_loader.N
    K = data_loader.K

    print(f"\n{'='*60}")
    print("DIRECT COST OPTIMIZATION")
    print(f"{'='*60}")

    # Initialization
    init_assignment = spectral_clustering(data_loader, sc_assign_labels=sc_assign_labels, random_state=random_state)
    assignment = init_assignment.copy()
    breakdown = data_loader.compute_cost(assignment)
    best_cost = breakdown["total"]
    best_assignment = assignment.copy()

    print(f"Initial cost: {best_cost:.2f}")

    min_size = int(data_loader.constraints.get("min_ta_size", 0))
    max_size = int(data_loader.constraints.get("max_ta_size", N))

    no_improvement_count = 0

    for iteration in range(max_iters):
        improved = False

        # Try N random single-cell moves per iteration
        for _ in range(N):
            cell_i = np.random.randint(0, N)
            current_tac = assignment[cell_i]

            # ensure we don't make TAC smaller than min_size
            if (assignment == current_tac).sum() <= min_size:
                continue

            new_tac = np.random.randint(0, K)
            if new_tac == current_tac:
                continue
            if (assignment == new_tac).sum() >= max_size:
                continue

            # Apply move
            assignment[cell_i] = new_tac
            new_break = data_loader.compute_cost(assignment)
            new_cost = new_break["total"]

            if new_cost < best_cost:
                best_cost = new_cost
                best_assignment = assignment.copy()
                improved = True
            else:
                # revert
                assignment[cell_i] = current_tac

        if not improved:
            no_improvement_count += 1
            if no_improvement_count >= 10:
                # perturb the best assignment slightly
                print(f"  Iter {iteration}: perturbing (stuck at {best_cost:.2f})")
                assignment = best_assignment.copy()
                for _ in range(5):
                    c = np.random.randint(0, N)
                    if (assignment == assignment[c]).sum() > min_size:
                        assignment[c] = np.random.randint(0, K)
                no_improvement_count = 0
        else:
            no_improvement_count = 0
            if iteration % 10 == 0:
                br = data_loader.compute_cost(best_assignment)
                print(f"  Iter {iteration}: best_cost={best_cost:.2f} paging={br['paging']:.2f} mobility={br['mobility']:.2f}")

    print("\nOptimization complete!")
    final_break = data_loader.compute_cost(best_assignment)
    print(f"Final cost: {final_break['total']:.2f}")
    print(f"  Paging: {final_break['paging']:.2f}; Mobility: {final_break['mobility']:.2f}")

    return best_assignment
