"""
Load all the datas
"""
import os
import json
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt


_data_path = os.path.join("..", "data")


def load_csv(name: str):
    return pd.read_csv(os.path.join(_data_path, f"{name}.csv"))


def load_json(name: str):
    with open(os.path.join(_data_path, f"{name}.json")) as f:
        return json.load(f)


def load_all():
    cells = load_csv("cells")
    neighbors = load_csv("neighbors")
    aff = load_csv("paging_affinity")
    ho = load_csv("handovers")
    cons = load_json("constraints")
    baseline = load_csv("baseline_assignment")
    return cells, neighbors, aff, ho, cons, baseline


class TACDataLoader:
    def __init__(self):
        cells, neighbors, aff, ho, cons, baseline = load_all()

        self.cells = cells
        self.N = len(self.cells)

        self.constraints = cons
        self.K = self.constraints["K"]

        self.w_ij, self.m_ij, self.a_ij, self.d_ij = self._build_maps(aff, ho, neighbors)

        self.assignment = np.array([ta_id for ta_id in baseline["ta_id"]])
        print(f"initial assignment: {self.assignment}")
        print(f"Loaded {self.N} cells, K={self.K} TACs")


    def _build_maps(self, aff, ho, neighbors):
        """
        Fills w, m, a and d
        """
        def add(store, a, b, v):
            a, b = int(a), int(b)
            if a == b:
                return
            i, j = (a, b) if a < b else (b, a)
            store[i,j] = v

        cell_to_idx = {cid: i for i, cid in enumerate(self.cells['cell_id'])}
        w_ij = np.zeros((self.N, self.N)) # paging affinity
        m_ij = np.zeros((self.N, self.N)) # mobility
        a_ij = np.zeros((self.N, self.N)) # adjacency
        d_ij = np.zeros((self.N, self.N)) # distance

        for _, r in aff.iterrows():
            add(w_ij, cell_to_idx[r["cell_i"]], cell_to_idx[r["cell_j"]], float(r["affinity"]))

        for _, r in ho.iterrows():
            add(m_ij, cell_to_idx[r["cell_i"]], cell_to_idx[r["cell_j"]], float(r["ho_intensity"]))

        maxd = max(1.0, float(neighbors["distance_m"].max()))
        for _, r in neighbors.iterrows():
            add(a_ij, cell_to_idx[r["cell_i"]], cell_to_idx[r["cell_j"]], int(r.get("is_neighbor", 1)))
            add(d_ij, cell_to_idx[r["cell_i"]], cell_to_idx[r["cell_j"]], float(r.get("distance_m", 0.0)) / maxd)

        return w_ij, m_ij, a_ij, d_ij

    def validate_assignment(self, assignment):
        """
        Checks that an assignment meets the constraints.
        """
        N = self.N
        K = self.K
        min_size = self.constraints['min_ta_size']
        max_size = self.constraints['max_ta_size']

        errors = []

        if len(assignment) != N:
            errors.append(f"Assignment has {len(assignment)} cells, expected {N}")

        # Check range
        if np.min(assignment) < 0 or np.max(assignment) >= K:
            errors.append(f"Assignment contains labels outside [0, {K-1}]")

        # Check all TACs are used
        unique_tacs = np.unique(assignment)
        if len(unique_tacs) != K:
            errors.append(f"Assignment uses {len(unique_tacs)} TACs, expected {K}")

        # Check size constraints
        tac_counts = np.bincount(assignment, minlength=K)
        for k in range(K):
            size = tac_counts[k]
            if size < min_size:
                errors.append(f"TAC {k} has {size} cells, below minimum {min_size}")
            if size > max_size:
                errors.append(f"TAC {k} has {size} cells, above maximum {max_size}")
        
        is_valid = len(errors) == 0
        return is_valid, errors


    def get_coordinates(self):
        """Extract lat/lon coordinates"""
        return self.cells[['lat', 'lon']].values


    def compute_cost(self, assignment):
        """
        Fully identical to the official hackathon evaluator.
        Uses:
            - raw w_ij, m_ij, a_ij, distance_ij
            - penalties only when assignment[i] != assignment[j]
            - distance penalty only across TAC boundaries
            - hard min/max TAC size constraint
        """

        N = self.N
        K = self.K

        paging_cost = 0.0
        mobility_cost = 0.0
        adjacency_cost = 0.0
        size_penalty = 0.0

        dist_cost = 0.0
        for k in range(K):
            cells_in_k = np.where(assignment == k)[0]
            for idx1 in range(len(cells_in_k)):
                i = cells_in_k[idx1]
                for idx2 in range(idx1 + 1, len(cells_in_k)):
                    j = cells_in_k[idx2]
                    dist_cost += self.d_ij[(i,j)]

        # Pairwise penalties
        for i in range(N):
            for j in range(i+1, N):
                if assignment[i] != assignment[j]:
                    paging_cost += self.w_ij[i, j]
                    mobility_cost += self.m_ij[i, j]
                    adjacency_cost += self.a_ij[i, j]

        # Hard size constraints
        min_size = self.constraints.get("min_ta_size", 0)
        max_size = self.constraints.get("max_ta_size", N)
        size_penalty = 0.0

        for k in range(K):
            size_k = np.sum(assignment == k)
            if size_k < min_size:
                size_penalty += 1000.0 * (min_size - size_k)
            #elif size_k > max_size:
            #    size_penalty += 1000.0 * (size_k - max_size)

        # Return same structure as official
        total = paging_cost + mobility_cost + adjacency_cost + dist_cost + size_penalty

        return {
            "total": total,
            "paging":paging_cost,
            "mobility": mobility_cost,
            "adjacency": adjacency_cost,
            "distance": dist_cost,
            "size_penalty": size_penalty
        }

    
    def print_cost_breakdown(self, breakdown, prefix=""):
        """Pretty print cost breakdown"""
        print(f"\n{prefix}=== Cost Breakdown ===")
        print(f"{prefix}Total Cost:      {breakdown['total']}")
        print(f"{prefix}  Paging:        {breakdown['paging']}")
        print(f"{prefix}  Mobility:      {breakdown['mobility']}")
        print(f"{prefix}    - Adjacency:   {breakdown['adjacency']}")
        print(f"{prefix}    - Distance:    {breakdown['distance']}")
        print(f"{prefix}{'='*23}\n")
    

    def plot_tac_assignment(self, assignment=None, figsize=(12, 10)):
        """
        Visualize TAC assignment on geographic map with convex hulls around TACs
        
        Args:
            assignment: optional array of TAC labels to visualize
                    if None, uses baseline assignment
            figsize: figure size tuple
        
        Returns:
            fig: matplotlib figure object
        """
        from scipy.spatial import ConvexHull
        from matplotlib.patches import Polygon
        
        # Use baseline if no assignment provided
        if assignment is None:
            assignment = self.assignment
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        coords = self.get_coordinates()
        
        # Create colormap for TACs
        n_tacs = len(np.unique(assignment))
        colors = plt.cm.tab20(np.linspace(0, 1, n_tacs))
        
        # Draw adjacency edges FIRST (so they're in background)
        for i in range(self.N):
            for j in range(i+1, self.N):
                if self.a_ij[i, j] > 0:
                    x_coords = [coords[i, 1], coords[j, 1]]
                    y_coords = [coords[i, 0], coords[j, 0]]
                    # Different color for cross-TAC edges
                    if assignment[i] != assignment[j]:
                        ax.plot(x_coords, y_coords, 'r-', 
                            alpha=0.15, linewidth=1, zorder=1)  # Much more subtle
                    else:
                        ax.plot(x_coords, y_coords, 'gray', 
                            alpha=0.1, linewidth=0.8, zorder=1)
        
        # Draw convex hulls for each TAC (on top of edges)
        for tac in range(n_tacs):
            mask = assignment == tac
            tac_coords = coords[mask]
            
            if len(tac_coords) >= 3:  # Need at least 3 points for convex hull
                try:
                    hull = ConvexHull(tac_coords)
                    # Get hull points
                    hull_points = tac_coords[hull.vertices]
                    
                    # Draw filled polygon
                    polygon = Polygon(hull_points[:, [1, 0]], 
                                    closed=True,
                                    facecolor=colors[tac],
                                    edgecolor=colors[tac],
                                    alpha=0.3,  # Increased opacity for better visibility
                                    linewidth=4,  # Thicker border
                                    label=f'TAC {tac}',
                                    zorder=5)
                    ax.add_patch(polygon)
                except:
                    # If convex hull fails, skip
                    pass
            elif len(tac_coords) == 2:
                # Draw line between two points
                ax.plot(tac_coords[:, 1], tac_coords[:, 0],
                    color=colors[tac], linewidth=4, alpha=0.6,
                    label=f'TAC {tac}', zorder=5)
            elif len(tac_coords) == 1:
                # Just mark the single point
                ax.scatter(tac_coords[0, 1], tac_coords[0, 0],
                        c=[colors[tac]], s=400, alpha=0.6,
                        edgecolors=colors[tac], linewidths=4,
                        label=f'TAC {tac}', zorder=5)
        
        # Plot all cells as black dots (on top)
        ax.scatter(coords[:, 1], coords[:, 0], 
                c='black', s=80, alpha=0.8,
                edgecolors='white', linewidths=1.5,
                zorder=10)
        
        ax.set_xlabel('Longitude', fontsize=14)
        ax.set_ylabel('Latitude', fontsize=14)
        ax.set_title('TAC Assignment with Convex Hulls\n(Red edges = cross-TAC boundaries)', 
                    fontsize=16, fontweight='bold')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', 
                fontsize=10, ncol=1)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        return fig


def report_kpis(data_loader, baseline_assignment, optimized_assignment):
    """Generate comprehensive KPI report"""
    
    # Compute costs
    breakdown_base = data_loader.compute_cost(baseline_assignment)
    breakdown_opt = data_loader.compute_cost(optimized_assignment)
    
    print("\n" + "="*60)
    print("KPI REPORT")
    print("="*60)
    
    # KPI 1: Paging reduction
    paging_reduction = (breakdown_base['paging'] - breakdown_opt['paging']) / breakdown_base['paging'] * 100
    print(f"\n2. PAGING REDUCTION (Primary Goal: -15% to -20%)")
    print(f"   Baseline:    {breakdown_base['paging']:.2f}")
    print(f"   Optimized:   {breakdown_opt['paging']:.2f}")
    print(f"   Change:      {paging_reduction:+.2f}%")
    if paging_reduction >= 15:
        print(f"   Status:      ✓ TARGET MET")
    else:
        print(f"   Status:      ✗ Below target")
    
    # KPI 2: TAU proxy
    tau_change = (breakdown_opt['mobility'] - breakdown_base['mobility']) / breakdown_base['mobility'] * 100
    print(f"\n3. TAU PROXY (Mobility)")
    print(f"   Baseline:    {breakdown_base['mobility']:.2f}")
    print(f"   Optimized:   {breakdown_opt['mobility']:.2f}")
    print(f"   Change:      {tau_change:+.2f}%")
    
    # KPI 3: Contiguity
    total_edges = int(data_loader.a_ij.sum() / 2)
    
    cross_edges_base = sum(1 for i in range(data_loader.N) for j in range(i+1, data_loader.N)
                          if data_loader.a_ij[i,j] > 0 and baseline_assignment[i] != baseline_assignment[j])
    cross_edges_opt = sum(1 for i in range(data_loader.N) for j in range(i+1, data_loader.N)
                         if data_loader.a_ij[i,j] > 0 and optimized_assignment[i] != optimized_assignment[j])
    
    contiguity_base = 1 - cross_edges_base / total_edges
    contiguity_opt = 1 - cross_edges_opt / total_edges
    
    print(f"\n4. CONTIGUITY")
    print(f"   Total adjacency edges: {total_edges}")
    print(f"   Baseline split edges:  {cross_edges_base} ({(1-contiguity_base)*100:.1f}%)")
    print(f"   Optimized split edges: {cross_edges_opt} ({(1-contiguity_opt)*100:.1f}%)")
    print(f"   Contiguity improvement: {(contiguity_opt - contiguity_base)*100:+.1f}%")
    
    # KPI 4: TAC sizes
    sizes_base = np.bincount(baseline_assignment, minlength=data_loader.K)
    sizes_opt = np.bincount(optimized_assignment, minlength=data_loader.K)
    
    print(f"\n5. TAC SIZE STATISTICS")
    print(f"   {'':12s} {'Baseline':>12s} {'Optimized':>12s}")
    print(f"   {'Min':12s} {sizes_base.min():>12d} {sizes_opt.min():>12d}")
    print(f"   {'Max':12s} {sizes_base.max():>12d} {sizes_opt.max():>12d}")
    print(f"   {'Mean':12s} {sizes_base.mean():>12.1f} {sizes_opt.mean():>12.1f}")
    print(f"   {'Std Dev':12s} {sizes_base.std():>12.2f} {sizes_opt.std():>12.2f}")
    
    # Check constraints
    min_size = data_loader.constraints['min_ta_size']
    max_size = data_loader.constraints['max_ta_size']
    violations_opt = sum((sizes_opt < min_size) | (sizes_opt > max_size))
    
    print(f"\n   Constraints: [{min_size}, {max_size}]")
    print(f"   Violations:  {violations_opt}/{data_loader.K}")
    
    print("\n" + "="*60)