import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from itertools import permutations

def perfect_matching(matrix: np.ndarray, tol: float = 1e-8):
    """
    Brute-force any perfect matching in a square matrix.
    Returns a list of (row, col) index pairs, or None if none exists.
    """
    n, m = matrix.shape
    if n != m:
        return None
    for perm in permutations(range(n)):
        if all(matrix[i, perm[i]] > tol for i in range(n)):
            return list(zip(range(n), perm))
    return None

def visual_birkhoff(matrix: np.ndarray,
                    left_labels: list[str],
                    right_labels: list[str],
                    tol: float = 1e-8):
    """
    Perform the Birkhoff–von Neumann peel-off algorithm AND visualize each step:
     - All positive-weight edges are drawn with line width ∝ current weight
     - The chosen perfect matching is overlaid in red
    """
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Matrix must be square for a perfect matching decomposition.")

    mat = matrix.astype(float).copy()
    step = 0

    while np.any(mat > tol):
        step += 1
        pm_idx = perfect_matching(mat, tol)
        if pm_idx is None:
            print(f"❌ Step {step}: no perfect matching found; stopping.")
            break

        # Compute δ and convert to label-pairs
        δ = min(mat[i,j] for i,j in pm_idx)
        pm_labels = [(left_labels[i], right_labels[j]) for i,j in pm_idx]
        print(f"Step {step}: δ = {δ:.4f}, matching = {pm_labels}")

        # Build the bipartite graph of current weights
        G = nx.Graph()
        # add nodes
        G.add_nodes_from(left_labels, bipartite=0)
        G.add_nodes_from(right_labels, bipartite=1)
        # add edges for all positive weights
        for i, u in enumerate(left_labels):
            for j, v in enumerate(right_labels):
                w = mat[i,j]
                if w > tol:
                    G.add_edge(u, v, weight=w)

        # layout: left at x=0, right at x=1
        pos = {u: (0, -i) for i, u in enumerate(left_labels)}
        pos.update({v: (1, -j) for j, v in enumerate(right_labels)})

        plt.figure(figsize=(6,4))
        nx.draw_networkx_nodes(G, pos, node_size=500)
        nx.draw_networkx_labels(G, pos, font_size=10)

        all_edges = list(G.edges())
        widths   = [G[u][v]['weight'] * 8 for u,v in all_edges]
        nx.draw_networkx_edges(G, pos, edgelist=all_edges, width=widths, alpha=0.6)

        nx.draw_networkx_edges(G, pos, edgelist=pm_labels, width=3, edge_color='red')

        plt.title(f"Step {step}: δ = {δ:.3f}")
        plt.axis('off')
        plt.tight_layout()
        plt.show()

        # subtract δ from each matched edge
        for i, j in pm_idx:
            mat[i, j] -= δ

def build_matrix_from_edges(left_labels: list[str],
                            right_labels: list[str],
                            edges: list[tuple[str, str, float]]):
    """
    Construct an n×n weight matrix from an explicit list of edges.
    Unspecified entries remain zero.
    """
    n, m = len(left_labels), len(right_labels)
    if n != m:
        raise ValueError("Left and right must have the same length for Birkhoff.")
    mat = np.zeros((n, m), dtype=float)
    idxL = {u:i for i, u in enumerate(left_labels)}
    idxR = {v:j for j, v in enumerate(right_labels)}
    for u, v, w in edges:
        mat[idxL[u], idxR[v]] = w
    return mat

if __name__ == "__main__":
    # — Example 1: Balanced 4×4 with all edges = 0.25 —
    left_4  = ["L0","L1","L2","L3"]
    right_4 = ["R0","R1","R2","R3"]
    M4 = np.full((4, 4), 0.25)
    print("=== Decomposing 4×4 uniform doubly‐stochastic matrix ===")
    visual_birkhoff(M4, left_4, right_4)

    # — Example 2: custom edges between a–d and bikta–tzrif—villa-martef
    left_custom  = ["a", "b", "c", "d"]
    right_custom = ["bikta", "tzrif", "villa", "martef"]
    edges_custom = [
        ("a", "bikta", 0.7), ("a", "tzrif", 0.3),
        ("b", "villa", 0.7), ("b", "martef", 0.3),
        ("c", "bikta", 0.3), ("c", "villa", 0.3), ("c", "martef", 0.4),
        ("d", "tzrif", 0.7), ("d", "martef", 0.3),
    ]
    M_custom = build_matrix_from_edges(left_custom, right_custom, edges_custom)
    print("\n=== Decomposing custom matrix ===")
    visual_birkhoff(M_custom, left_custom, right_custom)
