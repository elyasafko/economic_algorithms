def max_mean_cycle(graph):
    """
    Given a directed graph as an adjacency dict mapping u -> list of (v, weight),
    returns (cycle_vertices, mean_weight) where 'cycle_vertices' is a list
    of nodes forming a cycle of maximum average weight, and 'mean_weight' is
    that average.
    """
    # 1) Collect all vertices
    vertices = list(graph.keys())
    for u in graph:
        for v, _ in graph[u]:
            if v not in vertices:
                vertices.append(v)
    n = len(vertices)
    idx = {v: i for i, v in enumerate(vertices)}

    # 2) DP arrays: f[k][i] = best weight of a k-edge walk ending at i
    NEG_INF = float('-inf')
    f = [[NEG_INF]*n for _ in range(n+1)]
    pred = [[None]*n for _ in range(n+1)]
    # base case: 0 edges has weight 0
    for i in range(n):
        f[0][i] = 0

    # 3) Fill DP for k = 1..n
    for k in range(1, n+1):
        for u in graph:
            ui = idx[u]
            prev_w = f[k-1][ui]
            if prev_w == NEG_INF:
                continue
            for v, w_uv in graph[u]:
                vi = idx[v]
                cand = prev_w + w_uv
                if cand > f[k][vi]:
                    f[k][vi] = cand
                    pred[k][vi] = ui

    # 4) Compute maximum mean mu
    best_mu = NEG_INF
    best_v = best_k = None
    for vi in range(n):
        if f[n][vi] == NEG_INF:
            continue
        # for this vertex, find the k minimizing the average
        worst_avg = float('inf')
        arg_k = None
        for k in range(n):
            if f[k][vi] == NEG_INF:
                continue
            avg = (f[n][vi] - f[k][vi]) / (n - k)
            if avg < worst_avg:
                worst_avg = avg
                arg_k = k
        # now worst_avg is the max‐mean for cycles reachable to vi
        if arg_k is not None and worst_avg > best_mu:
            best_mu = worst_avg
            best_v, best_k = vi, arg_k

    if best_v is None:
        # no cycles at all
        return [], 0.0

    # 5) Reconstruct any walk of length n ending at best_v
    walk = []
    cur = best_v
    for k in range(n, 0, -1):
        walk.append(cur)
        cur = pred[k][cur]
    walk.append(cur)
    walk.reverse()  # now walk[0..n] is the sequence of n edges

    # 6) Extract the first repeated vertex to get a cycle
    seen = {}
    for i, v in enumerate(walk):
        if v in seen:
            j = seen[v]
            cycle_idxs = walk[j:i+1]
            cycle = [vertices[x] for x in cycle_idxs]
            return cycle, best_mu
        seen[v] = i

    # fallback (shouldn't happen if there's any cycle)
    return [], best_mu


# -------------------
# Example usage/tests
# -------------------
if __name__ == "__main__":
    # --- existing tests ---
    G1 = {
      'A': [('B',1)],
      'B': [('C',1)],
      'C': [('A',1)]
    }
    cycle1, mu1 = max_mean_cycle(G1)
    print("G1 (triangle):", cycle1, mu1)  # expect a 3-cycle, mu=1

    G2 = {
      'A': [('B',2),('C',0)],
      'B': [('A',2)],
      'C': [('D',1)],
      'D': [('C',1)]
    }
    cycle2, mu2 = max_mean_cycle(G2)
    print("G2 (two 2-vertex cycles):", cycle2, mu2)  # expect A<->B, mu=2

    G3 = {
      1: [(2, -1)],
      2: [(3, -2)],
      3: [(1, -3)],
      4: [(5, 5)],
      5: [(4, 5)]
    }
    cycle3, mu3 = max_mean_cycle(G3)
    print("G3 (mixed signs):", cycle3, mu3)  # expect 4<->5, mu=5

    # --- new tests ---

    # 4) No cycles at all
    G4 = {
      'X': [('Y', 10)],
      'Y': [('Z', -5)],
      'Z': []  # no way back
    }
    cycle4, mu4 = max_mean_cycle(G4)
    print("G4 (no cycles):", cycle4, mu4)  # expect [], 0.0 by convention

    # 5) Single self-loop
    G5 = {
      'A': [('A', 3)],
      'B': [('C', 1)],
      'C': [('B', 1)]
    }
    cycle5, mu5 = max_mean_cycle(G5)
    print("G5 (self-loop vs 2-cycle):", cycle5, mu5)
    # expect ['A','A'], mu=3 since 3 > (1+1)/2 = 1

    # 6) Two disjoint 3-cycles with same mean
    G6 = {
      'P': [('Q', 4)], 'Q': [('R', 2)], 'R': [('P', 0)],  # mean = (4+2+0)/3 = 2
      'X': [('Y', 3)], 'Y': [('Z', 3)], 'Z': [('X', 0)]   # mean = (3+3+0)/3 = 2
    }
    cycle6, mu6 = max_mean_cycle(G6)
    print("G6 (equal-mean 3-cycles):", cycle6, mu6)  # any 3-cycle, mu=2

    # 7) Longer cycle vs short high-weight cycle
    G7 = {
      1: [(2,10)], 2: [(3,10)], 3: [(4,10)], 4: [(1,10)],   # 4-cycle mean=10
      5: [(6,12)], 6: [(5,8)]                               # 2-cycle mean=(12+8)/2 =10
    }
    cycle7, mu7 = max_mean_cycle(G7)
    print("G7 (4-cycle vs 2-cycle both μ=10):", cycle7, mu7)
    # mu=10, but you may get either the 4-cycle or the 2-cycle

    # 8) Random small graph test
    import random
    random.seed(42)
    # build a 5-node fully connected graph with weights in [-5, 5]
    nodes = list(range(5))
    G8 = {u: [] for u in nodes}
    for u in nodes:
        for v in nodes:
            if u != v:
                G8[u].append((v, random.randint(-5, 5)))
    cycle8, mu8 = max_mean_cycle(G8)
    print("G8 (random 5-node):", cycle8, mu8)
    # just sanity-check that it runs without crashing

    print("\nAll tests completed.")

