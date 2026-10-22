"""Compute walking distance from each demand point to each candidate point.

Reads from data/: demand.csv, candidates.csv, walk_network.graphml
Writes data/dist.npy: matrix (num demand points x num candidates), in meters.
Entries unreachable within CUTOFF_M are set to inf.
"""

import math
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

DATA = Path("data")
CUTOFF_M = 6000    # max walking distance to compute
MAX_SNAP_M = 300   # report points farther from the road than this
RADII = [300, 500, 800, 1000]

# approximate lat/lon to meters, accurate enough for a few-km area
LAT0 = 21.03
M_PER_DEG_LAT = 111_320
M_PER_DEG_LON = 111_320 * math.cos(math.radians(LAT0))


def to_xy(lon, lat):
    return np.column_stack([
        np.asarray(lon, dtype=float) * M_PER_DEG_LON,
        np.asarray(lat, dtype=float) * M_PER_DEG_LAT,
    ])


def load_graph():
    raw = nx.read_graphml(DATA / "walk_network.graphml")
    G = nx.Graph()  # treat walking paths as undirected
    for n, d in raw.nodes(data=True):
        G.add_node(n, x=float(d["x"]), y=float(d["y"]))
    for u, v, d in raw.edges(data=True):
        w = float(d["length"])
        if G.has_edge(u, v):
            w = min(w, G[u][v]["length"])
        G.add_edge(u, v, length=w)
    return G


def main():
    G = load_graph()
    nodes = list(G.nodes)
    node_xy = to_xy([G.nodes[n]["x"] for n in nodes], [G.nodes[n]["y"] for n in nodes])
    tree = cKDTree(node_xy)

    demand = pd.read_csv(DATA / "demand.csv")
    cand = pd.read_csv(DATA / "candidates.csv")

    # snap each point to its nearest road node, then add the off-road walk to that node
    d_snap, d_idx = tree.query(to_xy(demand["lon"], demand["lat"]))
    c_snap, c_idx = tree.query(to_xy(cand["lon"], cand["lat"]))
    print(f"demand points: {len(demand)}, candidates: {len(cand)}, road nodes: {len(nodes)}")
    print(f"demand points: max snap distance {d_snap.max():.0f} m, "
          f"{(d_snap > MAX_SNAP_M).sum()} points farther than {MAX_SNAP_M} m")
    print(f"candidates: max snap distance {c_snap.max():.0f} m, "
          f"{(c_snap > MAX_SNAP_M).sum()} points farther than {MAX_SNAP_M} m")

    dist = np.full((len(demand), len(cand)), np.inf, dtype=np.float32)
    demand_nodes = [nodes[k] for k in d_idx]
    for j, k in enumerate(c_idx):
        lengths = nx.single_source_dijkstra_path_length(
            G, nodes[k], cutoff=CUTOFF_M, weight="length"
        )
        on_road = np.array([lengths.get(n, np.inf) for n in demand_nodes])
        dist[:, j] = on_road + d_snap + c_snap[j]

    np.save(DATA / "dist.npy", dist)
    print(f"saved data/dist.npy {dist.shape}")
    print(f"reachable pairs within {CUTOFF_M} m: {np.isfinite(dist).mean() * 100:.1f}%")

    # population covered if a locker were placed at every candidate
    pop = demand["population"].to_numpy()
    nearest = dist.min(axis=1)
    print("if placed at every candidate point:")
    for r in RADII:
        share = pop[nearest <= r].sum() / pop.sum() * 100
        print(f"  radius {r} m covers {share:.1f}% of population")


if __name__ == "__main__":
    main()