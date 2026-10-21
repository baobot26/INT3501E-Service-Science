"""Tính khoảng cách đi bộ từ mỗi điểm nhu cầu đến mỗi điểm ứng viên.

Đọc từ data/: demand.csv, candidates.csv, walk_network.graphml
Ghi ra data/dist.npy: ma trận (số điểm nhu cầu x số điểm ứng viên), đơn vị mét.
Ô nào không đi tới được trong CUTOFF_M thì để inf.
"""

import math
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

DATA = Path("data")
CUTOFF_M = 6000    # quãng đường đi bộ tối đa tính đến
MAX_SNAP_M = 300   # điểm cách đường xa hơn mức này thì báo
RADII = [300, 500, 800, 1000]

# đổi lat/lon ra mét bằng công thức gần đúng, đủ chính xác với khu vực vài km
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
    G = nx.Graph()  # coi đường đi bộ là hai chiều
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

    # mỗi điểm gắn vào nút đường gần nhất, cộng thêm đoạn đi từ điểm đó ra nút
    d_snap, d_idx = tree.query(to_xy(demand["lon"], demand["lat"]))
    c_snap, c_idx = tree.query(to_xy(cand["lon"], cand["lat"]))
    print(f"điểm nhu cầu: {len(demand)}, điểm ứng viên: {len(cand)}, nút đường: {len(nodes)}")
    print(f"điểm nhu cầu cách đường xa nhất {d_snap.max():.0f} m, "
          f"{(d_snap > MAX_SNAP_M).sum()} điểm xa hơn {MAX_SNAP_M} m")
    print(f"điểm ứng viên cách đường xa nhất {c_snap.max():.0f} m, "
          f"{(c_snap > MAX_SNAP_M).sum()} điểm xa hơn {MAX_SNAP_M} m")

    dist = np.full((len(demand), len(cand)), np.inf, dtype=np.float32)
    demand_nodes = [nodes[k] for k in d_idx]
    for j, k in enumerate(c_idx):
        lengths = nx.single_source_dijkstra_path_length(
            G, nodes[k], cutoff=CUTOFF_M, weight="length"
        )
        on_road = np.array([lengths.get(n, np.inf) for n in demand_nodes])
        dist[:, j] = on_road + d_snap + c_snap[j]

    np.save(DATA / "dist.npy", dist)
    print(f"đã lưu data/dist.npy {dist.shape}")
    print(f"cặp đi tới được trong {CUTOFF_M} m: {np.isfinite(dist).mean() * 100:.1f}%")

    # nếu đặt tủ ở tất cả điểm ứng viên thì phủ được bao nhiêu dân
    pop = demand["population"].to_numpy()
    nearest = dist.min(axis=1)
    print("nếu đặt ở tất cả điểm ứng viên:")
    for r in RADII:
        share = pop[nearest <= r].sum() / pop.sum() * 100
        print(f"  bán kính {r} m phủ {share:.1f}% dân")


if __name__ == "__main__":
    main()
