import osmnx as ox
import numpy as np
import torch
import random
 
 
def load_graph(place_name="Cottbus, Germany"):
    G = ox.graph_from_place(place_name, network_type='drive')
    return G
 
 
def build_graph_data(G):
    # Map OSM node IDs → contiguous indices 0..N-1
    nodes = list(G.nodes())
    node_id_map = {node: i for i, node in enumerate(nodes)}
 
    # Node features: (lat, lon, degree)
    features = []
    for node in nodes:
        lat = G.nodes[node]['y']
        lon = G.nodes[node]['x']
        degree = G.degree[node]
        features.append([lat, lon, degree])
 
    x = torch.tensor(features, dtype=torch.float)
 
    # Edge index — add both directions so message passing is bidirectional
    edges = []
    for u, v in G.edges():
        edges.append([node_id_map[u], node_id_map[v]])
        edges.append([node_id_map[v], node_id_map[u]])
 
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
 
    return x, edge_index, node_id_map
 
 
def random_walk(G, start, length=8):
    walk = [start]
    prev = None
 
    for _ in range(length - 1):
        current = walk[-1]
        neighbors = list(G.neighbors(current))
 
        if not neighbors:
            break  # dead-end node — stop early
 
        # Anti-backtracking: exclude the previous node if alternatives exist
        forward_neighbors = [n for n in neighbors if n != prev]
        candidates = forward_neighbors if forward_neighbors else neighbors
 
        next_node = random.choice(candidates)
        walk.append(next_node)
        prev = current
 
    return walk
 
 
def generate_trajectories(G, node_map, num_traj=5000, length=8):
    nodes = list(G.nodes())
    trajectories = []
 
    for _ in range(num_traj):
        start = random.choice(nodes)            # uniform start node
        traj = random_walk(G, start, length)    # OSM IDs
        traj = [node_map[n] for n in traj]      # → contiguous indices
        trajectories.append(traj)
 
    return trajectories
 
 
def train_val_test_split(trajectories, val_ratio=0.1, test_ratio=0.1, seed=42):
    """ 
    Default: 80 % train, 10 % val, 10 % test.
    """
    rng = random.Random(seed)
    data = trajectories[:]
    rng.shuffle(data)
 
    n = len(data)
    n_test = int(n * test_ratio)
    n_val  = int(n * val_ratio)
 
    test  = data[:n_test]
    val   = data[n_test: n_test + n_val]
    train = data[n_test + n_val:]
 
    return train, val, test