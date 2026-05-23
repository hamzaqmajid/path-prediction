import osmnx as ox
import networkx as nx
import numpy as np
import torch
import random


def load_graph(place_name="Cottbus, Germany"):
    G = ox.graph_from_place(place_name, network_type='drive')
    return G


def build_graph_data(G):
    # Step 1: Map node IDs → indices
    nodes = list(G.nodes())
    node_id_map = {node: i for i, node in enumerate(nodes)}

    # Step 2: Node features (lat, lon, degree)
    features = []

    for node in nodes:
        lat = G.nodes[node]['y']
        lon = G.nodes[node]['x']
        degree = G.degree[node]

        features.append([lat, lon, degree])

    x = torch.tensor(features, dtype=torch.float)

    # Step 3: Build edge index
    edges = []

    for u, v in G.edges():
        edges.append([node_id_map[u], node_id_map[v]])
        edges.append([node_id_map[v], node_id_map[u]])  # undirected

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    return x, edge_index, node_id_map


def generate_random_trajectory(G, node_id_map, walk_length=10):
    nodes = list(G.nodes())

    current = random.choice(nodes)
    trajectory = [node_id_map[current]]
    prev = None

    for _ in range(walk_length - 1):
        neighbors = list(G.neighbors(current))

        # remove going back immediately (key fix)
        if prev is not None and prev in neighbors:
            neighbors.remove(prev)

        if len(neighbors) == 0:
            break

        next_node = random.choice(neighbors)

        trajectory.append(node_id_map[next_node])

        prev = current
        current = next_node

    return trajectory


def generate_trajectories(G, node_id_map, num_traj=1000, walk_length=10):
    trajectories = []

    for _ in range(num_traj):
        traj = generate_random_trajectory(G, node_id_map, walk_length)
        
        if len(traj) == walk_length:
            trajectories.append(traj)

    return trajectories