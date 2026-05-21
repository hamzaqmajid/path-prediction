import osmnx as ox
import networkx as nx
import numpy as np
import torch

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