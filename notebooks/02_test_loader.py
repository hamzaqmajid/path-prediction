from src.data_loader import load_graph, build_graph_data

G = load_graph("Cottbus, Germany")

x, edge_index, node_map = build_graph_data(G)

print("Node feature shape:", x.shape)
print("Edge index shape:", edge_index.shape)
print("Total nodes:", len(node_map))