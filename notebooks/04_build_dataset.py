from src.data_loader import load_graph, build_graph_data, generate_trajectories
from src.dataset import build_sequence_dataset

G = load_graph("Cottbus, Germany")
x, edge_index, node_map = build_graph_data(G)

trajectories = generate_trajectories(G, node_map, num_traj=20, walk_length=8)

inputs, targets = build_sequence_dataset(trajectories)

print("Total samples:", len(inputs))
print("Example input:", inputs[0])
print("Example target:", targets[0])