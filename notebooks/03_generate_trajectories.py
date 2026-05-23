from src.data_loader import load_graph, build_graph_data, generate_trajectories

G = load_graph("Cottbus, Germany")
x, edge_index, node_map = build_graph_data(G)

trajectories = generate_trajectories(G, node_map, num_traj=5, walk_length=8)

for i, traj in enumerate(trajectories):
    print(f"Trajectory {i}: {traj}")