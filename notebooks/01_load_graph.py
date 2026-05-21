import osmnx as ox
import networkx as nx
import matplotlib.pyplot as plt

# Step 1: Choose a location
place_name = "Karachi, Pakistan"

# Step 2: Download road network (drive = cars)
G = ox.graph_from_place(place_name, network_type='drive')

# Step 3: Print basic info
print("Nodes:", G.number_of_nodes())
print("Edges:", G.number_of_edges())

# Step 4: Plot the graph
fig, ax = ox.plot_graph(G)