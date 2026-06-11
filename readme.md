# Graph Neural Networks vs Transformer Models for Road Network Path Prediction

## 🚀 Project Overview

This project compares **Graph Neural Networks (GNNs)** and **Transformer models** for the task of **road network path prediction**. Given a partial trajectory on a road graph, the goal is to predict the next road segment (node).

The system is designed to evaluate how **spatial learning (GNNs)** compares against **temporal sequence learning (Transformers)** in modeling movement patterns on real-world road networks.

---

## 🎯 Problem Statement

Given:

* A road network represented as a graph
* A trajectory of visited nodes (road intersections)

Predict:

* The next most likely node in the trajectory

This is relevant for:

* Navigation systems
* Autonomous driving
* Route optimization
* Mobility prediction

---

## 🧠 Key Idea

We compare two modeling paradigms:

### 🔷 Graph Neural Networks (GNN)

* Operates directly on the road graph structure
* Learns spatial relationships between connected nodes
* Uses GraphSAGE-based message passing

### 🔶 Transformer Model

* Operates on node sequences (trajectories)
* Uses self-attention to model temporal dependencies
* Learns long-range movement patterns

---

## 🏗️ System Architecture

```
OSM Road Network
        ↓
Graph Construction (NetworkX)
        ↓
Trajectory Generation (Random Walks)
        ↓
Node Mapping (OSM IDs → contiguous indices)
        ↓
Training Data Preparation
        ↓
 ┌───────────────────────┐
 │   GNN Model           │
 │   Transformer Model   │
 └───────────────────────┘
        ↓
Evaluation & Comparison
```

---

## 📁 Project Structure

```
bmw-path-prediction/
│
├── data/
│   └── (generated trajectories / graphs)
│
├── notebooks/
│   ├── 03_generate_trajectories.py
│
├── src/
│   ├── data_loader.py
│   ├── graph_model.py
│   ├── transformer_model.py
│   ├── train_gnn.py
│   ├── train_transformer.py
│
├── results/
│   └── (plots, metrics, logs)
│
├── README.md
```

---

## ⚙️ Methodology

### 1. Graph Construction

* Road network is extracted using OSM data
* Represented using NetworkX graph
* Nodes represent intersections, edges represent roads

### 2. Trajectory Generation

* Random walks are performed on the graph
* Improved walk avoids immediate backtracking
* Produces realistic movement sequences

### 3. Node Mapping

* OSM node IDs are mapped to contiguous indices (0 → N)
* Required for embedding-based models

### 4. Models

#### 🔷 GNN (GraphSAGE)

* Input: Graph structure + node features
* Learns spatial dependencies
* Predicts next node using neighborhood aggregation

#### 🔶 Transformer

* Input: sequence of node indices
* Uses embedding + self-attention
* Predicts next node in sequence

---

## 📊 Evaluation Metrics

The following metrics are used:

* **Cross-Entropy Loss**
* **Top-1 Accuracy**
* **Confusion Matrix (partial analysis)**

---

## 📈 Key Findings (Preliminary)

* Transformer shows faster convergence on sequential data
* GNN captures structural relationships but lacks temporal memory
* Data quality (trajectory realism) significantly impacts performance

---

## ⚠️ Limitations

* Trajectories are generated using synthetic random walks
* No real GPS trajectory data used
* Limited evaluation on real-world driving behavior
* Confusion matrix size is reduced due to large output space

---

## 🔮 Future Work

* Integrate Node2Vec biased random walks
* Use real GPS trajectory datasets
* Add temporal GNN models (e.g., T-GCN)
* Improve transformer with positional encoding enhancements
* Extend to multi-step path prediction

---

## 🛠️ Tech Stack

* Python
* PyTorch
* PyTorch Geometric
* NetworkX
* OSMnx
* NumPy

---

## 🧪 How to Run

### 1. Install dependencies

```bash
pip install torch torchvision torch-geometric networkx osmnx numpy
```

### 2. Generate trajectories

```bash
python notebooks/03_generate_trajectories.py
```

### 3. Train GNN

```bash
python src/train_gnn.py
```

### 4. Train Transformer

```bash
python src/train_transformer.py
```

---

## 📌 Author

Hamza Majid
Focus: AI, Machine Learning, Graph Learning

---

## 🧠 Conclusion

This project demonstrates a comparative study between spatial and sequential deep learning models for trajectory prediction. It highlights the strengths of Transformer-based architectures in sequence modeling and provides a foundation for further research in graph-based mobility prediction.
