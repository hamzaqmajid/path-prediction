"""
data_loader.py

Three improvements over previous version
─────────────────────────────────────────
1. Direction-biased random walks (Fix 1 — most important)
   Neighbours are weighted by how well they continue the current heading.
   A forward-continuing neighbour gets weight ~1.0; a sharp U-turn gets
   weight near 0.  This creates learnable directional momentum in the data.
   The Transformer can now use trajectory history to predict the heading
   and therefore the next node.

2. Richer node features (Fix 2)
   Previous: (lat, lon, degree) — 3 features, nearly identical for all nodes.
   New: (lat, lon, degree, road_type, speed_limit, in_bearing, out_bearing)
   — 7 features.  road_type and speed_limit come from the dominant edge
   attribute at each node; bearings capture the typical approach/departure
   angle.  All features are normalised to [0, 1] or [-1, 1] so no single
   feature dominates the GNN.

3. train_val_test_split unchanged — kept as-is.
"""

import math
import random
import numpy as np
import torch
import osmnx as ox


# ── Road type encoding ────────────────────────────────────────────────────────
# Higher = more major road.  Unknown/rare types default to 0.5 (mid-range).
ROAD_TYPE_SCORE = {
    'motorway':       1.0,
    'motorway_link':  0.9,
    'trunk':          0.85,
    'trunk_link':     0.8,
    'primary':        0.75,
    'primary_link':   0.7,
    'secondary':      0.6,
    'secondary_link': 0.55,
    'tertiary':       0.45,
    'tertiary_link':  0.4,
    'unclassified':   0.3,
    'residential':    0.25,
    'living_street':  0.15,
    'service':        0.1,
    'track':          0.05,
}

# Default speed limits by road type (km/h), normalised later to [0, 1]
DEFAULT_SPEED = {
    'motorway':      130,
    'motorway_link':  80,
    'trunk':         100,
    'trunk_link':     60,
    'primary':        70,
    'primary_link':   50,
    'secondary':      50,
    'secondary_link': 50,
    'tertiary':       50,
    'tertiary_link':  50,
    'unclassified':   50,
    'residential':    30,
    'living_street':  10,
    'service':        20,
    'track':          20,
}
MAX_SPEED = 130.0


def _edge_highway(data):
    """Return the highway tag as a plain string (handles list values)."""
    hw = data.get('highway', 'unclassified')
    if isinstance(hw, list):
        hw = hw[0]
    return hw


def _edge_speed(data, highway):
    """Return speed limit in km/h as a float."""
    ms = data.get('maxspeed', None)
    if ms is not None:
        if isinstance(ms, list):
            ms = ms[0]
        try:
            return float(str(ms).split()[0])
        except (ValueError, AttributeError):
            pass
    return float(DEFAULT_SPEED.get(highway, 50))


def _bearing(lat1, lon1, lat2, lon2):
    """
    Compass bearing in degrees [0, 360) from point 1 → point 2.
    Uses the spherical law of cosines approximation (accurate enough for
    city-scale distances).
    """
    d_lon = math.radians(lon2 - lon1)
    lat1r = math.radians(lat1)
    lat2r = math.radians(lat2)
    x = math.sin(d_lon) * math.cos(lat2r)
    y = (math.cos(lat1r) * math.sin(lat2r)
         - math.sin(lat1r) * math.cos(lat2r) * math.cos(d_lon))
    return (math.degrees(math.atan2(x, y)) + 360) % 360


def _angle_diff(a, b):
    """Signed angular difference a→b in [-180, 180]."""
    diff = (b - a + 180) % 360 - 180
    return diff


# ── Graph loading ─────────────────────────────────────────────────────────────

def load_graph(place_name="Cottbus, Germany"):
    G = ox.graph_from_place(place_name, network_type='drive')
    return G


# ── Feature extraction ────────────────────────────────────────────────────────

def build_graph_data(G):
    """
    Build node feature matrix and edge index from an OSMnx graph.

    Node features (7-dimensional, all normalised)
    ─────────────────────────────────────────────
    0  lat          — latitude,  normalised to [0,1] within bounding box
    1  lon          — longitude, normalised to [0,1] within bounding box
    2  degree       — node degree / max_degree
    3  road_type    — dominant road type score (0=track … 1=motorway)
    4  speed_limit  — dominant speed limit / 130 km/h
    5  in_bearing   — mean incoming bearing (sin-encoded, so periodic)
    6  out_bearing  — mean outgoing bearing (sin-encoded)

    FIX: previous version used only (lat, lon, degree) — 3 raw features
    with almost no discriminative power between nodes.  7 features with
    road semantics give the GNN real signal to distinguish intersection types.
    """
    nodes = list(G.nodes())
    node_id_map = {node: i for i, node in enumerate(nodes)}

    # Bounding box for lat/lon normalisation
    lats = [G.nodes[n]['y'] for n in nodes]
    lons = [G.nodes[n]['x'] for n in nodes]
    lat_min, lat_max = min(lats), max(lats)
    lon_min, lon_max = min(lons), max(lons)
    lat_range = (lat_max - lat_min) or 1.0
    lon_range = (lon_max - lon_min) or 1.0

    max_degree = max(dict(G.degree()).values()) or 1

    features = []
    for node in nodes:
        lat = G.nodes[node]['y']
        lon = G.nodes[node]['x']
        deg = G.degree[node]

        # Collect attributes from all edges incident to this node
        road_scores, speeds, in_bearings, out_bearings = [], [], [], []

        for u, v, data in G.edges(node, data=True):
            hw    = _edge_highway(data)
            spd   = _edge_speed(data, hw)
            score = ROAD_TYPE_SCORE.get(hw, 0.3)

            road_scores.append(score)
            speeds.append(spd)

            # Bearing from node → neighbour (outgoing) and reverse (incoming)
            lat_v = G.nodes[v]['y']
            lon_v = G.nodes[v]['x']
            bearing_out = _bearing(lat, lon, lat_v, lon_v)
            bearing_in  = (bearing_out + 180) % 360

            out_bearings.append(bearing_out)
            in_bearings.append(bearing_in)

        road_type  = np.mean(road_scores) if road_scores else 0.3
        speed      = np.mean(speeds)      if speeds      else 50.0
        # Encode bearings as mean unit vector angle → single scalar in [-1,1]
        in_bear    = math.degrees(math.atan2(
            np.mean([math.sin(math.radians(b)) for b in in_bearings])  if in_bearings  else 0,
            np.mean([math.cos(math.radians(b)) for b in in_bearings])  if in_bearings  else 1,
        )) / 180.0
        out_bear   = math.degrees(math.atan2(
            np.mean([math.sin(math.radians(b)) for b in out_bearings]) if out_bearings else 0,
            np.mean([math.cos(math.radians(b)) for b in out_bearings]) if out_bearings else 1,
        )) / 180.0

        features.append([
            (lat - lat_min) / lat_range,    # normalised lat
            (lon - lon_min) / lon_range,    # normalised lon
            deg / max_degree,               # normalised degree
            road_type,                      # road type score [0,1]
            speed / MAX_SPEED,              # normalised speed
            in_bear,                        # mean incoming bearing [-1,1]
            out_bear,                       # mean outgoing bearing [-1,1]
        ])

    x = torch.tensor(features, dtype=torch.float)

    # Edge index — bidirectional for undirected message passing
    edges = []
    for u, v in G.edges():
        edges.append([node_id_map[u], node_id_map[v]])
        edges.append([node_id_map[v], node_id_map[u]])

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    return x, edge_index, node_id_map


# ── Direction-biased random walk ──────────────────────────────────────────────

def random_walk(G, start, length=8, alpha=3.0):
    """
    Direction-biased random walk with anti-backtracking.

    FIX: previous walk chose uniformly among forward neighbours — no
    directional signal whatsoever.  This walk weights each candidate by
    how well it continues the current heading:

        weight(neighbour) = exp(alpha * cos(angle_diff))

    where angle_diff is the angle between the current heading and the
    direction to that neighbour.  alpha controls bias strength:
      alpha=0  →  uniform (same as before)
      alpha=3  →  strongly prefers straight-ahead; occasional turns
      alpha=5  →  almost always straight; rarely turns

    With alpha=3, the generated trajectories have genuine directional
    momentum so the Transformer can learn "vehicle was heading north →
    likely continues north rather than turning south."

    Parameters
    ----------
    G      : NetworkX OSM graph
    start  : OSM node ID
    length : walk length
    alpha  : direction bias strength (default 3.0)
    """
    walk    = [start]
    heading = None   # degrees [0,360), None until second step

    for _ in range(length - 1):
        current   = walk[-1]
        prev      = walk[-2] if len(walk) >= 2 else None
        neighbors = list(G.neighbors(current))

        if not neighbors:
            break

        # Anti-backtracking: drop immediate predecessor if alternatives exist
        forward = [n for n in neighbors if n != prev]
        candidates = forward if forward else neighbors

        lat_c = G.nodes[current]['y']
        lon_c = G.nodes[current]['x']

        if heading is None or len(candidates) == 1:
            # First step or forced — choose uniformly
            next_node = random.choice(candidates)
        else:
            # Weight each candidate by directional alignment
            weights = []
            for nb in candidates:
                lat_nb = G.nodes[nb]['y']
                lon_nb = G.nodes[nb]['x']
                bear   = _bearing(lat_c, lon_c, lat_nb, lon_nb)
                diff   = _angle_diff(heading, bear)          # [-180, 180]
                # cos maps: 0° diff→1.0, 90°→0.0, 180°→-1.0
                w = math.exp(alpha * math.cos(math.radians(diff)))
                weights.append(w)

            total = sum(weights)
            probs = [w / total for w in weights]
            next_node = random.choices(candidates, weights=probs, k=1)[0]

        # Update heading for next step
        lat_next = G.nodes[next_node]['y']
        lon_next = G.nodes[next_node]['x']
        heading  = _bearing(lat_c, lon_c, lat_next, lon_next)

        walk.append(next_node)

    return walk


# ── Trajectory generation ─────────────────────────────────────────────────────

def generate_trajectories(G, node_map, num_traj=5000, length=8, alpha=3.0):
    """
    Generate `num_traj` direction-biased walks of length `length`.

    alpha is passed through to random_walk.  Set alpha=0 to reproduce
    the old uniform behaviour for ablation comparison.
    """
    nodes        = list(G.nodes())
    trajectories = []

    for _ in range(num_traj):
        start = random.choice(nodes)
        traj  = random_walk(G, start, length, alpha=alpha)
        traj  = [node_map[n] for n in traj]
        trajectories.append(traj)

    return trajectories


# ── Train / val / test split ──────────────────────────────────────────────────

def train_val_test_split(trajectories, val_ratio=0.1, test_ratio=0.1, seed=42):
    """80 / 10 / 10 split with fixed seed for reproducibility."""
    rng  = random.Random(seed)
    data = trajectories[:]
    rng.shuffle(data)

    n      = len(data)
    n_test = int(n * test_ratio)
    n_val  = int(n * val_ratio)

    test  = data[:n_test]
    val   = data[n_test : n_test + n_val]
    train = data[n_test + n_val :]

    return train, val, test