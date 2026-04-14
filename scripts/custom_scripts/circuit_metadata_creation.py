
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


################## Fan in and fan out count ############################


def compute_fanin_fanout_cones(fanin0, fanin1, fanin2, node_type):
    """
    Computes fan-in and fan-out node counts for each node in a DAG.

    Fan-in count: number of nodes in the transitive fan-in cone (upstream)
    Fan-out count: number of nodes in the transitive fan-out cone (downstream)

    Parameters:
        fanin0, fanin1, fanin2: np.ndarray of shape (N,)
            Node indices of fan-ins, -1 for PI/const nodes
        node_type: np.ndarray of shape (N,)
            0=const, 1=PI, 2=gate (used to skip PIs if needed)

    Returns:
        fanin_count: np.ndarray of shape (N,) - # upstream nodes
        fanout_count: np.ndarray of shape (N,) - # downstream nodes
    """

    N = len(node_type)

    # --- FAN-IN computation (upstream nodes) ---
    fanin_count = np.zeros(N, dtype=int)

    # store sets of upstream nodes for each node
    ancestors = [set() for _ in range(N)]

    # assume topological order
    for i in range(N):
        f0, f1, f2 = fanin0[i], fanin1[i], fanin2[i]
        current_ancestors = set()
        for f in (f0, f1, f2):
            if f >= 0:
                current_ancestors |= ancestors[f]
                current_ancestors.add(f)
        ancestors[i] = current_ancestors
        fanin_count[i] = len(current_ancestors)

    # --- FAN-OUT computation (downstream nodes) ---
    # reverse edges: for each node, store its fanouts
    fanouts = [[] for _ in range(N)]
    for i in range(N):
        for f in (fanin0[i], fanin1[i], fanin2[i]):
            if f >= 0:
                fanouts[f].append(i)

    fanout_count = np.zeros(N, dtype=int)
    descendants = [set() for _ in range(N)]

    # process nodes in reverse topological order
    for i in reversed(range(N)):
        current_descendants = set()
        for fo in fanouts[i]:
            current_descendants |= descendants[fo]
            current_descendants.add(fo)
        descendants[i] = current_descendants
        fanout_count[i] = len(current_descendants)

    return fanin_count, fanout_count


########################## Circuit depth #######################

import numpy as np
import time

s = time.time()


# Assume df is your DataFrame
num_nodes = len(df)

# Prepare fanin arrays: shape (num_nodes, 3)
fanin = df[['fanin0', 'fanin1', 'fanin2']].to_numpy()

# Initialize depths
depth = np.zeros(num_nodes, dtype=int)

# Mask of nodes whose depth is already computed
computed = np.zeros(num_nodes, dtype=bool)

# Consts and inputs have depth 0
computed[df['node_type'].isin([0,1]).to_numpy()] = True

# Iteratively compute depths
while not np.all(computed):
    # Nodes not yet computed
    mask = ~computed
    for i in np.where(mask)[0]:
        # Fanin indices for this node
        f0, f1, f2 = fanin[i]
        fanins = [f for f in [f0,f1,f2] if f >= 0]  # ignore -1

        # If all fanins are computed, we can compute this node
        if all(computed[f] for f in fanins):
            depth[i] = 1 + max([depth[f] for f in fanins], default=0)
            computed[i] = True

max_depth = depth.max()
print("Max depth:", max_depth)

print(time.time()-s)


import numpy as np


mig_1_path = (
    '/home/ramaudruz/data_dir/4bi_8bo_rnd_in_fix_out/output/multiplier_4bi_8bo_permuti_flowy/gnn_data_collection/'
    'synth_out/res_00000000002928/run_20260304_170241_699X/mig_circuits_1000.parquet'
)

df = pd.read_parquet(mig_1_path)

num_nodes = len(df)
fanin = df[['fanin0', 'fanin1', 'fanin2']].to_numpy()

# Initialize depth array
depth = np.zeros(num_nodes, dtype=int)

# Compute in-degree (number of valid fan-ins)
in_deg = np.sum(fanin >= 0, axis=1)

# Queue of nodes with in-degree 0 (inputs and consts)
queue = list(np.where(in_deg == 0)[0])

# Build fanout lists to propagate depth
fanout_list = [[] for _ in range(num_nodes)]
for i in range(num_nodes):
    for f in fanin[i]:
        if f >= 0:
            fanout_list[f].append(i)

# Process nodes in topological order
while queue:
    node = queue.pop(0)
    for child in fanout_list[node]:
        # Update child depth
        depth[child] = max(depth[child], depth[node] + 1)
        # Decrement in-degree
        in_deg[child] -= 1
        # If all fanins processed, add to queue
        if in_deg[child] == 0:
            queue.append(child)

max_depth = depth.max()
print("Max depth:", max_depth)

#################   Node depth ###################


import numpy as np

def compute_node_depth_ranges(df):
    num_nodes = len(df)
    fanin = df[['fanin0','fanin1','fanin2']].to_numpy()

    # --- Forward pass: from inputs to gates ---
    in_deg = np.sum(fanin >= 0, axis=1)
    queue = list(np.where(in_deg == 0)[0])

    # Build fanout lists
    fanout_list = [[] for _ in range(num_nodes)]
    for i in range(num_nodes):
        for f in fanin[i]:
            if f >= 0:
                fanout_list[f].append(i)

    # Initialize min/max depth from inputs
    min_depth = np.zeros(num_nodes, dtype=int)
    max_depth = np.zeros(num_nodes, dtype=int)

    while queue:
        node = queue.pop(0)
        for child in fanout_list[node]:
            # Min depth: take min over all fan-ins +1
            if min_depth[child] == 0:
                min_depth[child] = min_depth[node] + 1
            else:
                min_depth[child] = min(min_depth[child], min_depth[node]+1)
            # Max depth: take max over all fan-ins +1
            max_depth[child] = max(max_depth[child], max_depth[node]+1)
            in_deg[child] -= 1
            if in_deg[child] == 0:
                queue.append(child)

    # --- Backward pass: from outputs to inputs ---
    po_nodes = np.where(df['is_output'])[0]

    # Build reverse fanout (fanin -> node)
    reverse_fanout = [[] for _ in range(num_nodes)]
    for i in range(num_nodes):
        for f in fanin[i]:
            if f >= 0:
                reverse_fanout[i].append(f)

    # Initialize min/max depth to outputs
    min_to_po = np.full(num_nodes, np.inf)
    max_to_po = np.zeros(num_nodes, dtype=int)

    queue = list(po_nodes)
    for node in po_nodes:
        min_to_po[node] = 0
        max_to_po[node] = 0

    while queue:
        node = queue.pop(0)
        for parent in reverse_fanout[node]:
            # Min depth to PO
            min_to_po[parent] = min(min_to_po[parent], min_to_po[node]+1)
            # Max depth to PO
            max_to_po[parent] = max(max_to_po[parent], max_to_po[node]+1)
            queue.append(parent)

    # Convert min_to_po to int
    min_to_po = min_to_po.astype(int)

    return min_depth, max_depth, min_to_po, max_to_po

# Usage
min_d, max_d, min_out, max_out = compute_node_depth_ranges(df)
print(min_d[:10], max_d[:10], min_out[:10], max_out[:10])


#### Fan out count ####

import numpy as np

fanin = df[['fanin0','fanin1','fanin2']].to_numpy()
num_nodes = len(df)

fanout = np.zeros(num_nodes, dtype=np.int32)

for col in range(3):
    f = fanin[:, col]
    mask = f >= 0
    np.add.at(fanout, f[mask], 1)


### Adjacency matrix

fanin = df[['fanin0','fanin1','fanin2']].to_numpy()
phase = df[['phase0','phase1','phase2']].to_numpy()

rows = np.repeat(np.arange(num_nodes), 3)
cols = fanin.flatten()
phases = phase.flatten()

valid = cols >= 0

rows = rows[valid]
cols = cols[valid]
phases = phases[valid]

A_reg = np.zeros((num_nodes, num_nodes), dtype=np.uint8)
A_inv = np.zeros((num_nodes, num_nodes), dtype=np.uint8)

A_reg[cols[phases==0], rows[phases==0]] = 1
A_inv[cols[phases==1], rows[phases==1]] = 1


################ Reconvergence ####################

import numpy as np

def reconvergence_metrics(fanin0, fanin1, fanin2, node_type):
    """
    Computes max and mean reconvergence per node.

    Returns:
        reconv_max : (N,)
        reconv_mean : (N,)
    """

    N = len(node_type)

    # each node stores its ancestor set
    ancestors = [set() for _ in range(N)]

    reconv_max = np.zeros(N, dtype=float)
    reconv_mean = np.zeros(N, dtype=float)

    for i in range(N):

        if node_type[i] != 2:  # not a gate
            continue

        a = fanin0[i]
        b = fanin1[i]
        c = fanin2[i]

        sets = [
            ancestors[a] | {a},
            ancestors[b] | {b},
            ancestors[c] | {c}
        ]

        overlaps = []

        pairs = [(0,1), (0,2), (1,2)]

        for x,y in pairs:
            inter = sets[x] & sets[y]
            union = sets[x] | sets[y]

            if len(union) == 0:
                overlaps.append(0)
            else:
                overlaps.append(len(inter) / len(union))

        reconv_max[i] = max(overlaps)
        reconv_mean[i] = sum(overlaps) / 3

        ancestors[i] = sets[0] | sets[1] | sets[2]

    return reconv_max, reconv_mean

t = reconvergence_metrics(df['fanin0'].to_numpy(), df['fanin1'].to_numpy(), df['fanin2'].to_numpy(), df['node_type'].to_numpy())

visualize_mig_df(df)


