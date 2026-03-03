import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def visualize_mig_df(df):
    """
    Visualize MIG stored in dataframe format.
    Required columns:
        node_type
        fanin0, fanin1, fanin2
        phase0, phase1, phase2
        is_output
    """
    TYPE_LABEL = {
        0: "C",  # CONST
        1: "I",  # PI
        2: "M",  # MAJ
    }
    TYPE_COLOR = {
        0: "#d3d3d3",   # light gray (CONST)
        1: "#87ceeb",   # light blue (PI)
        2: "#98fb98",   # light green (MAJ)
        3: "#f08080",   # light coral (OUTPUT)
    }
    G = nx.DiGraph()
    # --------------------
    # Add structural nodes
    # --------------------
    for node_id, row in df.iterrows():
        node_type = row["node_type"]
        label = TYPE_LABEL[node_type]
        G.add_node(node_id, label=label, node_type=node_type)
    # --------------------
    # Add structural edges
    # --------------------
    for node_id, row in df.iterrows():
        for i in range(3):
            fanin = row[f"fanin{i}"]
            if fanin >= 0:
                phase = row[f"phase{i}"]
                G.add_edge(
                    fanin,
                    node_id,
                    inverted=bool(phase)
                )
    # --------------------
    # Add artificial outputs
    # --------------------
    output_offset = len(df)
    for node_id, row in df.iterrows():
        if row["is_output"] == 1:
            out_id = output_offset
            output_offset += 1
            G.add_node(out_id, label="O", node_type=3)
            G.add_edge(node_id, out_id, inverted=False)
    # --------------------
    # Layout (Graphviz DOT)
    # --------------------
    A = nx.nx_agraph.to_agraph(G)
    A.graph_attr.update(ranksep="1.8", nodesep="0.9")
    A.layout("dot")
    pos = nx.nx_agraph.graphviz_layout(G, prog="dot")
    # --------------------
    # Drawing
    # --------------------
    plt.figure(figsize=(24, 18))
    node_labels = nx.get_node_attributes(G, "label")
    # Node colors
    node_colors = [
        TYPE_COLOR[G.nodes[n]["node_type"]]
        for n in G.nodes()
    ]
    nx.draw_networkx_nodes(
        G,
        pos,
        node_color=node_colors,
        edgecolors="black",
        node_size=900,
    )
    nx.draw_networkx_labels(
        G,
        pos,
        labels=node_labels,
        font_size=12,
        font_weight="bold"
    )
    # Separate edges
    normal_edges = [
        (u, v) for u, v, d in G.edges(data=True)
        if not d.get("inverted", False)
    ]
    inverted_edges = [
        (u, v) for u, v, d in G.edges(data=True)
        if d.get("inverted", False)
    ]
    # Draw edges
    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=normal_edges,
        edge_color="black",
        width=1.5
    )
    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=inverted_edges,
        style="dashed",
        edge_color="#444444",  # dark gray
        width=1.5
    )
    plt.axis("off")
    plt.title("MIG Visualization", fontsize=18)
    plt.show()


def compute_fanin_cones_optimized(df):
    """
    Returns:
        cones: list of int bitmasks
        Bit i of cones[n] = 1 if node i is in fanin cone of n
    """
    N = len(df)
    cones = [0] * N

    # Topological order: since nodes only depend on lower IDs
    # (as in your example), we iterate in order.
    # If not guaranteed, you should compute topo order first.
    for node in range(N):
        row = df.loc[node]
        mask = 0

        for i in range(3):
            fanin = row[f"fanin{i}"]
            if fanin >= 0:
                mask |= (1 << fanin)      # direct fanin
                mask |= cones[fanin]      # transitive fanin

        cones[node] = mask

    return cones


def simulate_mig_64bit_fast(df, seed=None):
    """
    Ultra-optimized 64-bit simulation.
    No pandas access inside the loop.

    Returns:
        values: np.ndarray uint64, size N
    """

    if seed is not None:
        np.random.seed(seed)

    N = len(df)

    # Extract columns once (VERY important)
    node_type = df["node_type"].to_numpy(dtype=np.int8)

    fanin0 = df["fanin0"].to_numpy(dtype=np.int32)
    fanin1 = df["fanin1"].to_numpy(dtype=np.int32)
    fanin2 = df["fanin2"].to_numpy(dtype=np.int32)

    phase0 = df["phase0"].to_numpy(dtype=np.bool_)
    phase1 = df["phase1"].to_numpy(dtype=np.bool_)
    phase2 = df["phase2"].to_numpy(dtype=np.bool_)

    values = np.zeros(N, dtype=np.uint64)

    # Pre-generate random PI values in one vectorized call
    pi_mask = (node_type == 1)
    values[pi_mask] = np.random.randint(
        0, 2**64, size=np.sum(pi_mask), dtype=np.uint64
    )

    # CONST nodes already zero (node_type == 0)

    # Process MAJ nodes in order
    for node in range(N):
        if node_type[node] != 2:
            continue

        a = values[fanin0[node]]
        b = values[fanin1[node]]
        c = values[fanin2[node]]

        if phase0[node]:
            a = ~a
        if phase1[node]:
            b = ~b
        if phase2[node]:
            c = ~c

        values[node] = (a & b) | (a & c) | (b & c)

    return values

def compute_fanin_cones_fast(df):
    """
    Bitmask-based fanin cones.
    No pandas in loop.
    """

    N = len(df)

    fanin0 = df["fanin0"].to_numpy(dtype=np.int32)
    fanin1 = df["fanin1"].to_numpy(dtype=np.int32)
    fanin2 = df["fanin2"].to_numpy(dtype=np.int32)

    cones = [0] * N

    for node in range(N):
        mask = 0

        for f in (fanin0[node], fanin1[node], fanin2[node]):
            if f >= 0:
                mask |= (1 << f)
                mask |= cones[f]

        cones[node] = mask

    return cones


import numpy as np

def simulate_mig_matrix(df, n_patterns=64, seed=None):
    """
    True parallel simulation.

    Returns:
        inputs  : (n_patterns, n_pi)
        values  : (n_nodes, n_patterns)
    """

    if seed is not None:
        np.random.seed(seed)

    N = len(df)

    # Extract arrays once
    node_type = df["node_type"].to_numpy(dtype=np.int8)

    fanin0 = df["fanin0"].to_numpy(dtype=np.int32)
    fanin1 = df["fanin1"].to_numpy(dtype=np.int32)
    fanin2 = df["fanin2"].to_numpy(dtype=np.int32)

    phase0 = df["phase0"].to_numpy(dtype=np.bool_)
    phase1 = df["phase1"].to_numpy(dtype=np.bool_)
    phase2 = df["phase2"].to_numpy(dtype=np.bool_)

    # Identify PI nodes
    pi_nodes = np.where(node_type == 1)[0]
    n_pi = len(pi_nodes)

    # -----------------------------------
    # 1️⃣ Generate random PI patterns
    # -----------------------------------
    inputs = np.random.randint(
        0, 2, size=(n_patterns, n_pi), dtype=np.uint8
    )

    # -----------------------------------
    # 2️⃣ Allocate node value matrix
    # -----------------------------------
    values = np.zeros((N, n_patterns), dtype=np.uint8)

    # Assign PI values
    for i, node in enumerate(pi_nodes):
        values[node] = inputs[:, i]

    # CONST nodes already 0

    # -----------------------------------
    # 3️⃣ Evaluate MAJ nodes
    # -----------------------------------
    for node in range(N):
        if node_type[node] != 2:
            continue

        a = values[fanin0[node]]
        b = values[fanin1[node]]
        c = values[fanin2[node]]

        if phase0[node]:
            a = 1 - a
        if phase1[node]:
            b = 1 - b
        if phase2[node]:
            c = 1 - c

        # MAJ = (a&b)|(a&c)|(b&c)
        values[node] = (a & b) | (a & c) | (b & c)

    return inputs, values


mig_1_path = (
    '/home/ramaudruz/data_dir/4bi_8bo_rnd_in_fix_out/output/multiplier_4bi_8bo_permuti_flowy/tc_sme_n_3007/'
    'synth_out/res_00000000000000/mig_cache/mig_output_round_0.parquet'
)

df = pd.read_parquet(mig_1_path)


simulate_mig_64bit_fast(df)

compute_fanin_cones_fast(df)

import time
s = time.time()
a, b = simulate_mig_matrix(df)
print(time.time() - s)

mig_2_path = (
    '/home/ramaudruz/data_dir/4bi_8bo_rnd_in_fix_out/output/multiplier_4bi_8bo_permuti_flowy/tc_sme_n_3007/'
    'synth_out/res_00000000000001/mig_cache/mig_output_round_6601.parquet'
)

df2 = pd.read_parquet(mig_2_path)

visualize_mig_df(df2)


df3 = pd.read_parquet('/home/ramaudruz/data_dir/4bi_8bo_rnd_in_fix_out/output/multiplier_4bi_8bo_permuti_flowy/tc_sme_n_3007/synth_out_cache_260227/res_00000000000000/mig_cache/mig_output_round_0.parquet')


import os
data_dir = '/home/ramaudruz/data_dir/4bi_8bo_rnd_in_fix_out/output/multiplier_4bi_8bo_permuti_flowy/tc_sme_n_3007/synth_out_cache_260227/res_00000000000000/mig_cache/'

dir_list = os.listdir(data_dir)

import pyarrow as pa
import pyarrow.parquet as pq

output_path = '/home/ramaudruz/data_dir/4bi_8bo_rnd_in_fix_out/output/multiplier_4bi_8bo_permuti_flowy/tc_sme_n_3007/synth_out_cache_260227/res_00000000000000/output_example.parquet'

for i in range(10_000):

    t = time.time()
    df = pd.read_parquet(data_dir + f'mig_output_round_{i}.parquet')
    print(time.time() - t)
    df.insert(0, 'round', i)

    if i == 0:
        initial_table = pa.Table.from_pandas(df, preserve_index=False)
        writer = pq.ParquetWriter(
            output_path,
            initial_table.schema,
            compression="zstd"
        )
    pa_table = pa.Table.from_pandas(df, preserve_index=False)
    writer.write_table(pa_table)

writer.close()

import time
t = time.time()
pf = pq.ParquetFile(output_path)
pa_table = pf.read_row_group(10)
t2 = time.time()
sample_df = pa_table.to_pandas()

print(time.time() - t)
print(t2 - t)












