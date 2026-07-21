import json
import torch
from torch_geometric.data import Data

def load_and_prepare_graph(file_path):
    with open(file_path, 'r') as f:
        graph_data = json.load(f)

    pos_sources = []
    pos_targets = []
    neg_sources = []
    neg_targets = []

    # Separate edges based on their political weight (alliance vs. conflict)
    for edge in graph_data['edges']:
        if edge["weight"] == 1:
            pos_sources.append(edge["source"])
            pos_targets.append(edge["target"])
        elif edge["weight"] == -1:
            neg_sources.append(edge["source"])
            neg_targets.append(edge["target"])

    pos_edge_index = torch.tensor([pos_sources, pos_targets], dtype=torch.long) # `torch.long` indicates that these are long integers (64-bit).
    neg_edge_index = torch.tensor([neg_sources, neg_targets], dtype=torch.long) 

    # Determine the total number of unique nodes 
    all_nodes = pos_sources + pos_targets + neg_sources + neg_targets
    num_nodes = max(all_nodes) + 1

    # Create the initial node features tensor 
    x = torch.ones((num_nodes, 1), dtype=torch.float) # each node (a politician) are set to 1 (.ones) so they ignore there caracteristic and only use links between them

    # Assemble the final graph object
    graph = Data(
        x=x,
        pos_edge_index=pos_edge_index,
        neg_edge_index=neg_edge_index
    )

    return graph