import torch
import json
from torch_geometric.nn import SignedGCN
from preprocessing import load_and_prepare_graph
from utils import train_model

graph = load_and_prepare_graph("data/processed/bundestag_signed.json")

pos_edge_index = graph.pos_edge_index
neg_edge_index = graph.neg_edge_index

if hasattr(graph, 'num_nodes') and graph.num_nodes is not None:
    num_nodes = graph.num_nodes
else:
    num_nodes = int(torch.max(torch.cat([pos_edge_index, neg_edge_index], dim=1)).item()) + 1

model = SignedGCN(in_channels=64, hidden_channels=64, num_layers=2)

x = model.create_spectral_features(pos_edge_index, neg_edge_index, num_nodes=num_nodes)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Appareil utilisé pour l'entraînement : {device}")

model = model.to(device)
x = x.to(device)
pos_edge_index = pos_edge_index.to(device)
neg_edge_index = neg_edge_index.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=50)

print("Training...")
train_model(model, x, pos_edge_index, neg_edge_index, optimizer, scheduler, epochs=500)
print("Finished !")

model.eval()
with torch.no_grad():
    final_embeddings = model(x, pos_edge_index, neg_edge_index)
    embeddings_list = final_embeddings.tolist()

with open("embeddings.json", "w") as f:
    json.dump(embeddings_list, f)