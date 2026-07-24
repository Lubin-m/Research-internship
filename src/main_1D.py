import torch
import json
from torch_geometric.nn import SignedGCN
from preprocessing import load_and_prepare_graph
from utils import train_model_1D

graph = load_and_prepare_graph("data/processed/bundestag_signed.json")

pos_edge_index = graph.pos_edge_index
neg_edge_index = graph.neg_edge_index

num_pos = pos_edge_index.size(1)
num_neg = neg_edge_index.size(1)
neg_weight = num_pos / num_neg if num_neg > 0 else 1.0
print(f"Positive edges : {num_pos} | Negatives edges : {num_neg}")
print(f"Weight applied to neg edges : {neg_weight:.2f}")

num_nodes = int(torch.max(torch.cat([pos_edge_index, neg_edge_index], dim=1)).item()) + 1 

model = SignedGCN(in_channels=64, hidden_channels=128, num_layers=2)

x = model.create_spectral_features(pos_edge_index, neg_edge_index, num_nodes=num_nodes) # instead of giving 1 to every node, this function gives smart starting position to every node

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # defines the device used (GPU if there is one)
print(f"Appareil utilisé pour l'entraînement : {device}")

model = model.to(device)  # forces every elements to be in the same space (either everything in the GPU or in the RAM)
x = x.to(device)
pos_edge_index = pos_edge_index.to(device)
neg_edge_index = neg_edge_index.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=50)

print("Training...")
train_model_1D(model, x, pos_edge_index, neg_edge_index, neg_weight, optimizer, scheduler, epochs=500)
print("Finished !")

model.eval()
with torch.no_grad(): # gives the final result
    final_embeddings = model(x, pos_edge_index, neg_edge_index)
    embeddings_list = final_embeddings.tolist() # save the result in a simple list to use it for visualisation

json_path = "/content/Research-internship/embeddings_1D.json"

with open(json_path, "w") as f:
    json.dump(embeddings_list, f) # transform the list into a .json