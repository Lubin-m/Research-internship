import torch

def train_model(model, x, pos_edge_index, neg_edge_index, optimizer, epochs=100):
    model.train()
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        z = model(x, pos_edge_index, neg_edge_index)
        
        loss = model.loss(z, pos_edge_index, neg_edge_index)
        
        loss.backward()
        optimizer.step()

        if epoch % 30 == 0:
            print(f"Epoch {epoch} | Loss: {loss.item():.4f}")