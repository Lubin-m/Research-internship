import torch
import torch.nn.functional as F

def train_model(model, x, pos_edge_index, neg_edge_index, neg_weight, optimizer, scheduler=None, epochs=100):
    model.train()
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        z = model(x, pos_edge_index, neg_edge_index)
        z = F.normalize(z, p=2, dim=1) # Avoid model collapse

        # Loss function
        pos_pred = (z[pos_edge_index[0]] * z[pos_edge_index[1]]).sum(dim=1)
        neg_pred = (z[neg_edge_index[0]] * z[neg_edge_index[1]]).sum(dim=1)
        
        pos_loss = -F.logsigmoid(pos_pred).mean()
        neg_loss = -F.logsigmoid(-neg_pred).mean()
        
        loss = pos_loss + (neg_weight * neg_loss) # As their is 80% positives edges and only 20% negatives, I put more weight to the negatives ones because they are more importante usually to define political structure
        
        loss.backward()
        optimizer.step()
        
        if scheduler is not None:
            scheduler.step(loss.item())

        if epoch % 30 == 0:
            print(f"Epoch {epoch} | Loss: {loss.item():.4f}")