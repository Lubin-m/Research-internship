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
        
        loss = pos_loss + (neg_weight * neg_loss) # As their is 80% of positives edges and only 20% negatives, I put more weight to the negatives ones because they are more importante usually to define political structure
        
        loss.backward()
        optimizer.step()
        
        if scheduler is not None:
            scheduler.step(loss.item())

        if epoch % 30 == 0:
            print(f"Epoch {epoch} | Loss: {loss.item():.4f}")

def train_model_1D(model, x, pos_edge_index, neg_edge_index, neg_weight, optimizer, scheduler=None, epochs=100):
    model.train()
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        z = model(x, pos_edge_index, neg_edge_index)
        z = torch.tanh(z)

        # Loss function
        pi, pj = pos_edge_index
        ni, nj = neg_edge_index
        
        # 1. Le nouveau "Décodeur par Distance" 
        # Au lieu de multiplier z[i] * z[j], on calcule leur distance au carré.
        # On met un signe "-" car le réseau veut maximiser les amis (distance proche de 0)
        pos_score = -((z[pi] - z[pj]) ** 2).squeeze()
        
        # On fait pareil pour les ennemis (le réseau voudra que ce score soit très négatif)
        neg_score = -((z[ni] - z[nj]) ** 2).squeeze()
        
        # 2. Ta fonction de Loss classique (log-sigmoid)
        pos_loss = -torch.log(torch.sigmoid(pos_score) + 1e-8).mean()
        neg_loss = -torch.log(1 - torch.sigmoid(neg_score) + 1e-8).mean()
        
        loss = pos_loss + (neg_weight * neg_loss) # As their is 80% of positives edges and only 20% negatives, I put more weight to the negatives ones because they are more importante usually to define political structure
        
        loss.backward()
        optimizer.step()
        
        if scheduler is not None:
            scheduler.step(loss.item())

        if epoch % 30 == 0:
            print(f"Epoch {epoch} | Loss: {loss.item():.4f}")