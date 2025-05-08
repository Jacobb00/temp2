import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
import matplotlib.pyplot as plt

def train_model(model, train_data, val_data, optimizer, device, epochs=100, patience=10, 
               batch_size=None, verbose=True):
    """
    Train the GNN model for link prediction.
    
    Args:
        model: GNN model
        train_data: Training data (PyG Data object)
        val_data: Validation data (PyG Data object)
        optimizer: PyTorch optimizer
        device: Device to run training on
        epochs: Number of training epochs
        patience: Patience for early stopping
        batch_size: Batch size for mini-batch training (if None, use full-batch)
        verbose: Whether to print progress
        
    Returns:
        trained model and training history
    """
    model.to(device)
    
    # Move data to device
    x = train_data.x.to(device)
    edge_index = train_data.edge_index.to(device)
    
    if hasattr(train_data, 'edge_attr') and train_data.edge_attr is not None:
        edge_attr = train_data.edge_attr.to(device)
    else:
        edge_attr = None
    
    # Initialize variables for early stopping
    best_val_auc = 0
    counter = 0
    best_model_state = None
    
    # Training history
    history = {
        'train_loss': [],
        'val_auc': []
    }
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        
        # Forward pass
        optimizer.zero_grad()
        
        # Get node embeddings
        if edge_attr is not None:
            z = model(x, edge_index, edge_attr)
        else:
            z = model(x, edge_index)
        
        # Compute loss on training edges
        pos_edge_index = train_data.edge_index[:, train_data.train_mask]
        neg_edge_index = train_data.edge_index[:, ~train_data.train_mask]
        
        pos_out = model.decode(z, pos_edge_index)
        neg_out = model.decode(z, neg_edge_index)
        
        pos_loss = -torch.log(torch.sigmoid(pos_out) + 1e-15).mean()
        neg_loss = -torch.log(1 - torch.sigmoid(neg_out) + 1e-15).mean()
        loss = pos_loss + neg_loss
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Validation
        val_auc = evaluate(model, x, val_data.edge_index, edge_attr=edge_attr, device=device)
        
        # Record history
        history['train_loss'].append(loss.item())
        history['val_auc'].append(val_auc)
        
        # Print progress
        if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Val AUC: {val_auc:.4f}')
        
        # Early stopping
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            counter = 0
            best_model_state = {key: value.cpu().clone() for key, value in model.state_dict().items()}
        else:
            counter += 1
            if counter >= patience:
                if verbose:
                    print(f'Early stopping at epoch {epoch}')
                break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, history


def evaluate(model, x, edge_index, edge_attr=None, device='cpu'):
    """
    Evaluate the model on validation or test data.
    
    Args:
        model: Trained GNN model
        x: Node features
        edge_index: Edge indices for the graph
        edge_attr: Edge attributes (optional)
        device: Device to run evaluation on
        
    Returns:
        AUC score
    """
    model.eval()
    
    with torch.no_grad():
        # Forward pass
        if edge_attr is not None:
            z = model(x, edge_index, edge_attr)
        else:
            z = model(x, edge_index)
        
        # Compute predictions
        out = model.decode(z, edge_index)
        prob = torch.sigmoid(out)
        
        # Use first half of edges as positive and second half as negative
        split = edge_index.size(1) // 2
        y_true = torch.cat([torch.ones(split), torch.zeros(edge_index.size(1) - split)])
        y_pred = prob.cpu()
        
        # Calculate AUC score
        auc = roc_auc_score(y_true, y_pred)
    
    return auc


def predict(model, x, edge_index, edge_attr=None, target_edge_index=None, device='cpu'):
    """
    Make predictions on target edges.
    
    Args:
        model: Trained GNN model
        x: Node features
        edge_index: Edge indices for the graph
        edge_attr: Edge attributes (optional)
        target_edge_index: Target edge indices to predict on
        device: Device to run prediction on
        
    Returns:
        Probability predictions for target edges
    """
    model.eval()
    
    with torch.no_grad():
        # Move data to device
        x = x.to(device)
        edge_index = edge_index.to(device)
        
        if edge_attr is not None:
            edge_attr = edge_attr.to(device)
        
        if target_edge_index is not None:
            target_edge_index = target_edge_index.to(device)
        else:
            target_edge_index = edge_index
        
        # Forward pass to get node embeddings
        if edge_attr is not None:
            z = model(x, edge_index, edge_attr)
        else:
            z = model(x, edge_index)
        
        # Predict on target edges
        pred = model.decode(z, target_edge_index).sigmoid().cpu().numpy()
    
    return pred


def plot_training_history(history):
    """
    Plot training history.
    
    Args:
        history: Dictionary containing training history
    """
    plt.figure(figsize=(12, 4))
    
    # Plot training loss
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'])
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    # Plot validation AUC
    plt.subplot(1, 2, 2)
    plt.plot(history['val_auc'])
    plt.title('Validation AUC')
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()


def compute_metrics(y_true, y_scores):
    """
    Compute various evaluation metrics.
    
    Args:
        y_true: True labels
        y_scores: Predicted scores
        
    Returns:
        Dictionary of metrics
    """
    # ROC AUC
    roc_auc = roc_auc_score(y_true, y_scores)
    
    # Precision-Recall AUC
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    pr_auc = auc(recall, precision)
    
    return {
        'roc_auc': roc_auc,
        'pr_auc': pr_auc
    } 