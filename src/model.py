import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, TransformerConv
from torch_geometric.nn import GCN2Conv, GINConv, EdgeConv, GraphConv
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree

class TemporalGNN(torch.nn.Module):
    """
    Base class for temporal graph neural networks.
    """
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout=0.5):
        super(TemporalGNN, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.dropout = dropout
        
    def reset_parameters(self):
        for layer in self.convs:
            layer.reset_parameters()
        for layer in self.lins:
            layer.reset_parameters()
    
    def encode(self, x, edge_index, edge_attr=None):
        # To be implemented by subclasses
        raise NotImplementedError
    
    def decode(self, z, edge_index):
        row, col = edge_index
        return (z[row] * z[col]).sum(dim=-1)
    
    def forward(self, x, edge_index, edge_attr=None):
        z = self.encode(x, edge_index, edge_attr)
        return z


class TemporalGCN(TemporalGNN):
    """
    Temporal Graph Convolutional Network for link prediction.
    """
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout=0.5, edge_dim=None):
        super(TemporalGCN, self).__init__(in_channels, hidden_channels, out_channels, num_layers, dropout)
        
        self.convs = torch.nn.ModuleList()
        self.lins = torch.nn.ModuleList()
        
        self.convs.append(GCNConv(in_channels, hidden_channels))
        
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        
        self.convs.append(GCNConv(hidden_channels, out_channels))
        
        # Linear layers for edge features if provided
        if edge_dim is not None:
            self.edge_encoder = nn.Linear(edge_dim, hidden_channels)
        else:
            self.edge_encoder = None
            
        # Linear layers for skip connections
        self.lins.append(nn.Linear(in_channels, hidden_channels))
        self.lins.append(nn.Linear(hidden_channels, out_channels))
        
        self.reset_parameters()
    
    def reset_parameters(self):
        super(TemporalGCN, self).reset_parameters()
        if self.edge_encoder is not None:
            self.edge_encoder.reset_parameters()
    
    def encode(self, x, edge_index, edge_attr=None):
        if self.edge_encoder is not None and edge_attr is not None:
            edge_embedding = self.edge_encoder(edge_attr)
        else:
            edge_embedding = None
        
        x_skip = self.lins[0](x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.convs[0](x, edge_index))
        x = x + x_skip
        
        for i in range(1, self.num_layers - 1):
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = F.relu(self.convs[i](x, edge_index))
        
        x_skip = self.lins[1](x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        x = x + x_skip
        
        return x
    
    def forward(self, edge_index):
        return self.decode(self.node_embeddings, edge_index)
    
    def predict_link(self, edge_index):
        return torch.sigmoid(self.forward(edge_index))


class EdgeTemporalGNN(torch.nn.Module):
    """
    Edge-focused Temporal Graph Neural Network for link prediction.
    """
    def __init__(self, node_features_dim, edge_features_dim, hidden_dim, output_dim, num_layers=2, dropout=0.5):
        super(EdgeTemporalGNN, self).__init__()
        
        self.node_features_dim = node_features_dim
        self.edge_features_dim = edge_features_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Node embeddings
        self.node_encoder = nn.Sequential(
            nn.Linear(node_features_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Edge embeddings
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_features_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # GNN layers
        self.convs = torch.nn.ModuleList()
        self.convs.append(TransformerConv(hidden_dim, hidden_dim, heads=4, dropout=dropout, edge_dim=hidden_dim))
        
        for _ in range(num_layers - 2):
            self.convs.append(TransformerConv(hidden_dim * 4, hidden_dim, heads=4, dropout=dropout, edge_dim=hidden_dim))
        
        if num_layers > 1:
            self.convs.append(TransformerConv(hidden_dim * 4, output_dim, heads=1, dropout=dropout, edge_dim=hidden_dim))
        else:
            self.convs.append(TransformerConv(hidden_dim, output_dim, heads=1, dropout=dropout, edge_dim=hidden_dim))
        
        # Link predictor
        self.link_predictor = LinkPredictor(output_dim, hidden_dim, 1, num_layers=2, dropout=dropout)
    
    def forward(self, x, edge_index, edge_attr):
        # Encode node features
        x = self.node_encoder(x)
        
        # Encode edge features
        edge_emb = self.edge_encoder(edge_attr)
        
        # Apply GNN layers
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index, edge_attr=edge_emb)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.convs[-1](x, edge_index, edge_attr=edge_emb)
        
        return x
    
    def predict_link(self, x, edge_index, edge_attr, target_edge_index):
        # Get node embeddings
        node_emb = self.forward(x, edge_index, edge_attr)
        
        # Predict links for target edges
        return self.link_predictor(node_emb, target_edge_index)


class LinkPredictor(torch.nn.Module):
    """
    Link predictor module that takes node embeddings and outputs link probability.
    """
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout=0.5):
        super(LinkPredictor, self).__init__()
        
        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels * 2, hidden_channels))
        
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
        
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))
        
        self.dropout = dropout
    
    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
    
    def forward(self, x, edge_index):
        x_i = x[edge_index[0]]
        x_j = x[edge_index[1]]
        x = torch.cat([x_i, x_j], dim=-1)
        
        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.lins[-1](x)
        return torch.sigmoid(x)


class TGNNModel(torch.nn.Module):
    """
    Temporal Graph Neural Network with time encoding.
    """
    def __init__(self, node_features_dim, edge_features_dim, hidden_dim, output_dim, 
                 time_enc_dim=12, num_layers=2, dropout=0.5):
        super(TGNNModel, self).__init__()
        
        self.node_features_dim = node_features_dim
        self.edge_features_dim = edge_features_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.time_enc_dim = time_enc_dim
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Time encoding layer
        self.time_encoder = TimeEncoder(time_enc_dim)
        
        # Node embedding layer
        self.node_embedding = nn.Linear(node_features_dim, hidden_dim)
        
        # Edge embedding layer (includes edge type and time features)
        self.edge_embedding = nn.Linear(edge_features_dim + time_enc_dim, hidden_dim)
        
        # GNN layers
        self.convs = torch.nn.ModuleList()
        self.convs.append(TransformerConv(hidden_dim, hidden_dim, heads=4, dropout=dropout, edge_dim=hidden_dim))
        
        for _ in range(num_layers - 2):
            self.convs.append(TransformerConv(hidden_dim * 4, hidden_dim, heads=4, dropout=dropout, edge_dim=hidden_dim))
        
        if num_layers > 1:
            self.convs.append(TransformerConv(hidden_dim * 4, output_dim, heads=1, dropout=dropout, edge_dim=hidden_dim))
        else:
            self.convs.append(TransformerConv(hidden_dim, output_dim, heads=1, dropout=dropout, edge_dim=hidden_dim))
        
        # Link predictor
        self.link_predictor = LinkPredictor(output_dim, hidden_dim, 1, num_layers=2, dropout=dropout)
    
    def forward(self, x, edge_index, edge_attr):
        # Extract edge features and timestamps
        if edge_attr.shape[1] > 1:
            edge_features = edge_attr[:, :-1]  # All but the last column
            timestamps = edge_attr[:, -1]      # Last column contains timestamps
        else:
            # If there's only one feature, it's the edge type
            edge_features = edge_attr
            timestamps = torch.ones_like(edge_attr[:, 0])  # Use default timestamp
            
        # Encode timestamps
        time_embeddings = self.time_encoder(timestamps)
        
        # Encode node features
        x = self.node_embedding(x)
        
        # Ensure proper dimensions for concatenation
        if len(edge_features.shape) == 1:
            edge_features = edge_features.unsqueeze(1)
            
        # Ensure all tensors have proper shapes for concatenation
        if time_embeddings.shape[0] != edge_features.shape[0]:
            # Handle mismatch
            if time_embeddings.shape[0] > edge_features.shape[0]:
                time_embeddings = time_embeddings[:edge_features.shape[0]]
            else:
                # Pad edge_features
                padding = torch.zeros(time_embeddings.shape[0] - edge_features.shape[0], 
                                     edge_features.shape[1], 
                                     device=edge_features.device)
                edge_features = torch.cat([edge_features, padding], dim=0)
                
        # Encode edge features with time
        edge_emb = self.edge_embedding(torch.cat([edge_features, time_embeddings], dim=1))
        
        # Apply GNN layers
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index, edge_attr=edge_emb)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.convs[-1](x, edge_index, edge_attr=edge_emb)
        
        return x
    
    def decode(self, z, edge_index):
        """
        Decode node embeddings to edge probabilities.
        
        Args:
            z: Node embeddings
            edge_index: Edge indices
            
        Returns:
            Edge scores
        """
        row, col = edge_index
        return (z[row] * z[col]).sum(dim=-1)
    
    def predict_link(self, x, edge_index, edge_attr, target_edge_index):
        # Get node embeddings
        node_emb = self.forward(x, edge_index, edge_attr)
        
        # Predict links for target edges
        return self.link_predictor(node_emb, target_edge_index)


class TimeEncoder(torch.nn.Module):
    """
    Time encoding module that converts timestamps to time embeddings.
    """
    def __init__(self, dimension):
        super(TimeEncoder, self).__init__()
        
        self.dimension = dimension
        self.w = torch.nn.Parameter(torch.Tensor(1, dimension))
        self.b = torch.nn.Parameter(torch.Tensor(1, dimension))
        
        self.reset_parameters()
    
    def reset_parameters(self):
        # Use small values for initialization to prevent overflow
        torch.nn.init.normal_(self.w, mean=0.0, std=0.1)
        torch.nn.init.constant_(self.b, 0)
    
    def forward(self, t):
        # Normalize timestamps to prevent numerical issues
        t_min = t.min()
        t_max = t.max()
        if t_max > t_min:
            t_normalized = (t - t_min) / (t_max - t_min)
        else:
            t_normalized = torch.zeros_like(t)
            
        # Add small epsilon to avoid exact zeros
        t_normalized = t_normalized + 1e-6
        
        # Convert to proper shape
        t_normalized = t_normalized.unsqueeze(1).float()
        
        # Apply periodic encoding with clipping to ensure stability
        output = torch.cos(self.w.clamp(min=-5.0, max=5.0) * t_normalized + self.b)
        
        return output 