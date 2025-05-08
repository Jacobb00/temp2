import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.transforms import RandomLinkSplit

def process_temporal_data(data_dict, time_windows):
    """
    Process temporal graph data for link prediction.
    
    Args:
        data_dict: Dictionary with preprocessed data
        time_windows: List of time window DataFrames
        
    Returns:
        List of PyTorch Geometric Data objects for each time window
    """
    node_features = torch.FloatTensor(data_dict['node_features'])
    num_nodes = node_features.shape[0]
    
    temporal_graphs = []
    
    for window_df in time_windows:
        if len(window_df) < 2:  # Skip windows with too few edges
            continue
            
        # Create edge index tensor
        edge_index = torch.tensor(window_df[['source_mapped', 'target_mapped']].values.T, dtype=torch.long)
        
        # Create edge attributes
        edge_attr = torch.tensor(np.column_stack([
            window_df['edge_type'].values,
            window_df['timestamp'].values
        ]), dtype=torch.float)
        
        # Create Data object
        graph = Data(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_attr,
            num_nodes=num_nodes
        )
        
        # Add training mask
        num_edges = edge_index.size(1)
        graph.train_mask = torch.ones(num_edges, dtype=torch.bool)
        
        temporal_graphs.append(graph)
    
    return temporal_graphs

def generate_negative_edges(pos_edge_index, num_nodes, num_samples):
    """
    Generate negative edges (non-existent edges) for training.
    """
    # Convert positive edges to set for efficient lookup
    pos_edges = set(map(tuple, pos_edge_index.t().tolist()))
    
    neg_edges = []
    while len(neg_edges) < num_samples:
        # Generate random node pairs
        src = np.random.randint(0, num_nodes)
        dst = np.random.randint(0, num_nodes)
        
        # Check if this edge already exists or is a self-loop
        if src != dst and (src, dst) not in pos_edges:
            neg_edges.append([src, dst])
            pos_edges.add((src, dst))  # Add to set to avoid duplicates
    
    return torch.tensor(neg_edges, dtype=torch.long).t()

def prepare_link_prediction_data(temporal_graphs, val_ratio=0.15, test_ratio=0.15):
    """
    Prepare data for temporal link prediction by splitting into train/val/test.
    
    Args:
        temporal_graphs: List of temporal graph Data objects
        val_ratio: Ratio of validation edges
        test_ratio: Ratio of test edges
        
    Returns:
        Dictionary containing train/val/test splits for each time window
    """
    splits = []
    
    for graph in temporal_graphs:
        try:
            # Ensure we have enough edges for splitting
            num_edges = graph.edge_index.size(1)
            if num_edges < 10:  # Skip if too few edges
                continue
                
            # Create train/val/test splits
            transform = RandomLinkSplit(
                num_val=val_ratio,
                num_test=test_ratio,
                is_undirected=False,
                add_negative_train_samples=True,
                neg_sampling_ratio=1.0,
                split_labels=True
            )
            
            # Apply transform safely
            try:
                train_data, val_data, test_data = transform(graph)
                splits.append((train_data, val_data, test_data))
            except Exception as e:
                print(f"Warning: Skipping graph due to transform error: {e}")
                continue
                
        except Exception as e:
            print(f"Warning: Error processing graph: {e}")
            continue
    
    if not splits:
        raise ValueError("No valid splits could be created from the temporal graphs")
    
    return splits

def process_test_data(test_df, data_dict):
    """
    Process test data for prediction.
    """
    node_features = torch.FloatTensor(data_dict['node_features'])
    num_nodes = node_features.shape[0]
    
    # Create edge index tensor
    edge_index = torch.tensor(test_df[['source_mapped', 'target_mapped']].values.T, dtype=torch.long)
    
    # Create edge attributes based on available columns
    if 'edge_type' in test_df.columns:
        if 'timestamp' in test_df.columns:
            # Both edge_type and timestamp
            edge_attr = torch.tensor(np.column_stack([
                test_df['edge_type'].values,
                test_df['timestamp'].values
            ]), dtype=torch.float)
        elif 'timestamp_start' in test_df.columns:
            # edge_type and timestamp_start
            edge_attr = torch.tensor(np.column_stack([
                test_df['edge_type'].values,
                test_df['timestamp_start'].values
            ]), dtype=torch.float)
        else:
            # Only edge_type
            edge_attr = torch.tensor(test_df['edge_type'].values.reshape(-1, 1), dtype=torch.float)
    else:
        edge_attr = None
    
    # Create Data object
    test_graph = Data(
        x=node_features,
        edge_index=edge_index,
        edge_attr=edge_attr,
        num_nodes=num_nodes
    )
    
    return test_graph 