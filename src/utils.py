import numpy as np
import torch
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder

def load_data(edges_path, node_features_path, edge_type_features_path):
    """
    Load and preprocess the temporal graph data.
    
    Args:
        edges_path: Path to the edges CSV file
        node_features_path: Path to the node features CSV file
        edge_type_features_path: Path to the edge type features CSV file
        
    Returns:
        processed data dictionary
    """
    # Load edge data
    print("Loading edge data...")
    edges_df = pd.read_csv(edges_path, header=None)
    
    # Rename columns for clarity - Dataset A has 4 columns
    edges_df.columns = ['source', 'target', 'edge_type', 'timestamp']
    
    # Load node features
    print("Loading node features...")
    node_features_df = pd.read_csv(node_features_path, header=None)
    
    # Replace -1 values with NaN (missing values)
    node_features_df.replace(-1, np.nan, inplace=True)
    
    # Load edge type features
    print("Loading edge type features...")
    edge_type_features_df = pd.read_csv(edge_type_features_path, header=None)
    
    # Get unique node IDs and map to consecutive integers
    unique_nodes = pd.concat([edges_df['source'], edges_df['target']]).unique()
    node_mapping = {node: idx for idx, node in enumerate(unique_nodes)}
    inverse_node_mapping = {idx: node for node, idx in node_mapping.items()}
    
    # Map edge sources and targets to consecutive integers
    edges_df['source_mapped'] = edges_df['source'].map(node_mapping)
    edges_df['target_mapped'] = edges_df['target'].map(node_mapping)
    
    # Create node features matrix
    # Initialize with zeros and fill with available features
    node_features = np.zeros((len(unique_nodes), node_features_df.shape[1]))
    for node_id in unique_nodes:
        if node_id in node_features_df.index:
            node_features[node_mapping[node_id]] = node_features_df.loc[node_id].fillna(0).values
    
    # Create edge type features dictionary
    edge_type_features_dict = {}
    for _, row in edge_type_features_df.iterrows():
        edge_type_features_dict[row[0]] = row[1:].values
    
    return {
        'edges_df': edges_df,
        'node_features': node_features,
        'edge_type_features_dict': edge_type_features_dict,
        'node_mapping': node_mapping,
        'inverse_node_mapping': inverse_node_mapping
    }

def create_time_windows(edges_df, window_size=86400, stride=86400):
    """
    Create time windows for temporal graph processing.
    
    Args:
        edges_df: DataFrame containing edge information
        window_size: Size of time window in seconds (default: 1 day)
        stride: Stride between consecutive windows in seconds (default: 1 day)
        
    Returns:
        list of DataFrames, each representing edges in a time window
    """
    min_time = edges_df['timestamp'].min()
    max_time = edges_df['timestamp'].max()
    
    time_windows = []
    for start_time in range(min_time, max_time, stride):
        end_time = start_time + window_size
        window_edges = edges_df[(edges_df['timestamp'] >= start_time) & (edges_df['timestamp'] < end_time)]
        if len(window_edges) > 0:
            time_windows.append(window_edges)
    
    return time_windows

def evaluate_model(model, test_edges, test_non_edges=None, device='cpu'):
    """
    Evaluate the model on test data.
    
    Args:
        model: Trained GNN model
        test_edges: Positive test edges
        test_non_edges: Negative test edges (if None, will be generated)
        device: Device to run the model on
        
    Returns:
        AUC score
    """
    model.eval()
    
    with torch.no_grad():
        # Prepare positive edges
        pos_edge_indices = torch.tensor(test_edges[['source_mapped', 'target_mapped']].values.T, 
                                       dtype=torch.long, device=device)
        pos_probs = model(pos_edge_indices).cpu().numpy()
        
        # Prepare negative edges if provided
        if test_non_edges is not None:
            neg_edge_indices = torch.tensor(test_non_edges[['source_mapped', 'target_mapped']].values.T, 
                                           dtype=torch.long, device=device)
            neg_probs = model(neg_edge_indices).cpu().numpy()
        
        # Combine predictions and true labels
        y_true = np.concatenate([np.ones(len(pos_probs)), np.zeros(len(neg_probs))])
        y_scores = np.concatenate([pos_probs, neg_probs])
        
        # Calculate AUC score
        auc = roc_auc_score(y_true, y_scores)
        
    return auc

def prepare_test_data(input_file, node_mapping):
    """
    Prepare test data for prediction.
    
    Args:
        input_file: Path to the input test file
        node_mapping: Dictionary mapping original node IDs to consecutive integers
        
    Returns:
        processed test data
    """
    test_df = pd.read_csv(input_file)
    
    # Map node IDs to consecutive integers
    test_df['source_mapped'] = test_df['source'].map(node_mapping)
    test_df['target_mapped'] = test_df['target'].map(node_mapping)
    
    return test_df

def write_predictions(output_file, predictions):
    """
    Write model predictions to a CSV file.
    
    Args:
        output_file: Path to the output file
        predictions: Model predictions
    """
    pd.DataFrame({'prediction': predictions}).to_csv(output_file, index=False) 