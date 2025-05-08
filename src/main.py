import os
import torch
import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm
import time

from utils import load_data, create_time_windows, prepare_test_data, write_predictions
from data_processing import (
    process_temporal_data, 
    generate_negative_edges,
    prepare_link_prediction_data,
    process_test_data
)
from model import TemporalGCN, EdgeTemporalGNN, TGNNModel
from train import train_model, evaluate, predict, plot_training_history

def parse_args():
    parser = argparse.ArgumentParser(description='Temporal Link Prediction')
    parser.add_argument('--data_dir', type=str, default='data', help='Directory containing the datasets')
    parser.add_argument('--model', type=str, default='tgnn', help='Model type: tgnn, edge_gnn, or gcn')
    parser.add_argument('--hidden_dim', type=int, default=64, help='Hidden dimension size')
    parser.add_argument('--output_dim', type=int, default=32, help='Output dimension size')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of GNN layers')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='Weight decay')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--patience', type=int, default=10, help='Patience for early stopping')
    parser.add_argument('--window_size', type=int, default=86400 * 7, help='Time window size in seconds (default: 7 days)')
    parser.add_argument('--stride', type=int, default=86400 * 7, help='Stride between consecutive windows in seconds')
    parser.add_argument('--max_windows', type=int, default=5, help='Maximum number of time windows to process')
    parser.add_argument('--temperature', type=float, default=3.0, help='Temperature scaling for predictions')
    parser.add_argument('--no_cuda', action='store_true', default=False, help='Disables CUDA training')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--test_mode', action='store_true', default=False, help='Test mode')
    
    args = parser.parse_args()
    
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    
    return args

def main():
    args = parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    
    # Set device
    device = torch.device('cuda' if args.cuda else 'cpu')
    print(f'Using device: {device}')
    
    # Data paths
    edges_path = os.path.join(args.data_dir, 'edges_train_A.csv')
    node_features_path = os.path.join(args.data_dir, 'node_features.csv')
    edge_type_features_path = os.path.join(args.data_dir, 'edge_type_features.csv')
    test_input_path = os.path.join(args.data_dir, 'input_A_initial.csv')  # Test input
    test_output_path = 'output_A.csv'  # Test output
    
    # Load data
    print("Loading data...")
    data_dict = load_data(edges_path, node_features_path, edge_type_features_path)
    
    # Create time windows
    print("Creating time windows...")
    time_windows = create_time_windows(data_dict['edges_df'], args.window_size, args.stride)
    
    # Limit number of time windows to process
    if len(time_windows) > args.max_windows:
        print(f"Limiting to {args.max_windows} time windows (out of {len(time_windows)})")
        time_windows = time_windows[:args.max_windows]
    
    # Process data for temporal link prediction
    print("Processing temporal data...")
    temporal_graphs = process_temporal_data(data_dict, time_windows)
    
    # Prepare data splits for training/validation/testing
    print("Preparing link prediction data...")
    data_splits = prepare_link_prediction_data(temporal_graphs)
    
    # Model parameters
    node_features_dim = data_dict['node_features'].shape[1]
    
    # Check the actual edge feature dimensions
    if data_splits and hasattr(data_splits[0][0], 'edge_attr') and data_splits[0][0].edge_attr is not None:
        if len(data_splits[0][0].edge_attr.shape) > 1:
            edge_features_dim = data_splits[0][0].edge_attr.shape[1]  # Use actual dimension
        else:
            edge_features_dim = 1  # Single feature
    else:
        edge_features_dim = 1  # Default to 1 if no edge features
        
    print(f"Node features dimension: {node_features_dim}")
    print(f"Edge features dimension: {edge_features_dim}")
    
    hidden_dim = args.hidden_dim
    output_dim = args.output_dim
    
    # Initialize model
    print(f"Initializing {args.model} model...")
    if args.model == 'tgnn':
        model = TGNNModel(
            node_features_dim=node_features_dim,
            edge_features_dim=edge_features_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=args.num_layers,
            dropout=args.dropout
        )
    elif args.model == 'edge_gnn':
        model = EdgeTemporalGNN(
            node_features_dim=node_features_dim,
            edge_features_dim=edge_features_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=args.num_layers,
            dropout=args.dropout
        )
    else:  # Default to GCN
        model = TemporalGCN(
            in_channels=node_features_dim,
            hidden_channels=hidden_dim,
            out_channels=output_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
            edge_dim=edge_features_dim
        )
    
    # Set up optimizer with gradient clipping
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    if args.test_mode:
        # Test mode: load the model and make predictions
        print("Test mode: Making predictions...")
        
        # Prepare test data
        test_df = prepare_test_data(test_input_path, data_dict['node_mapping'])
        test_graph = process_test_data(test_df, data_dict)
        
        # Load model (assuming it's been trained and saved)
        if os.path.exists('saved_model.pt'):
            model.load_state_dict(torch.load('saved_model.pt', map_location=device))
            print("Loaded saved model.")
        else:
            print("No saved model found. Training a new model...")
            # Train the model on a single time window
            if data_splits:
                train_data, val_data, _ = data_splits[0]  # Use the first time window for simplicity
                model, history = train_model(
                    model=model,
                    train_data=train_data,
                    val_data=val_data,
                    optimizer=optimizer,
                    device=device,
                    epochs=args.epochs,
                    patience=args.patience
                )
                # Save the model
                torch.save(model.state_dict(), 'saved_model.pt')
                print("Model saved.")
                
                # Plot training history
                plot_training_history(history)
            else:
                print("No valid data splits found. Cannot train a model.")
                return
        
        # Make predictions
        x = test_graph.x
        edge_index = test_graph.edge_index
        edge_attr = test_graph.edge_attr if hasattr(test_graph, 'edge_attr') else None
        target_edge_index = torch.tensor(test_df[['source_mapped', 'target_mapped']].values.T, dtype=torch.long)
        
        predictions = predict(
            model=model,
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            target_edge_index=target_edge_index,
            device=device,
            temperature=args.temperature
        )
        
        # Write predictions to output file
        write_predictions(test_output_path, predictions)
        print(f"Predictions saved to {test_output_path}")
    
    else:
        # Training mode
        print("Training mode: Training and evaluating model...")
        
        # Keep track of best model and validation performance
        best_val_auc = 0
        best_epoch = 0
        best_model_state = None
        
        # Train the model for each time window
        if data_splits:
            for i, (train_data, val_data, test_data) in enumerate(data_splits):
                print(f"Training on time window {i+1}/{len(data_splits)}...")
                
                model, history = train_model(
                    model=model,
                    train_data=train_data,
                    val_data=val_data,
                    optimizer=optimizer,
                    device=device,
                    epochs=args.epochs,
                    patience=args.patience
                )
                
                # Check if this model is better than previous ones
                val_auc = history['val_auc'][-1] if history['val_auc'] else 0
                if val_auc > best_val_auc:
                    best_val_auc = val_auc
                    best_epoch = i
                    best_model_state = {key: value.cpu().clone() for key, value in model.state_dict().items()}
            
            # Save the best model
            if best_model_state is not None:
                model.load_state_dict(best_model_state)
                torch.save(model.state_dict(), 'saved_model.pt')
                print(f"Best model from time window {best_epoch+1} saved with validation AUC: {best_val_auc:.4f}")
            
            # Plot training history
            plot_training_history(history)
        else:
            print("No valid data splits found. Cannot train a model.")

if __name__ == "__main__":
    start_time = time.time()
    main()
    print(f"Total execution time: {time.time() - start_time:.2f} seconds") 