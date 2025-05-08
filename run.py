import os
import argparse
import subprocess

def parse_args():
    parser = argparse.ArgumentParser(description='Run Temporal Link Prediction')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'], 
                        help='Mode: train or test')
    parser.add_argument('--model', type=str, default='tgnn', choices=['tgnn', 'edge_gnn', 'gcn'], 
                        help='Model type: tgnn, edge_gnn, or gcn')
    parser.add_argument('--cuda', action='store_true', help='Use CUDA if available')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--hidden_dim', type=int, default=128, help='Hidden dimension size')
    parser.add_argument('--temperature', type=float, default=15.0, help='Temperature scaling for predictions')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Construct command
    cmd = ['python', 'src/main.py']
    
    # Add arguments
    if args.mode == 'test':
        cmd.append('--test_mode')
    
    cmd.extend(['--model', args.model])
    cmd.extend(['--epochs', str(args.epochs)])
    cmd.extend(['--hidden_dim', str(args.hidden_dim)])
    cmd.extend(['--temperature', str(args.temperature)])
    
    if not args.cuda:
        cmd.append('--no_cuda')
    
    # Run command
    print(f"Running command: {' '.join(cmd)}")
    subprocess.run(cmd)

if __name__ == "__main__":
    main() 