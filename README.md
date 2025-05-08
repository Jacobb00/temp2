# Temporal Link Prediction Using Graph Neural Networks

This project focuses on temporal link prediction in graphs using PyTorch Geometric.

## Project Structure
- `data/`: Contains the dataset files
- `src/`: Contains the source code
  - `data_processing.py`: Code for preprocessing data
  - `model.py`: Temporal GNN models
  - `train.py`: Training and evaluation code
  - `utils.py`: Utility functions
  - `main.py`: Main script to run the pipeline

## Setup and Installation
1. Clone this repository
2. Install the dependencies:
```
pip install -r requirements.txt
```

## Running the Project
```
python src/main.py
```

## Dataset
The project uses a dynamic event graph dataset with entities as nodes and events as edges.
- `edges_train_A.csv`: Temporal edges with source node, destination node, edge type, and timestamp
- `node_features.csv`: Node features with categorical attributes
- `edge_type_features.csv`: Edge type features with anonymized categorical attributes 