# Temporal Link Prediction Approach

## Problem Overview
This project tackles the problem of temporal link prediction in dynamic graphs, where the goal is to predict the probability of edge formation between given nodes within a specific time range. Unlike traditional link prediction, temporal link prediction accounts for the evolving nature of graphs, making it suitable for various real-world applications like social network event forecasting and user engagement prediction.

## Data Understanding
We work with a dynamic event graph dataset (Dataset A) containing:
- **edges_train_A.csv**: Contains temporal edges with source node, destination node, edge type, and timestamp information.
- **node_features.csv**: Contains node features with categorical attributes (-1 indicates missing values).
- **edge_type_features.csv**: Contains edge type features with anonymized categorical attributes.

## Technical Approach

### 1. Data Preprocessing
- **Feature Engineering**: Handle missing values in node features using zero-imputation.
- **Temporal Windowing**: Divide the temporal graph into discrete time windows for capturing temporal dynamics.
- **Node and Edge Indexing**: Map node IDs to consecutive integers for efficient graph representation.
- **Data Splitting**: Create train/validation/test splits for each time window.

### 2. Model Architecture
We implement three different GNN architectures for temporal link prediction:

#### A. Temporal GCN (TemporalGCN)
- Base Graph Convolutional Network with temporal edge features
- Skip connections between layers for improved gradient flow
- Specialized edge encoding for temporal information

#### B. Edge-focused Temporal GNN (EdgeTemporalGNN)
- Utilizes rich edge features and temporal information
- Uses TransformerConv layers for enhanced attention to edge attributes
- Includes separate encoders for nodes and edges

#### C. Temporal Graph Neural Network (TGNNModel)
- Time-aware architecture with specialized time encoding
- Processes timestamps using periodic encoding for capturing temporal patterns
- Combines edge type and time features for enhanced representation

### 3. Training Strategy
- **Temporal Awareness**: Models are trained on multiple time windows to capture temporal dynamics.
- **Early Stopping**: Implemented to prevent overfitting based on validation AUC.
- **Negative Sampling**: Generation of non-existent edges as negative samples.
- **Loss Function**: Binary cross-entropy loss for link prediction.

### 4. Evaluation
- **Primary Metric**: Area Under the ROC Curve (AUC)
- **Secondary Metrics**: Precision-Recall AUC for imbalanced scenarios
- **Model Selection**: Best model selected based on validation performance across time windows

## Implementation Details

### Key Components
1. **Data Processing Module** (`data_processing.py`): Handles temporal graph processing, negative sampling, and data preparation.
2. **Model Module** (`model.py`): Contains the GNN model implementations with temporal awareness.
3. **Training Module** (`train.py`): Implements training, evaluation, and prediction functions.
4. **Main Script** (`main.py`): Integrates all components for end-to-end execution.

### Technical Innovations
1. **Time Encoding**: Custom time encoder to capture temporal patterns in edge formation.
2. **Adaptive Model Selection**: Framework to choose the best model type based on the dataset characteristics.
3. **Hierarchical Feature Processing**: Separate processing of node features, edge features, and temporal information.

## Results and Observations
- The TGNN model generally performs best due to its specialized time encoding mechanism.
- Edge features significantly improve prediction quality, especially in dynamic event graphs.
- Temporal windowing helps capture evolving graph patterns effectively.

## Usage Instructions
Run training:
```
python run.py --mode train --model tgnn --epochs 100
```

Generate predictions:
```
python run.py --mode test --model tgnn
```

## Future Improvements
1. Implement more sophisticated time encoding methods (e.g., continuous-time embeddings).
2. Explore heterogeneous graph models for better edge type representation.
3. Use contrastive learning for enhanced node embeddings in temporal settings.
4. Investigate memory-based approaches for long-term temporal dependencies. 