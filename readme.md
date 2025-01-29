## Environment Setup

### Requirements
- Python 3.8+
- PyTorch 1.12+
- DGL 1.0+
- scikit-learn
- numpy
- pandas

You can install the required packages using:
```bash
pip install torch dgl scikit-learn numpy pandas
```

## Running the Code

### Basic Usage
To train the model with default parameters:
```bash
python main.py
```

### Main Parameters
- `--seed`: Random seed (default: 12)
- `--n-hidden`: Number of hidden units (default: 16)
- `--n-layers`: Number of GNN layers (default: 2)
- `--dropout`: Dropout rate (default: 0.1)
- `--lr`: Learning rate (default: 1e-3)
- `--n-epochs`: Number of training epochs (default: 1000)
- `--trials`: Number of trials (default: 10)
- `--use_attention`: Enable attention mechanism (0/1)
- `--use_contrast`: Enable contrastive learning (0/1)
- `--use_llm`: LLM enhancement type ('none'/'type0'/'type1')
- `--lambda_contrast`: Weight for contrastive loss (default: 0.05)
- `--temperature`: Temperature for contrastive loss (default: 0.5)


### Output
The model will output:
- Training process metrics
- Final test results including AUC-ROC, AUC-PR, and Recall@K
- Results will be saved to CSV file (default: 'results/ours_results_tmp.csv')

## Data
Place your data files in the `processed_data` directory:
- `hetero_graph_woLLM_split.pth`: Graph without LLM enhancement
- `hetero_graph_wLLM_type0_split.pth`: Graph with Type-0 LLM enhancement
- `hetero_graph_wLLM_type1_split.pth`: Graph with Type-1 LLM enhancement
