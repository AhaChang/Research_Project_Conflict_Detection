import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch
import dgl
from model import SHADE
from sklearn.metrics import roc_auc_score, average_precision_score
import os
import random
import pandas as pd
def set_seed(seed=3407):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    dgl.random.seed(seed)
    torch.use_deterministic_algorithms(True)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

def pyg_to_dgl(data):
    # Store graph data in dictionary
    graph_data = {}
    
    # Convert edge data
    for edge_type in data.edge_types:
        src_type, edge_name, dst_type = edge_type
        edge_index = data[edge_type].edge_index
        graph_data[edge_type] = (edge_index[0], edge_index[1])
    
    # Create heterogeneous graph
    g = dgl.heterograph(graph_data)
    
    # Add node features
    for node_type in data.node_types:
        if 'x' in data[node_type]:
            g.nodes[node_type].data['feat'] = data[node_type].x
        if 'y' in data[node_type]:
            g.nodes[node_type].data['label'] = data[node_type].y
    
    return g

def calculate_metrics_all_types(logits_dict, labels_dict, idx_dict):
    # Collect predictions and labels for all nodes
    all_probs = []
    all_labels = []
    all_preds = []
    
    for ntype in logits_dict.keys():
        if ntype not in labels_dict or ntype not in idx_dict:
            continue
            
        logits = logits_dict[ntype]
        labels = labels_dict[ntype]
        idx = idx_dict[ntype]
        
        # Get predicted probabilities and true labels
        probs = F.softmax(logits[idx], dim=1).detach().cpu().numpy()
        labels_np = labels[idx].cpu().numpy()
        preds = logits[idx].argmax(dim=1).cpu().numpy()
        
        all_probs.append(probs)
        all_labels.append(labels_np)
        all_preds.append(preds)
    
    # Combine results from all nodes
    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)
    all_preds = np.concatenate(all_preds)
    
    # Calculate overall metrics
    auc_roc = roc_auc_score(all_labels, all_probs[:, 1])
    auc_pr = average_precision_score(all_labels, all_probs[:, 1])
    
    k = int(np.sum(all_labels))
    recall_k = np.sum(all_labels[all_probs[:, 1].argsort()[-k:]]) / np.sum(all_labels)

    return {
        'auc_roc': auc_roc,
        'auc_pr': auc_pr,
        'recall_k': recall_k,
        'k': k
    }

def supervised_info_nce_loss(features, labels, src, dst, temperature=0.5, k_pos=5, k_neg=5):
    """
    Supervised contrastive learning loss with graph structure, selecting TopK positive and negative samples
    Args:
        features: Training node features [N_train, hidden_dim]
        labels: Training node labels [N_train]
        src: Source node indices in training set
        dst: Target node indices in training set
        temperature: Temperature parameter
        k_pos: Number of positive samples to select
        k_neg: Number of negative samples to select
    """
    # Normalize features
    features = F.normalize(features, dim=1)
    
    # Calculate similarity matrix
    sim_matrix = torch.matmul(features, features.T) / temperature
    
    # Create adjacency matrix
    adj_matrix = torch.zeros(len(features), len(features), device=features.device)
    valid_edges = (src < len(features)) & (dst < len(features))
    src = src[valid_edges]
    dst = dst[valid_edges]
    adj_matrix[src, dst] = 1
    
    # Define positive candidates: adjacent and same label
    same_label_mask = (labels.unsqueeze(0) == labels.unsqueeze(1))
    pos_candidates = same_label_mask & adj_matrix.bool()
    
    # Define negative candidates: different label nodes
    neg_candidates = ~same_label_mask
    
    # Remove self
    diag_mask = torch.eye(len(features), dtype=torch.bool, device=features.device)
    pos_candidates.masked_fill_(diag_mask, False)
    neg_candidates.masked_fill_(diag_mask, False)
    
    # Calculate exp(sim/τ)
    exp_sim = torch.exp(sim_matrix)
    
    # Select TopK positive samples
    pos_sim_matrix = exp_sim * pos_candidates
    pos_mask = torch.zeros_like(pos_sim_matrix)
    for i in range(len(features)):
        if pos_candidates[i].sum() > 0:
            # Get similarity of positive samples
            pos_sims = pos_sim_matrix[i][pos_candidates[i]]
            # Select TopK most similar positive samples
            k = min(k_pos, len(pos_sims))
            if k > 0:
                _, top_pos_indices = torch.topk(pos_sims, k=k)
                pos_indices = torch.where(pos_candidates[i])[0][top_pos_indices]
                pos_mask[i, pos_indices] = 1

    # Select TopK negative samples
    neg_sim_matrix = exp_sim * neg_candidates
    neg_mask = torch.zeros_like(neg_sim_matrix)
    for i in range(len(features)):
        if neg_candidates[i].sum() > 0:
            # Get similarity of negative samples
            neg_sims = neg_sim_matrix[i][neg_candidates[i]]
            # Select TopK most similar negative samples
            k = min(k_neg, len(neg_sims))
            if k > 0:
                _, top_neg_indices = torch.topk(neg_sims, k=k)
                neg_indices = torch.where(neg_candidates[i])[0][top_neg_indices]
                neg_mask[i, neg_indices] = 1
    
    # Calculate loss
    pos_sim = exp_sim * pos_mask
    neg_sim = exp_sim * neg_mask
    
    # Calculate loss only for nodes with positive samples
    valid_mask = pos_mask.sum(dim=1) > 0
    if valid_mask.sum() > 0:
        loss = -torch.log(pos_sim.sum(dim=1)[valid_mask] / (pos_sim.sum(dim=1) + neg_sim.sum(dim=1))[valid_mask]).mean()
    else:
        loss = torch.tensor(0.0, device=features.device)
    
    return loss

def main(args):
    # Load graph data
    if args.dataset == 'ours':
        if args.use_llm == 'type0':
            hetero_data = torch.load('processed_data/hetero_graph_wLLM_type0_split.pth', weights_only=False, map_location=torch.device('cpu'))
        elif args.use_llm == 'type1':
            hetero_data = torch.load('processed_data/hetero_graph_wLLM_type1_split.pth', weights_only=False, map_location=torch.device('cpu'))
        else:
            hetero_data = torch.load('processed_data/hetero_graph_woLLM_split.pth', weights_only=False, map_location=torch.device('cpu'))
        g = pyg_to_dgl(hetero_data)
        
        # Create dictionaries for labels and indices
        labels_dict = {}
        train_idx_dict, val_idx_dict, test_idx_dict = {},{},{}
        
        # For each node type, add to dictionary if there's a label
        for ntype in g.ntypes:
            if hasattr(hetero_data[ntype], 'y'):
                labels_dict[ntype] = hetero_data[ntype].y.long()
                train_idx_dict[ntype] = hetero_data[ntype].train_mask
                val_idx_dict[ntype] = hetero_data[ntype].val_mask
                test_idx_dict[ntype] = hetero_data[ntype].test_mask
        
        # Find maximum feature dimension
        max_dim = max(
            g.nodes[ntype].data['feat'].shape[1] 
            for ntype in g.ntypes 
            if 'feat' in g.nodes[ntype].data
        )
        
        # Pad each node type's features
        for ntype in g.ntypes:
            if 'feat' in g.nodes[ntype].data:
                feat = g.nodes[ntype].data['feat']
                curr_dim = feat.shape[1]
                if curr_dim < max_dim:
                    # Zero-pad feature dimension
                    padding = torch.zeros(feat.shape[0], max_dim - curr_dim, device=feat.device)
                    g.nodes[ntype].data['feat'] = torch.cat([feat, padding], dim=1)

    model = SHADE(
        g,
        h_dim=args.n_hidden,
        out_dim=2,
        num_bases=args.n_bases,
        num_hidden_layers=args.n_layers - 2,
        dropout=args.dropout,
        use_self_loop=args.use_self_loop,
        use_attention=args.use_attention,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2norm)

    # Add variables to track best performance
    best_epoch = 0
    best_val_auc = 0
    best_test_metrics = None
    best_model_state = None
    
    # Initialize early stopping variables
    best_loss = float('inf')
    patience = 20
    min_delta = 1e-4
    counter = 0
    
    # training loop
    model.train()
    for epoch in range(args.n_epochs):
        optimizer.zero_grad()
            
        node_features = {
            ntype: g.nodes[ntype].data['feat'] 
            for ntype in g.ntypes 
            if 'feat' in g.nodes[ntype].data
        }
        
        logits_dict, latent_h = model(node_features)
        
        # Calculate total loss
        loss = 0
        overall_weights = 0
        num_0, num_1 = 0, 0
        for ntype in labels_dict:
            if ntype in logits_dict and ntype in train_idx_dict:
                # Calculate class weights
                num_0_tmp = (labels_dict[ntype][train_idx_dict[ntype]].shape[0] - labels_dict[ntype][train_idx_dict[ntype]].sum())
                num_1_tmp = labels_dict[ntype][train_idx_dict[ntype]].sum()
                num_0 += num_0_tmp
                num_1 += num_1_tmp
        overall_weights = num_0/num_1

        for ntype in labels_dict:
            if ntype in logits_dict and ntype in train_idx_dict:
                # Calculate class weights
                weights = (labels_dict[ntype][train_idx_dict[ntype]].shape[0] - labels_dict[ntype][train_idx_dict[ntype]].sum())/labels_dict[ntype][train_idx_dict[ntype]].sum()

                type_loss = F.cross_entropy(
                    logits_dict[ntype][train_idx_dict[ntype]], 
                    labels_dict[ntype][train_idx_dict[ntype]],
                    weight=torch.tensor([1, overall_weights], device=labels_dict[ntype].device)  # Add class weights
                )
                loss += type_loss
        
        # Contrastive loss
        contrast_loss = 0
        for ntype in latent_h:
            if ntype not in train_idx_dict or ntype not in labels_dict:
                continue
            
            # Get features and labels of training set nodes
            train_features = latent_h[ntype][train_idx_dict[ntype]]
            train_labels = labels_dict[ntype][train_idx_dict[ntype]]
            
            # Get all related edge types
            relevant_etypes = []
            for etype in g.etypes:
                canonical_etype = g.to_canonical_etype(etype)
                _, _, dst_type = canonical_etype
                if dst_type == ntype:
                    relevant_etypes.append(canonical_etype)
            
            # Collect all related edges
            all_src = []
            all_dst = []
            for canonical_etype in relevant_etypes:
                src, dst = g.edges(etype=canonical_etype)
                # Get mapping of training set nodes
                train_nodes = torch.where(train_idx_dict[ntype])[0]
                node_to_train_idx = {int(n): i for i, n in enumerate(train_nodes)}
                
                # Only keep edges in training set and convert to training set indices
                edge_mask = train_idx_dict[ntype][dst]
                src = src[edge_mask]
                dst = dst[edge_mask]
                
                # Convert to relative indices in training set
                src_mapped = torch.tensor([node_to_train_idx.get(int(s), -1) for s in src], 
                                       device=src.device)
                dst_mapped = torch.tensor([node_to_train_idx.get(int(d), -1) for d in dst], 
                                        device=dst.device)
                
                # Only keep valid mapping
                valid_edges = (src_mapped >= 0) & (dst_mapped >= 0)
                all_src.append(src_mapped[valid_edges])
                all_dst.append(dst_mapped[valid_edges])
            
            # If there are related edges
            if args.use_contrast:
                if len(all_src) > 0 and all(len(src) > 0 for src in all_src):
                    src = torch.cat(all_src)
                    dst = torch.cat(all_dst)
                    
                    type_contrast_loss = supervised_info_nce_loss(
                        train_features,
                        train_labels,
                        src,
                        dst,
                        temperature=args.temperature
                    )
                    contrast_loss += type_contrast_loss
            else:
                contrast_loss = 0
        
        # 3. Total loss
        total_loss = loss + args.lambda_contrast * contrast_loss

        total_loss.backward()
        optimizer.step()
        
        # Evaluation
        model.eval()
        with torch.no_grad():
            node_features = {
                ntype: g.nodes[ntype].data['feat']
                for ntype in g.ntypes
                if 'feat' in g.nodes[ntype].data
            }
            logits_dict, latent_h = model(node_features)
            val_metrics = calculate_metrics_all_types(logits_dict, labels_dict, val_idx_dict)
            test_metrics = calculate_metrics_all_types(logits_dict, labels_dict, test_idx_dict)
        
            # Print training process
            print(f"Epoch {epoch:05d}, CLF Loss {loss:.4f}, Contrast Loss {contrast_loss:.4f}, Val AUC {val_metrics['auc_roc']:.4f}, PRC {val_metrics['auc_pr']:.4f}, RecK {val_metrics['recall_k']:.4f} Test AUC {test_metrics['auc_roc']:.4f}, PRC {test_metrics['auc_pr']:.4f}, RecK {test_metrics['recall_k']:.4f}")
            
            # Early stopping check
            if total_loss > best_loss - min_delta:
                counter += 1
                if counter >= patience:
                    print(f'Early stopping triggered at epoch {epoch}')
                    print(f'Best epoch: {best_epoch}, Best AUC: {best_val_auc:.4f}')
                    break
            else:
                counter = 0
                best_loss = total_loss
                best_epoch = epoch
                best_val_auc = val_metrics['auc_roc']
                best_test_metrics = test_metrics
    
    # Print best results
    print(f"\nTest results for best validation performance:")
    print(f"Epoch {best_epoch:05d}, Test AUC: {best_test_metrics['auc_roc']:.4f}, PRC: {best_test_metrics['auc_pr']:.4f}, RecK: {best_test_metrics['recall_k']:.4f}")
    return best_test_metrics
    # # Optional: Save best model
    # torch.save(best_model_state, f'best_model_seed{args.seed}.pth')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RGCN")
    parser.add_argument("--seed", type=int, default=12, help="random seed")
    parser.add_argument("--dropout", type=float, default=0.1, help="dropout probability")
    parser.add_argument("--n-hidden", type=int, default=16, help="number of hidden units")
    parser.add_argument("--gpu", type=int, default=-1, help="gpu")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--n-bases", type=int, default=-1, help="number of filter weight matrices, default: -1 [use all]")
    parser.add_argument("--n-layers", type=int, default=2, help="number of propagation rounds")
    parser.add_argument("-e", "--n_epochs", type=int, default=1000, help="number of training epochs")
    parser.add_argument("-d", "--dataset", type=str, default='ours', help="dataset to use")
    parser.add_argument("--l2norm", type=float, default=5e-3, help="l2 norm coef")
    parser.add_argument("--use-self-loop", default=False, action="store_true", help="include self feature as a special relation")
    parser.add_argument("--trials", type=int, default=10, help="number of trials")
    parser.add_argument("--temperature", type=float, default=0.5, help="temperature for contrastive loss")
    # hyper-parameters
    parser.add_argument("--lambda_contrast", type=float, default=0.05, help="lambda for contrastive loss")
    # ablation
    parser.add_argument("--use_attention", type=int, default=0, help="use attention")
    parser.add_argument("--use_contrast", type=int, default=0, help="use contrastive loss")
    parser.add_argument("--use_llm", default='type0', help="use LLM for semantic enhancement, ['none', 'type0', 'type1']")
    # Save results to CSV file
    parser.add_argument("--save_results_path", type=str, default='results/ours_results_tmp.csv', help="save results to CSV file")
    args = parser.parse_args()
    print(args)
    set_seed(args.seed)
    
    if args.use_attention == 1:
        args.use_attention = True
    else:
        args.use_attention = False
    if args.use_contrast == 1:
        args.use_contrast = True
    else:
        args.use_contrast = False
        
    all_results = {
        'auc_roc': [],
        'auc_pr': [],
        'recall_k': []
    }
    
    for trial in range(args.trials):
        best_test_metrics = main(args)
        for metric in ['auc_roc', 'auc_pr', 'recall_k']:
            all_results[metric].append(best_test_metrics[metric])
        
        print(f"Trial {trial+1} - Test AUC: {best_test_metrics['auc_roc']:.4f}, "
              f"PRC: {best_test_metrics['auc_pr']:.4f}, "
              f"RecK: {best_test_metrics['recall_k']:.4f}")
    
    # Calculate mean and standard deviation
    print("\nStatistics for all trials:")
    for metric in ['auc_roc', 'auc_pr', 'recall_k']:
        values = np.array(all_results[metric])
        mean = np.mean(values)
        std = np.std(values)
        print(f"{metric.upper()}: {mean:.4f} ± {std:.4f}")

    # Calculate mean results
    mean_results = {}
    for metric in ['auc_roc', 'auc_pr', 'recall_k']:
        values = np.array(all_results[metric])
        mean_results[f'{metric.upper()}_mean'] = np.mean(values)
        mean_results[f'{metric.upper()}_std'] = np.std(values)
    
    # If there's time information, add average time
    if 'time' in all_results:
        mean_results['Time'] = np.mean(all_results['time'])
    
    # Add all parameters to results
    args_dict = vars(args)
    mean_results.update(args_dict)
    
    # Create or append to CSV file
    result_df = pd.DataFrame([mean_results])
    if os.path.exists(args.save_results_path):
        result_df.to_csv(args.save_results_path, mode='a', header=False, index=False)
    else:
        result_df.to_csv(args.save_results_path, index=False)