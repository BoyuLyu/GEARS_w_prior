"""
Example script for multi-GPU distributed training using DistributedDataParallel

This script demonstrates how to train GEARS model across multiple GPUs
using PyTorch's DistributedDataParallel (DDP).
"""

import sys
sys.path.insert(0, '../')

from gears.train_distributed import train_multi_gpu

# Configuration for data splitting
split_config = {
    'split': 'simulation',          # Type of split: 'simulation', 'combo_seen0', etc.
    'seed': 1,                       # Random seed for reproducibility
    'train_gene_set_size': 0.75     # Fraction of genes for training
}

# Model and training configuration
args = {
    # Model architecture parameters
    'model_config': {
        'hidden_size': 64,                              # Hidden dimension
        'num_go_gnn_layers': 1,                         # GO graph GNN layers
        'num_gene_gnn_layers': 1,                       # Co-expression graph GNN layers
        'decoder_hidden_size': 16,                      # Decoder hidden size
        'num_similar_genes_go_graph': 20,               # K neighbors in GO graph
        'num_similar_genes_co_express_graph': 20,       # K neighbors in co-expression
        'coexpress_threshold': 0.4,                     # Correlation threshold
        'uncertainty': False,                            # Uncertainty mode
        'uncertainty_reg': 1.0,                         # Uncertainty regularization
        'direction_lambda': 0.1                         # Direction loss weight
    },
    
    # Training hyperparameters
    'batch_size': 32,                # Batch size per GPU
    'test_batch_size': 128,          # Test batch size per GPU
    'epochs': 20,                    # Number of training epochs
    'lr': 1e-3,                      # Learning rate
    'weight_decay': 5e-4,            # Weight decay
    
    # Save path
    'save_path': './models/gears_ddp'
}

if __name__ == '__main__':
    # Launch distributed training
    # This will spawn multiple processes, one for each GPU
    
    train_multi_gpu(
        pert_data_path='../data',           # Path to data directory
        dataset_name='vcc',                  # Dataset name (folder in data_path)
        split_config=split_config,           # Split configuration
        args=args,                           # Training arguments
        n_gpus=4,                            # Number of GPUs (None = all available)
        gpu_ids=None                         # Specific GPU IDs (None = 0,1,2,...)
    )
    
    print("\nTraining completed!")
    print(f"Model saved to: {args['save_path']}")

"""
USAGE EXAMPLES:

1. Train on all available GPUs:
   python train_ddp_example.py

2. Train on specific number of GPUs:
   Modify n_gpus parameter in train_multi_gpu()
   
3. Train on specific GPU IDs (e.g., GPUs 2, 3, 5, 7):
   train_multi_gpu(..., n_gpus=4, gpu_ids=[2, 3, 5, 7])

4. Adjust batch size:
   - Effective batch size = batch_size * n_gpus
   - For 4 GPUs with batch_size=32: effective_batch_size = 128

5. Monitor training:
   - Only GPU 0 (rank 0) will print training logs
   - All GPUs participate in training

IMPORTANT NOTES:

- Make sure to process your data first using:
  pert_data.new_data_process(dataset_name='vcc', adata=adata)
  
- The data should be in: {pert_data_path}/{dataset_name}/
  
- Each GPU process will load data independently
  
- Model is saved only by rank 0 process

- For better performance:
  - Increase batch_size if GPU memory allows
  - Adjust num_workers in get_dataloader_distributed()
  - Use pin_memory=True for faster data transfer
"""
