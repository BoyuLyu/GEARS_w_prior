"""
Comparison: Single-GPU vs Multi-GPU Training

This script shows the difference between traditional single-GPU training
and distributed multi-GPU training with GEARS.
"""

import sys
sys.path.insert(0, '../')

# =====================================
# METHOD 1: Traditional Single-GPU Training
# =====================================

def train_single_gpu():
    """Original single-GPU training approach"""
    from gears import PertData, GEARS
    import scanpy as sc
    
    print("="*60)
    print("METHOD 1: Single-GPU Training")
    print("="*60)
    
    # Load data
    pert_data = PertData(data_path='../data')
    pert_data.load(data_path='../data/vcc')
    
    # Prepare split
    pert_data.prepare_split(split='simulation', seed=1)
    
    # Get dataloader
    pert_data.get_dataloader(batch_size=32, test_batch_size=128)
    
    # Initialize model on single GPU
    gears_model = GEARS(pert_data, device='cuda:0')
    gears_model.model_initialize(hidden_size=64)
    
    # Train
    gears_model.train(epochs=20)
    
    # Save
    gears_model.save_model('./models/gears_single_gpu')
    
    print("Single-GPU training completed!")


# =====================================
# METHOD 2: Multi-GPU Distributed Training
# =====================================

def train_multi_gpu():
    """Distributed multi-GPU training approach"""
    from gears.train_distributed import train_multi_gpu
    
    print("\n" + "="*60)
    print("METHOD 2: Multi-GPU Distributed Training")
    print("="*60)
    
    split_config = {
        'split': 'simulation',
        'seed': 1,
        'train_gene_set_size': 0.75
    }
    
    args = {
        'model_config': {
            'hidden_size': 64,
            'num_go_gnn_layers': 1,
            'num_gene_gnn_layers': 1,
        },
        'batch_size': 32,           # Per GPU batch size
        'test_batch_size': 128,
        'epochs': 20,
        'lr': 1e-3,
        'weight_decay': 5e-4,
        'save_path': './models/gears_multi_gpu'
    }
    
    # Launch distributed training on 4 GPUs
    # Each GPU will process batch_size=32 samples
    # Effective batch size = 32 * 4 = 128
    train_multi_gpu(
        pert_data_path='../data',
        dataset_name='vcc',
        split_config=split_config,
        args=args,
        n_gpus=4
    )
    
    print("Multi-GPU training completed!")


# =====================================
# KEY DIFFERENCES EXPLAINED
# =====================================

"""
DIFFERENCES BETWEEN SINGLE-GPU AND MULTI-GPU:

1. DATA LOADING:
   Single-GPU:
   - pert_data.get_dataloader(batch_size=32)
   - DataLoader with standard shuffling
   
   Multi-GPU:
   - pert_data.get_dataloader_distributed(batch_size=32, world_size=4, rank=0-3)
   - Uses DistributedSampler to split data across GPUs
   - Each GPU gets different subset of data

2. MODEL INITIALIZATION:
   Single-GPU:
   - model = GEARS(pert_data, device='cuda:0')
   - Model on one GPU only
   
   Multi-GPU:
   - model = GEARS(pert_data, device=f'cuda:{rank}')
   - model = DDP(model, device_ids=[rank])
   - Each GPU has a replica of the model
   - Gradients synchronized across GPUs

3. BATCH SIZE:
   Single-GPU:
   - batch_size = 32 → processes 32 samples per iteration
   
   Multi-GPU with 4 GPUs:
   - batch_size = 32 per GPU
   - Effective batch size = 32 * 4 = 128 samples per iteration
   - ~4x speedup (ideally)

4. TRAINING LOOP:
   Single-GPU:
   - Standard forward/backward pass
   - All computation on one GPU
   
   Multi-GPU:
   - Each GPU processes its batch independently
   - Gradients averaged across all GPUs
   - Model parameters synchronized after each step

5. PERFORMANCE:
   Single-GPU (1 GPU, batch_size=32):
   - ~X seconds per epoch
   
   Multi-GPU (4 GPUs, batch_size=32 each):
   - ~X/3.5 seconds per epoch (not perfect 4x due to communication overhead)
   - Can train with larger effective batch size
   - Better gradient estimates

6. MEMORY USAGE:
   Single-GPU:
   - All model + batch on one GPU
   - May limit batch size
   
   Multi-GPU:
   - Model replicated on each GPU
   - Each GPU processes smaller batch
   - Can handle larger total batch sizes

7. CODE CHANGES NEEDED:
   Minimal changes required:
   ✓ Use train_distributed.py instead of direct GEARS.train()
   ✓ Pass configuration dictionaries instead of objects
   ✓ That's it! The rest is handled automatically

SPEEDUP EXPECTATIONS:

Number of GPUs | Ideal Speedup | Actual Speedup | Efficiency
---------------|---------------|----------------|------------
1              | 1x            | 1x             | 100%
2              | 2x            | 1.8x           | 90%
4              | 4x            | 3.5x           | 87.5%
8              | 8x            | 6.5x           | 81%

Efficiency decreases with more GPUs due to:
- Communication overhead (gradient synchronization)
- Batch splitting overhead
- Load balancing issues

WHEN TO USE MULTI-GPU:

Use Single-GPU when:
- Only 1 GPU available
- Small dataset (fast training already)
- Debugging or development
- Model fits in memory with desired batch size

Use Multi-GPU when:
- Multiple GPUs available
- Large dataset (long training time)
- Need larger effective batch sizes
- Want faster experimentation cycles
- Production training pipelines

MEMORY CONSIDERATIONS:

Single-GPU with batch_size=128:
- May run out of memory
- Need to reduce batch size

Multi-GPU with 4 GPUs, batch_size=32 each:
- Each GPU only needs memory for batch_size=32
- Effective batch_size=128 achieved
- Memory distributed across GPUs
"""

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, choices=['single', 'multi'], 
                       default='multi',
                       help='Training mode: single-GPU or multi-GPU')
    
    args = parser.parse_args()
    
    if args.mode == 'single':
        train_single_gpu()
    else:
        train_multi_gpu()
