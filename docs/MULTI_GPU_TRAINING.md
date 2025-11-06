# Multi-GPU Distributed Training Guide for GEARS

## Table of Contents
1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Detailed Modifications](#detailed-modifications)
4. [Usage Examples](#usage-examples)
5. [Performance Tuning](#performance-tuning)
6. [Troubleshooting](#troubleshooting)
7. [Advanced Topics](#advanced-topics)

---

## Overview

This guide explains how to use PyTorch DistributedDataParallel (DDP) for multi-GPU training with GEARS. DDP provides efficient distributed training with near-linear speedup.

### Why Use DDP?

- **Faster Training**: Near-linear speedup with multiple GPUs
- **Larger Batch Sizes**: Distribute batches across GPUs
- **Better Convergence**: Larger effective batch sizes improve gradient estimates
- **Efficient Communication**: All-reduce operations for gradient synchronization

### What Was Modified?

1. **`train_distributed.py`** (NEW): Main distributed training script
2. **`pertdata.py`**: Added `get_dataloader_distributed()` method
3. **`gears.py`**: Modified `train()`, `save_model()` for DDP compatibility

---

## Quick Start

### Prerequisites

```bash
# Required packages
pip install torch torch-geometric scanpy

# Verify multi-GPU setup
python -c "import torch; print(f'GPUs available: {torch.cuda.device_count()}')"
```

### Basic Usage

```python
from gears.train_distributed import train_multi_gpu

# Configure data splitting
split_config = {
    'split': 'simulation',
    'seed': 1,
    'train_gene_set_size': 0.75
}

# Configure model and training
args = {
    'model_config': {
        'hidden_size': 64,
        'num_go_gnn_layers': 1,
        'num_gene_gnn_layers': 1,
    },
    'batch_size': 32,      # Per GPU
    'epochs': 20,
    'lr': 1e-3,
    'save_path': './models/gears_ddp'
}

# Launch training on 4 GPUs
train_multi_gpu(
    pert_data_path='./data',
    dataset_name='vcc',
    split_config=split_config,
    args=args,
    n_gpus=4
)
```

---

## Detailed Modifications

### 1. train_distributed.py (New File)

**Location**: `gears/train_distributed.py`

**Key Functions**:

#### `setup_ddp(rank, world_size)`
Initializes the distributed process group.

```python
def setup_ddp(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
```

**What it does**:
- Sets up communication between GPU processes
- Uses NCCL backend for optimal GPU communication
- Assigns each process to a specific GPU

#### `train_ddp_worker(rank, world_size, ...)`
Worker function that runs on each GPU.

```python
def train_ddp_worker(rank, world_size, pert_data_path, dataset_name, split_config, args):
    # Setup DDP
    setup_ddp(rank, world_size)
    
    # Load data independently on each process
    pert_data = PertData(data_path=pert_data_path)
    pert_data.load(data_path=f'{pert_data_path}/{dataset_name}')
    pert_data.prepare_split(**split_config)
    
    # Get distributed dataloaders
    pert_data.get_dataloader_distributed(
        batch_size=args['batch_size'],
        world_size=world_size,
        rank=rank
    )
    
    # Initialize model and wrap with DDP
    gears_model = GEARS(pert_data, device=f'cuda:{rank}')
    gears_model.model_initialize(**args['model_config'])
    gears_model.model = DDP(gears_model.model, device_ids=[rank])
    
    # Train
    gears_model.train(epochs=args['epochs'], lr=args['lr'])
    
    # Save (only rank 0)
    if rank == 0:
        gears_model.save_model(args['save_path'])
    
    cleanup_ddp()
```

**Why separate workers?**
- Each GPU runs in its own Python process
- Processes communicate through NCCL for gradient synchronization
- Isolation prevents interference between GPUs

#### `train_multi_gpu(...)`
Main entry point for distributed training.

```python
def train_multi_gpu(pert_data_path, dataset_name, split_config, args, n_gpus=None):
    if n_gpus is None:
        n_gpus = torch.cuda.device_count()
    
    mp.spawn(
        train_ddp_worker,
        args=(n_gpus, pert_data_path, dataset_name, split_config, args),
        nprocs=n_gpus,
        join=True
    )
```

**What `mp.spawn()` does**:
- Spawns `n_gpus` processes
- Each process runs `train_ddp_worker` with a different `rank`
- Waits for all processes to complete

### 2. pertdata.py Modifications

**Added Method**: `get_dataloader_distributed()`

```python
def get_dataloader_distributed(self, batch_size, test_batch_size=None, 
                               world_size=1, rank=0, num_workers=4):
    from torch.utils.data.distributed import DistributedSampler
    
    # Create distributed sampler
    train_sampler = DistributedSampler(
        cell_graphs['train'],
        num_replicas=world_size,  # Total number of GPUs
        rank=rank,                 # Current GPU ID
        shuffle=True,
        drop_last=True
    )
    
    # Create dataloader with sampler
    train_loader = DataLoader(
        cell_graphs['train'],
        batch_size=batch_size,
        sampler=train_sampler,     # DistributedSampler instead of shuffle
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
```

**What DistributedSampler does**:
- Splits dataset into `world_size` equal parts
- Each GPU (rank) gets a different subset
- Ensures no data overlap between GPUs
- Automatically handles epoch shuffling

**Example with 4 GPUs and 1000 samples**:
- GPU 0: samples 0-249
- GPU 1: samples 250-499
- GPU 2: samples 500-749
- GPU 3: samples 750-999

### 3. gears.py Modifications

#### Added Distributed Attributes

```python
def __init__(self, pert_data, device='cuda', ...):
    # ... existing code ...
    
    # Distributed training attributes
    self.is_distributed = False
    self.rank = 0
    self.world_size = 1
```

#### Modified `train()` Method

**Key Changes**:

1. **Set epoch for DistributedSampler**:
```python
for epoch in range(epochs):
    if self.is_distributed and hasattr(train_loader, 'sampler'):
        train_loader.sampler.set_epoch(epoch)  # Important for shuffling!
```

**Why?** DistributedSampler needs the epoch to shuffle differently each epoch.

2. **Conditional Printing**:
```python
if not self.is_distributed or self.rank == 0:
    print_sys('Start Training...')
```

**Why?** Only rank 0 should print to avoid duplicate logs.

3. **Disable tqdm for non-zero ranks**:
```python
disable_tqdm = self.is_distributed and self.rank != 0
for step, batch in enumerate(tqdm(train_loader, disable=disable_tqdm)):
```

**Why?** Only rank 0 shows progress bar.

#### Modified `save_model()` Method

```python
def save_model(self, path):
    # ... existing code ...
    
    # Handle DistributedDataParallel wrapper
    model_to_save = self.best_model
    if isinstance(self.best_model, nn.parallel.DistributedDataParallel):
        model_to_save = self.best_model.module  # Unwrap DDP
    
    torch.save(model_to_save.state_dict(), os.path.join(path, 'model.pt'))
```

**Why?** DDP wraps the model, so we need to unwrap it before saving to avoid storing the wrapper structure.

---

## Usage Examples

### Example 1: Basic Multi-GPU Training

```python
from gears.train_distributed import train_multi_gpu

split_config = {'split': 'simulation', 'seed': 1, 'train_gene_set_size': 0.75}

args = {
    'model_config': {'hidden_size': 64},
    'batch_size': 32,
    'epochs': 20,
    'lr': 1e-3,
    'save_path': './models/gears_ddp'
}

train_multi_gpu(
    pert_data_path='./data',
    dataset_name='vcc',
    split_config=split_config,
    args=args,
    n_gpus=4
)
```

### Example 2: Specific GPU Selection

```python
# Use GPUs 2, 3, 5, 7 (maybe 0,1 are busy)
train_multi_gpu(
    pert_data_path='./data',
    dataset_name='vcc',
    split_config=split_config,
    args=args,
    n_gpus=4,
    gpu_ids=[2, 3, 5, 7]
)
```

### Example 3: All Available GPUs

```python
# Automatically use all available GPUs
train_multi_gpu(
    pert_data_path='./data',
    dataset_name='vcc',
    split_config=split_config,
    args=args,
    n_gpus=None  # None = use all
)
```

### Example 4: Different Split Types

```python
# Combo perturbation split
split_config = {
    'split': 'combo_seen0',
    'seed': 42,
    'train_gene_set_size': 0.8
}

train_multi_gpu(...)
```

### Example 5: Uncertainty Mode

```python
args = {
    'model_config': {
        'hidden_size': 64,
        'uncertainty': True,           # Enable uncertainty
        'uncertainty_reg': 1.0
    },
    'batch_size': 32,
    'epochs': 20,
    'lr': 1e-3,
    'save_path': './models/gears_uncertainty_ddp'
}

train_multi_gpu(...)
```

---

## Performance Tuning

### 1. Batch Size Selection

**Rule of Thumb**: Start with batch_size that fits on single GPU, then scale.

```python
# Single GPU: batch_size=32 fits in memory
# 4 GPUs: Use batch_size=32 per GPU → effective_batch=128

args = {
    'batch_size': 32,  # Per GPU
    # Effective batch size = 32 * 4 = 128
}
```

**Adjust based on**:
- GPU memory: Larger batch if memory allows
- Convergence: Larger batches may need higher learning rate
- Speed: Larger batches = fewer iterations but more compute per iteration

### 2. Number of Workers

```python
pert_data.get_dataloader_distributed(
    batch_size=32,
    num_workers=4  # Per GPU
)
```

**Guidelines**:
- Start with `num_workers=4`
- Increase if CPU utilization is low
- Decrease if memory usage is high
- Typical range: 2-8 workers per GPU

### 3. Learning Rate Scaling

When using larger effective batch sizes, scale learning rate:

```python
# Single GPU: lr=1e-3, batch_size=32
# 4 GPUs: lr=4e-3, batch_size=32*4=128

n_gpus = 4
base_lr = 1e-3
args = {
    'lr': base_lr * n_gpus,  # Linear scaling rule
    'batch_size': 32
}
```

**Alternative**: Use warmup schedule for large batch training.

### 4. Communication Optimization

For models with many parameters, reduce communication frequency:

```python
gears_model.model = DDP(
    gears_model.model,
    device_ids=[rank],
    find_unused_parameters=False,  # Faster if no unused params
    broadcast_buffers=False         # Faster if buffers don't need sync
)
```

### 5. Mixed Precision Training

Combine DDP with automatic mixed precision for even faster training:

```python
from torch.cuda.amp import autocast, GradScaler

# In train_ddp_worker:
scaler = GradScaler()

for batch in train_loader:
    with autocast():
        pred = model(batch)
        loss = loss_fct(...)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

**Benefits**:
- 2x faster training
- Reduced memory usage
- Negligible accuracy impact

---

## Troubleshooting

### Problem 1: "Connection timeout" or "Address already in use"

**Cause**: Port conflict or previous process not cleaned up.

**Solution**:
```python
# In setup_ddp(), change port:
os.environ['MASTER_PORT'] = '12356'  # Different port
```

Or kill existing processes:
```bash
pkill -f "python.*train_distributed"
```

### Problem 2: "CUDA out of memory"

**Cause**: Batch size too large for GPU.

**Solution**:
```python
# Reduce batch size per GPU
args = {
    'batch_size': 16,  # Instead of 32
}
```

### Problem 3: Different results across runs

**Cause**: Random seed not set consistently.

**Solution**:
```python
import random
import numpy as np
import torch

def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# Call in train_ddp_worker before anything else
set_seed(split_config['seed'])
```

### Problem 4: Slow training on multiple GPUs

**Possible causes**:
1. **Small batch size**: Increase batch_size to reduce overhead
2. **Too many workers**: Reduce num_workers
3. **Slow data loading**: Use pin_memory=True
4. **Communication bottleneck**: Check network between GPUs (for multi-node)

**Debug**:
```python
import time

# Add timing in training loop:
start = time.time()
for batch in train_loader:
    data_time = time.time() - start
    
    # Training step
    ...
    
    compute_time = time.time() - start - data_time
    
    if rank == 0:
        print(f"Data: {data_time:.3f}s, Compute: {compute_time:.3f}s")
    
    start = time.time()
```

### Problem 5: Model not saving

**Cause**: Only rank 0 saves model.

**Solution**: Check that rank 0 process completes successfully:
```python
# In train_ddp_worker:
if rank == 0:
    print(f"Saving model to {args['save_path']}")
    gears_model.save_model(args['save_path'])
    print("Model saved successfully!")
```

---

## Advanced Topics

### Multi-Node Training

For training across multiple machines:

```python
def setup_ddp_multi_node(rank, world_size, master_addr, master_port):
    os.environ['MASTER_ADDR'] = master_addr  # IP of main node
    os.environ['MASTER_PORT'] = master_port
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['RANK'] = str(rank)
    
    dist.init_process_group("nccl")
```

Launch on each node:
```bash
# Node 0 (4 GPUs):
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    --nnodes=2 \
    --node_rank=0 \
    --master_addr="192.168.1.1" \
    --master_port=12355 \
    train_script.py

# Node 1 (4 GPUs):
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    --nnodes=2 \
    --node_rank=1 \
    --master_addr="192.168.1.1" \
    --master_port=12355 \
    train_script.py
```

### Gradient Accumulation with DDP

Simulate even larger batch sizes:

```python
accumulation_steps = 4

for step, batch in enumerate(train_loader):
    pred = model(batch)
    loss = loss_fct(...) / accumulation_steps
    loss.backward()
    
    if (step + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

**Effective batch size**: batch_size * n_gpus * accumulation_steps

### Monitoring with Weights & Biases

```python
# Only initialize wandb on rank 0
if rank == 0:
    import wandb
    wandb.init(project='gears_ddp', name=f'run_{args["save_path"]}')

# Log only from rank 0
if rank == 0 and wandb:
    wandb.log({'loss': loss.item(), 'epoch': epoch})
```

### Checkpointing for Long Runs

```python
def save_checkpoint(model, optimizer, epoch, path):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.module.state_dict(),  # Unwrap DDP
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(checkpoint, path)

# In training loop:
if rank == 0 and (epoch + 1) % 5 == 0:
    save_checkpoint(model, optimizer, epoch, f'checkpoint_epoch_{epoch}.pt')
```

---

## Performance Comparison

### Expected Speedup

| Configuration | Time per Epoch | Speedup | Efficiency |
|--------------|----------------|---------|------------|
| 1 GPU, batch=32 | 100s | 1.0x | 100% |
| 2 GPUs, batch=32 each | 55s | 1.8x | 90% |
| 4 GPUs, batch=32 each | 29s | 3.4x | 85% |
| 8 GPUs, batch=32 each | 16s | 6.2x | 78% |

### Communication Overhead

The efficiency decreases with more GPUs due to:
1. **Gradient Synchronization**: All-reduce operations
2. **Load Balancing**: Uneven batch distribution
3. **Startup Overhead**: Process spawning

### When to Use DDP vs Single-GPU

**Use DDP when**:
- Training time > 1 hour on single GPU
- Multiple GPUs available
- Dataset size > 100k samples
- Need for experimentation speed

**Stick with Single-GPU when**:
- Training time < 30 minutes
- Only 1-2 GPUs available
- Small dataset
- Debugging or development

---

## Summary

**Key Points**:
1. DDP provides near-linear speedup for multi-GPU training
2. Minimal code changes required (use `train_distributed.py`)
3. Effective batch size = batch_size × n_gpus
4. Only rank 0 prints logs and saves models
5. DistributedSampler handles data partitioning automatically

**Next Steps**:
1. Run examples/train_ddp_example.py
2. Compare with single-GPU training
3. Tune batch size and learning rate
4. Monitor GPU utilization
5. Scale to more GPUs if needed

For more information, see:
- [PyTorch DDP Tutorial](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
- [NCCL Documentation](https://docs.nvidia.com/deeplearning/nccl/)
- GEARS Paper: [Original Publication]
