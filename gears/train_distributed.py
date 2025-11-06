import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
import os

def setup_ddp(rank, world_size):
    """Initialize distributed training"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup_ddp():
    """Clean up distributed training"""
    try:
        if dist.is_initialized():
            dist.destroy_process_group()
    except Exception as e:
        print(f"Warning during cleanup: {e}")

def train_ddp_worker(rank, world_size, pert_data_path, dataset_name, split_config, args):
    """
    Training worker for each GPU
    
    Parameters
    ----------
    rank: int
        GPU rank
    world_size: int
        Total number of GPUs
    pert_data_path: str
        Path to the perturbation data
    dataset_name: str
        Name of the dataset
    split_config: dict
        Configuration for data splitting
    args: dict
        Training arguments including:
        - model_config: dict with model initialization parameters
        - batch_size: int
        - epochs: int
        - lr: float
        - weight_decay: float
        - save_path: str
    """
    try:
        # Setup distributed training
        setup_ddp(rank, world_size)
        
        # Print only from rank 0
        if rank == 0:
            print(f"Starting distributed training on {world_size} GPUs")
            print(f"Rank {rank}: Primary process")
        
        # Each worker loads data independently
        from gears import PertData, GEARS
        
        # Load data on each process
        pert_data = PertData(data_path=pert_data_path)
        pert_data.load(data_path=f'{pert_data_path}/{dataset_name}')
        
        # Prepare split (use same seed for consistency across processes)
        pert_data.prepare_split(
            split=split_config.get('split', 'simulation'),
            seed=split_config.get('seed', 1),
            train_gene_set_size=split_config.get('train_gene_set_size', 0.75)
        )
        
        # Get distributed dataloaders
        pert_data.get_dataloader_distributed(
            batch_size=args['batch_size'],
            test_batch_size=args.get('test_batch_size', args['batch_size']),
            world_size=world_size,
            rank=rank,
            num_workers=args.get('num_workers', 4)
        )
        
        # Create model on this GPU
        gears_model = GEARS(pert_data, device=f'cuda:{rank}')
        gears_model.model_initialize(**args['model_config'])
        
        # Wrap model with DDP
        gears_model.model = DDP(
            gears_model.model,
            device_ids=[rank],
            output_device=rank,
            find_unused_parameters=True
        )
        
        # Store DDP info for later use
        gears_model.is_distributed = True
        gears_model.rank = rank
        gears_model.world_size = world_size
        
        # Train
        if rank == 0:
            print("Starting training...")
        
        gears_model.train(
            epochs=args['epochs'],
            lr=args.get('lr', 1e-3),
            weight_decay=args.get('weight_decay', 5e-4)
        )
        
        # Save model (only rank 0)
        if rank == 0:
            print("Saving model...")
            gears_model.save_model(args['save_path'])
            print(f"Model saved to {args['save_path']}")
        
        # Synchronize all processes before cleanup
        if dist.is_initialized():
            dist.barrier()
    
    except Exception as e:
        if rank == 0:
            print(f"Error in training worker (rank {rank}): {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Always cleanup, even if there was an error
        cleanup_ddp()

def train_multi_gpu(pert_data_path, dataset_name, split_config, args, n_gpus=None, gpu_ids=None):
    """
    Launch multi-GPU training using DistributedDataParallel
    
    Parameters
    ----------
    pert_data_path: str
        Path to the perturbation data directory
    dataset_name: str
        Name of the dataset (e.g., 'vcc', 'norman')
    split_config: dict
        Configuration for data splitting:
        - split: str (e.g., 'simulation', 'combo_seen0')
        - seed: int
        - train_gene_set_size: float
    args: dict
        Training configuration:
        - model_config: dict with hidden_size, num_go_gnn_layers, etc.
        - batch_size: int
        - test_batch_size: int (optional)
        - epochs: int
        - lr: float
        - weight_decay: float
        - num_workers: int (optional, default=4) - Number of CPU workers per GPU
        - save_path: str
    n_gpus: int
        Number of GPUs to use. If None, uses all available
    gpu_ids: list
        Specific GPU IDs to use (e.g., [0, 1, 2]). If None, uses first n_gpus
        
    Example
    -------
    >>> from gears.train_distributed import train_multi_gpu
    >>> 
    >>> split_config = {
    >>>     'split': 'simulation',
    >>>     'seed': 1,
    >>>     'train_gene_set_size': 0.75
    >>> }
    >>> 
    >>> args = {
    >>>     'model_config': {
    >>>         'hidden_size': 64,
    >>>         'num_go_gnn_layers': 1,
    >>>         'num_gene_gnn_layers': 1
    >>>     },
    >>>     'batch_size': 32,
    >>>     'epochs': 20,
    >>>     'lr': 1e-3,
    >>>     'weight_decay': 5e-4,
    >>>     'save_path': './models/gears_ddp'
    >>> }
    >>> 
    >>> train_multi_gpu(
    >>>     pert_data_path='./data',
    >>>     dataset_name='vcc',
    >>>     split_config=split_config,
    >>>     args=args,
    >>>     n_gpus=4
    >>> )
    """
    # Determine number of GPUs
    available_gpus = torch.cuda.device_count()
    
    if available_gpus == 0:
        raise RuntimeError("No CUDA GPUs available for distributed training")
    
    if n_gpus is None:
        n_gpus = available_gpus
    
    if n_gpus > available_gpus:
        print(f"Warning: Requested {n_gpus} GPUs but only {available_gpus} available. Using {available_gpus}.")
        n_gpus = available_gpus
    
    # Set up GPU visibility if specific IDs provided
    if gpu_ids is not None:
        if len(gpu_ids) != n_gpus:
            raise ValueError(f"Length of gpu_ids ({len(gpu_ids)}) must match n_gpus ({n_gpus})")
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))
        print(f"Training on GPUs: {gpu_ids}")
    else:
        print(f"Training on {n_gpus} GPUs: {list(range(n_gpus))}")
    
    # Set multiprocessing start method
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    
    # Launch distributed training
    print(f"\n{'='*60}")
    print(f"Launching Distributed Training")
    print(f"Number of GPUs: {n_gpus}")
    print(f"Batch size per GPU: {args['batch_size']}")
    print(f"Effective batch size: {args['batch_size'] * n_gpus}")
    print(f"{'='*60}\n")
    
    mp.spawn(
        train_ddp_worker,
        args=(n_gpus, pert_data_path, dataset_name, split_config, args),
        nprocs=n_gpus,
        join=True
    )
    
    print(f"\n{'='*60}")
    print("Distributed Training Completed!")
    print(f"{'='*60}\n")