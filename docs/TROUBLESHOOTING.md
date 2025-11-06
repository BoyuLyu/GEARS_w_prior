# Multi-GPU Training Troubleshooting Guide

## Issue: NCCL Cleanup Warning

### Problem
```
Warning: WARNING: destroy_process_group() was not called before program exit
```

### Cause
- Running `mp.spawn()` in Jupyter notebooks causes process cleanup issues
- NCCL backend doesn't cleanup properly in interactive environments
- Process termination via SIGTERM before proper cleanup

### Solutions

#### ✅ Solution 1: Run from Terminal (Recommended)
```bash
cd /work/boyu/ai_ml_projects/vcc/gears/run_gears
python train_multi_gpu.py
```

**Why this works:**
- Proper process lifecycle management
- Clean exit and cleanup
- No Jupyter kernel interference

#### ✅ Solution 2: Use Subprocess in Notebook
```python
import subprocess
import sys

result = subprocess.run(
    [sys.executable, 'train_multi_gpu.py'],
    capture_output=True,
    text=True
)
print(result.stdout)
```

**Why this works:**
- Isolates multiprocessing from notebook kernel
- Separate Python process with proper cleanup
- Captures output cleanly

#### ❌ What Doesn't Work
```python
# Don't do this in notebooks!
from gears.train_distributed import train_multi_gpu
train_multi_gpu(...)  # Will cause cleanup warnings
```

---

## Other Common Issues

### Issue: "Port already in use"

**Error:**
```
RuntimeError: Address already in use
```

**Solution:**
```python
# In train_distributed.py, change the port:
os.environ['MASTER_PORT'] = '12356'  # Use different port
```

Or kill existing processes:
```bash
pkill -f train_distributed
```

---

### Issue: "CUDA out of memory"

**Error:**
```
RuntimeError: CUDA out of memory
```

**Solutions:**

1. **Reduce batch size per GPU:**
   ```python
   args = {
       'batch_size': 16,  # Reduce from 32
       ...
   }
   ```

2. **Use gradient accumulation:**
   ```python
   # Simulate larger batch with multiple smaller batches
   accumulation_steps = 2
   effective_batch_size = batch_size * n_gpus * accumulation_steps
   ```

3. **Reduce model size:**
   ```python
   'model_config': {
       'hidden_size': 32,  # Reduce from 64
       ...
   }
   ```

---

### Issue: "No module named 'gears'"

**Error:**
```
ModuleNotFoundError: No module named 'gears'
```

**Solution:**
```python
import sys
sys.path.insert(0, '../GEARS_w_prior')
```

Or install in development mode:
```bash
cd /work/boyu/ai_ml_projects/vcc/gears/GEARS_w_prior
pip install -e .
```

---

### Issue: Hung Training Process

**Symptoms:**
- Training starts but never progresses
- All GPUs show 0% utilization
- No error messages

**Solutions:**

1. **Check if data is loaded:**
   ```python
   # Verify data exists
   ls -lh ./data/vcc_allgene/perturb_processed.h5ad
   ```

2. **Check GPU visibility:**
   ```bash
   nvidia-smi
   CUDA_VISIBLE_DEVICES=0,1 python train_multi_gpu.py
   ```

3. **Enable debug mode:**
   ```python
   # Add to train_distributed.py
   import logging
   logging.basicConfig(level=logging.DEBUG)
   ```

---

### Issue: Different Results Across Runs

**Cause:**
- Random initialization not synced across GPUs
- Different random seeds on different processes

**Solution:**
```python
# Ensure same seed across all processes
split_config = {
    'seed': 42,  # Fixed seed
    ...
}

# In training code
import torch
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
```

---

### Issue: Uneven GPU Utilization

**Symptoms:**
- GPU 0: 95% utilization
- GPU 1: 30% utilization

**Causes & Solutions:**

1. **Imbalanced data split:**
   ```python
   # DistributedSampler should handle this automatically
   # But check dataset sizes
   print(f"Rank {rank}: {len(train_loader.dataset)} samples")
   ```

2. **Different batch processing times:**
   - DDP synchronizes after each batch
   - Slower GPU becomes bottleneck
   - Check for hardware issues

3. **Model on wrong device:**
   ```python
   # Verify model placement
   print(f"Model on device: {next(model.parameters()).device}")
   ```

---

## Diagnostic Commands

### Check NCCL Version
```bash
python -c "import torch; print(torch.cuda.nccl.version())"
```

### Check Process Status
```bash
ps aux | grep python | grep train_multi_gpu
```

### Monitor GPU Memory Over Time
```bash
nvidia-smi --query-gpu=timestamp,name,memory.used,memory.total \
           --format=csv -l 1 | tee gpu_memory.log
```

### Check Distributed Backend
```python
import torch.distributed as dist
print(f"Available backends: {dist.Backend.NCCL}")
print(f"NCCL available: {torch.cuda.nccl.is_available([0,1])}")
```

---

## Best Practices

### ✅ DO:
- Run multi-GPU training from terminal/script
- Use subprocess when calling from notebooks
- Monitor GPU usage during training
- Save checkpoints regularly
- Use same random seed across processes
- Verify data loading before training

### ❌ DON'T:
- Run `mp.spawn()` directly in Jupyter
- Forget to sync processes before saving
- Mix different CUDA versions
- Ignore cleanup warnings (they matter!)
- Assume linear speedup (expect 80-90% efficiency)

---

## Quick Verification Checklist

Before starting multi-GPU training:

- [ ] Data processed and available
- [ ] GPUs visible: `nvidia-smi`
- [ ] Correct Python environment activated
- [ ] NCCL backend available
- [ ] Ports not in use (12355, 12356, etc.)
- [ ] Enough GPU memory for batch size
- [ ] Running from terminal (not notebook)
- [ ] Correct dataset name specified

---

## Getting Help

If issues persist:

1. Check full error traceback
2. Verify single-GPU training works first
3. Test with smaller batch size (e.g., 8)
4. Try with 2 GPUs before scaling to 4+
5. Check CUDA and PyTorch versions match
6. Review `docs/MULTI_GPU_TRAINING.md` for detailed info

## Still Having Issues?

Post the following information:
- Full error message and traceback
- Output of `nvidia-smi`
- PyTorch version: `python -c "import torch; print(torch.__version__)"`
- CUDA version: `nvcc --version`
- Training command used
- Relevant configuration (batch_size, n_gpus, etc.)
