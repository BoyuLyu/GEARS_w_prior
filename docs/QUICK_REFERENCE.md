# Multi-GPU Training Quick Reference

## Quick Commands

### Check Available GPUs
```bash
python -c "import torch; print(f'GPUs: {torch.cuda.device_count()}')"
nvidia-smi
```

### Run Multi-GPU Training
```bash
cd /work/boyu/ai_ml_projects/vcc/gears/GEARS_w_prior/examples
python train_ddp_example.py
```

### Monitor GPU Usage
```bash
watch -n 1 nvidia-smi
```

---

## Code Templates

### Template 1: Basic Multi-GPU Training

```python
from gears.train_distributed import train_multi_gpu

split_config = {
    'split': 'simulation',
    'seed': 1,
    'train_gene_set_size': 0.75
}

args = {
    'model_config': {'hidden_size': 64},
    'batch_size': 32,
    'epochs': 20,
    'lr': 1e-3,
    'save_path': './models/my_model'
}

train_multi_gpu(
    pert_data_path='./data',
    dataset_name='vcc',
    split_config=split_config,
    args=args,
    n_gpus=4
)
```

### Template 2: Custom GPU Selection

```python
# Use specific GPUs (e.g., 0, 1, 4, 5)
train_multi_gpu(
    ...,
    n_gpus=4,
    gpu_ids=[0, 1, 4, 5]
)
```

### Template 3: Load and Evaluate Model

```python
from gears import PertData, GEARS

# Load data
pert_data = PertData(data_path='./data')
pert_data.load(data_path='./data/vcc')
pert_data.prepare_split(split='simulation', seed=1)
pert_data.get_dataloader(batch_size=128)

# Load trained model
gears_model = GEARS(pert_data, device='cuda:0')
gears_model.load_pretrained('./models/my_model')

# Predict
predictions = gears_model.predict([['GENE1'], ['GENE2']])
```

---

## File Structure After Setup

```
GEARS_w_prior/
├── gears/
│   ├── __init__.py
│   ├── gears.py                 # ✅ Modified for DDP
│   ├── pertdata.py              # ✅ Modified for DDP
│   ├── train_distributed.py    # ✅ NEW - Main DDP script
│   ├── model.py
│   ├── inference.py
│   └── utils.py
├── examples/
│   ├── train_ddp_example.py    # ✅ NEW - Basic example
│   └── train_comparison.py     # ✅ NEW - Single vs Multi GPU
├── docs/
│   └── MULTI_GPU_TRAINING.md   # ✅ NEW - Full documentation
└── data/
    └── vcc/
        ├── perturb_processed.h5ad
        └── data_pyg/
            └── cell_graphs.pkl
```

---

## Common Parameters

### split_config
- `split`: 'simulation', 'combo_seen0', 'combo_seen1', 'combo_seen2', 'single'
- `seed`: Random seed (default: 1)
- `train_gene_set_size`: Fraction of genes for training (default: 0.75)

### model_config
- `hidden_size`: Hidden dimension (default: 64)
- `num_go_gnn_layers`: GO graph layers (default: 1)
- `num_gene_gnn_layers`: Co-expression layers (default: 1)
- `decoder_hidden_size`: Decoder hidden size (default: 16)
- `uncertainty`: Enable uncertainty mode (default: False)

### Training args
- `batch_size`: Batch size per GPU (default: 32)
- `test_batch_size`: Test batch size per GPU (default: 128)
- `epochs`: Number of epochs (default: 20)
- `lr`: Learning rate (default: 1e-3)
- `weight_decay`: Weight decay (default: 5e-4)

---

## Troubleshooting Checklist

- [ ] Check GPU availability: `nvidia-smi`
- [ ] Verify CUDA installation: `python -c "import torch; print(torch.cuda.is_available())"`
- [ ] Ensure data is processed: Check for `data/vcc/perturb_processed.h5ad`
- [ ] Check port conflicts: Try different port in `setup_ddp()`
- [ ] Reduce batch size if OOM error
- [ ] Kill hanging processes: `pkill -f train_distributed`
- [ ] Check logs: Only rank 0 prints output

---

## Performance Tips

1. **Batch Size**: Start with single-GPU batch size, scale up if memory allows
2. **Learning Rate**: Scale linearly with number of GPUs (lr = base_lr * n_gpus)
3. **Workers**: Use 4-8 workers per GPU for data loading
4. **Mixed Precision**: Add `autocast()` for 2x speedup
5. **Monitor**: Watch GPU utilization with `nvidia-smi`

---

## Expected Speedup

| GPUs | Speedup | Notes |
|------|---------|-------|
| 1    | 1.0x    | Baseline |
| 2    | ~1.8x   | Good for development |
| 4    | ~3.4x   | Recommended for production |
| 8    | ~6.2x   | Best for large datasets |

---

## Helpful Commands

```bash
# Kill all Python processes
pkill python

# Check GPU memory
nvidia-smi --query-gpu=memory.used,memory.total --format=csv

# Monitor specific GPUs
nvidia-smi -i 0,1,2,3 -l 1

# Check CUDA version
nvcc --version

# List PyTorch CUDA version
python -c "import torch; print(torch.version.cuda)"
```

---

## Contact & Support

For issues or questions:
1. Check `docs/MULTI_GPU_TRAINING.md` for detailed documentation
2. Review `examples/train_comparison.py` for side-by-side comparison
3. See PyTorch DDP documentation: https://pytorch.org/docs/stable/distributed.html
