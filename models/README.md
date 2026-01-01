# Models Directory

This directory will contain trained model artifacts after you run training.

**Files generated (not in git):**
- `fraud_model_v1.pt` - Trained PyTorch model
- `scaler.pkl` - Feature scaler
- `thresholds.json` - Decision thresholds
- `evaluation_plots.png` - Training plots

## How to Generate

### Option 1: Docker (Recommended)
```bash
docker-compose exec api python model/train.py
```

### Option 2: Local
```bash
PYTHONPATH=. python model/train.py
```

**Training takes ~3 minutes** on 50k transactions.

## Why Not in Git?

- 🚫 Large binary files (51KB model + assets)
- 🚫 Dataset-specific (your data ≠ my data)
- ✅ Easy to regenerate (one command)
- ✅ Keeps repository clean
