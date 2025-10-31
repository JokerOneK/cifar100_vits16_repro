# CIFAR-100 ViT-S/16 Reproduction (FT, BitFit, LoRA)

Reproduce the table results for CIFAR-100 with ViT-S/16 (everything except Back Razor).
The scripts measure:
- Top-1 accuracy
- Total training time (hours)
- Average power draw (Watts) via NVML
- Total energy (Wh)
- **GPU memory:** average/peak used (GB) and average GPU/Memory utilization (%)

## 0) Environment
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

> Requires an NVIDIA GPU with NVML (standard for most discrete GPUs).

## 1) Run single modes
```bash
python run_ft.py      # Full fine-tune (10 epochs, bs=32)
python bitfit.py      # BitFit (bias-only + head)
python lora_vit.py    # LoRA r=8 a=16 then r=4 a=8 (two runs)
```

## 2) Multi-seed sweep and CSVs
```bash
python sweep.py
```
This produces CSV files like `results_ft.csv`/`results_bitfit.csv`/`results_lora8.csv`/`results_lora4.csv` with the metrics above.

## Notes
- Input resolution is 224x224 (ViT requires it).
- Uses ImageNet-pretrained weights from `timm`.
- For SAM, set `use_sam=True` in the `train_one` call or use the `SWEEP_USE_SAM=1` env var for `sweep.py`.
- To pin a GPU index other than 0, set env var `CUDA_VISIBLE_DEVICES` (and/or set `GPU_INDEX`).


## Step scheduling / exact step size
- By default we use **cosine with 5% warmup**. To force an exact global step count (e.g., `15640`), set in the config:
```python
cfg = dict(lr=5e-4, wd=0.05, epochs=10, scheduler='cosine', total_steps_override=15640)
```
- To use **StepLR** in *iteration units*, set:
```python
cfg = dict(lr=5e-4, wd=0.05, epochs=10, scheduler='steplr', step_size=15640, gamma=0.1)
```
