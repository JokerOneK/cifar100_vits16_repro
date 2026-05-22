# VLM & Contrastive Learning Baselines — Results Summary

**Date collected:** 2026-04-06
**Hardware:** NVIDIA GeForce RTX 2060 SUPER, 8.0 GB VRAM
**Framework:** torch=2.4.0+cu118, torchvision=0.19.0+cu118, CUDA=11.8
**Datasets evaluated:** CIFAR-100 (100 classes, 50k train / 10k test) and DTD — Describable Textures Dataset (47 classes, 1880 train / 1880 test)

---

## Models Overview

| Model | Type | Params | Approach | HF Checkpoint |
|-------|------|--------|----------|---------------|
| DINOv2-Large | Self-supervised ViT | 304.4M | Feature extraction → k-NN + Linear probe | facebook/dinov2-large |
| OpenCLIP ViT-L/14 | Contrastive (CLIP) | 427.6M | Zero-shot classification | ViT-L-14 / datacomp_xl_s13b_b90k |
| SigLIP-Large | Contrastive (SigLIP) | 652.5M | Zero-shot classification | google/siglip-large-patch16-384 |
| SmolVLM-256M | VLM (VQA) | 256.5M | Zero-shot VQA | HuggingFaceTB/SmolVLM-256M-Instruct |
| MobileVLM V2-1.7B | VLM (VQA) | 1674.1M | Zero-shot VQA | mtgv/MobileVLM_V2-1.7B |
| PaLiGemma-3B | VLM (VQA) | 2923.5M | Zero-shot VQA | google/paligemma-3b-mix-224 |

---

## Main Results: CIFAR-100

| Model | Eval Method | Top-1 Acc (%) | Eval Time (s) | Avg Power (W) | Energy (J) | Energy (Wh) |
|-------|-------------|--------------|--------------|---------------|-----------|-------------|
| DINOv2-Large | k-NN (k=20) | **91.17** | 860.7 | 149.5 | 127,632 | 35.45 |
| DINOv2-Large | Linear probe (10 ep) | **91.71** | 865.8 | 147.6 | 127,792* | 35.50 |
| OpenCLIP ViT-L/14 | Zero-shot | 87.26 | 78.2 | 167.8 | 11,500 | 3.19 |
| SigLIP-Large | Zero-shot | 80.77 | 219.1 | 164.2 | 35,698 | 9.92 |
| MobileVLM V2-1.7B | Zero-shot VQA | 27.40 | 2741.1 | 73.7 | 200,623 | 55.73 |
| PaLiGemma-3B | Zero-shot VQA | 22.85 | 1952.1 | 83.8 | 162,130 | 45.04 |
| SmolVLM-256M | Zero-shot VQA | 13.52 | 5908.6 | N/A† | N/A† | N/A† |

\* DINOv2 linear probe energy is cumulative (includes feature extraction).
† SmolVLM energy monitoring failed (pynvml not available in that venv).

## Main Results: DTD (Describable Textures Dataset)

| Model | Eval Method | Top-1 Acc (%) | Eval Time (s) | Avg Power (W) | Energy (J) | Energy (Wh) |
|-------|-------------|--------------|--------------|---------------|-----------|-------------|
| DINOv2-Large | k-NN (k=20) | **73.99** | 76.3 | 126.5 | 7,481 | 2.08 |
| DINOv2-Large | Linear probe (10 ep) | 72.82 | 76.5 | 97.9 | 7,490* | 2.08 |
| OpenCLIP ViT-L/14 | Zero-shot | **67.77** | 25.8 | 122.1 | 1,571 | 0.44 |
| SigLIP-Large | Zero-shot | 70.96 | 52.7 | 130.2 | 5,709 | 1.59 |
| MobileVLM V2-1.7B | Zero-shot VQA | 3.99 | 874.3 | 66.5 | 57,910 | 16.09 |
| PaLiGemma-3B | Zero-shot VQA | 21.54 | 331.8 | 83.6 | 27,403 | 7.61 |
| SmolVLM-256M | Zero-shot VQA | 7.82 | 1172.8 | N/A† | N/A† | N/A† |

---

## DINOv2 Linear Probe — Training Curves

### CIFAR-100 (10 epochs, lr=0.001)

| Epoch | Train Loss | Test Acc (%) |
|-------|-----------|-------------|
| 1 | 3.7850 | 89.31 |
| 2 | 2.3572 | 89.83 |
| 3 | 1.4849 | 90.26 |
| 4 | 1.0431 | 90.58 |
| 5 | 0.8080 | 90.75 |
| 6 | 0.6679 | 91.02 |
| 7 | 0.5769 | 91.18 |
| 8 | 0.5126 | 91.45 |
| 9 | 0.4641 | 91.56 |
| 10 | 0.4267 | **91.71** |

### DTD (10 epochs, lr=0.001)

| Epoch | Train Loss | Test Acc (%) |
|-------|-----------|-------------|
| 1 | 3.8170 | 50.00 |
| 2 | 3.7391 | 65.11 |
| 3 | 3.6652 | 68.94 |
| 4 | 3.5921 | 70.48 |
| 5 | 3.5202 | 71.22 |
| 6 | 3.4499 | 71.54 |
| 7 | 3.3790 | 72.02 |
| 8 | 3.3023 | 72.45 |
| 9 | 3.2363 | 72.66 |
| 10 | 3.1729 | **72.82** |

---

## Raw epoch_metrics.csv Data

### DINOv2-Large — CIFAR-100
```
epoch,method,train_time_s,eval_time_s,total_time_s,train_energy_j,eval_energy_j,total_energy_j,avg_power_w,test_acc_pct,SAM_a1_b1,SAM_a2_b2,SAM_a3_b3,SAM_a4_b4,SAM_a5_b5
0,knn,0.0,860.684,860.684,0.0,127631.526,127631.526,149.455,91.17,0.178556,0.031882,0.005693,0.001016,0.000181
10,linear,2.715,0.142,865.757,127780.100,11.913,127792.014,147.607,91.71,0.179594,0.032254,0.005793,0.001040,0.000187
```

### DINOv2-Large — DTD
```
epoch,method,train_time_s,eval_time_s,total_time_s,train_energy_j,eval_energy_j,total_energy_j,avg_power_w,test_acc_pct,SAM_a1_b1,SAM_a2_b2,SAM_a3_b3,SAM_a4_b4,SAM_a5_b5
0,knn,0.0,76.319,76.319,0.0,7480.705,7480.705,126.484,73.99,0.190992,0.036478,0.006967,0.001331,0.000254
10,linear,0.071,0.004,76.517,7489.382,0.451,7489.833,97.885,72.82,0.187946,0.035324,0.006639,0.001248,0.000235
```

### MobileVLM V2-1.7B — CIFAR-100
```
epoch,train_time_s,eval_time_s,total_time_s,train_energy_j,eval_energy_j,total_energy_j,avg_power_w,test_acc_pct,SAM_a1_b1,SAM_a2_b2,SAM_a3_b3,SAM_a4_b4,SAM_a5_b5
0,0.0,2741.095,2741.095,0.0,200623.312,200623.312,73.709,27.40,0.051675,0.002670,0.000138,0.000007,0.000000
```

### MobileVLM V2-1.7B — DTD
```
epoch,train_time_s,eval_time_s,total_time_s,train_energy_j,eval_energy_j,total_energy_j,avg_power_w,test_acc_pct,SAM_a1_b1,SAM_a2_b2,SAM_a3_b3,SAM_a4_b4,SAM_a5_b5
0,0.0,874.276,874.276,0.0,57910.424,57910.424,66.490,3.99,0.008376,0.000070,0.000001,0.000000,0.000000
```

### OpenCLIP ViT-L/14 — CIFAR-100
```
epoch,train_time_s,eval_time_s,total_time_s,train_energy_j,eval_energy_j,total_energy_j,avg_power_w,test_acc_pct,SAM_a1_b1,SAM_a2_b2,SAM_a3_b3,SAM_a4_b4,SAM_a5_b5
0,0.0,78.205,78.205,0.0,11500.232,11500.232,167.836,87.26,0.214889,0.046177,0.009923,0.002132,0.000458
```

### OpenCLIP ViT-L/14 — DTD
```
epoch,train_time_s,eval_time_s,total_time_s,train_energy_j,eval_energy_j,total_energy_j,avg_power_w,test_acc_pct,SAM_a1_b1,SAM_a2_b2,SAM_a3_b3,SAM_a4_b4,SAM_a5_b5
0,0.0,25.848,25.848,0.0,1571.198,1571.198,122.088,67.77,0.212018,0.044952,0.009531,0.002021,0.000428
```

### PaLiGemma-3B — CIFAR-100
```
epoch,train_time_s,eval_time_s,total_time_s,train_energy_j,eval_energy_j,total_energy_j,avg_power_w,test_acc_pct,SAM_a1_b1,SAM_a2_b2,SAM_a3_b3,SAM_a4_b4,SAM_a5_b5
0,0.0,1952.066,1952.066,0.0,162130.405,162130.405,83.806,22.85,0.043859,0.001924,0.000084,0.000004,0.000000
```

### PaLiGemma-3B — DTD
```
epoch,train_time_s,eval_time_s,total_time_s,train_energy_j,eval_energy_j,total_energy_j,avg_power_w,test_acc_pct,SAM_a1_b1,SAM_a2_b2,SAM_a3_b3,SAM_a4_b4,SAM_a5_b5
0,0.0,331.793,331.793,0.0,27402.805,27402.805,83.603,21.54,0.048543,0.002356,0.000114,0.000006,0.000000
```

### SigLIP-Large — CIFAR-100
```
epoch,train_time_s,eval_time_s,total_time_s,train_energy_j,eval_energy_j,total_energy_j,avg_power_w,test_acc_pct,SAM_a1_b1,SAM_a2_b2,SAM_a3_b3,SAM_a4_b4,SAM_a5_b5
0,0.0,219.079,219.079,0.0,35697.650,35697.650,164.237,80.77,0.177414,0.031476,0.005584,0.000991,0.000176
```

### SigLIP-Large — DTD
```
epoch,train_time_s,eval_time_s,total_time_s,train_energy_j,eval_energy_j,total_energy_j,avg_power_w,test_acc_pct,SAM_a1_b1,SAM_a2_b2,SAM_a3_b3,SAM_a4_b4,SAM_a5_b5
0,0.0,52.674,52.674,0.0,5708.580,5708.580,130.177,70.96,0.188891,0.035680,0.006740,0.001273,0.000240
```

### SmolVLM-256M — CIFAR-100
```
epoch,train_time_s,eval_time_s,total_time_s,train_energy_j,eval_energy_j,total_energy_j,avg_power_w,test_acc_pct,SAM_a1_b1,SAM_a2_b2,SAM_a3_b3,SAM_a4_b4,SAM_a5_b5
0,0.0,5908.579,5908.579,0.0,0.0,0.0,0.0,13.52,nan,nan,nan,nan,nan
```

### SmolVLM-256M — DTD
```
epoch,train_time_s,eval_time_s,total_time_s,train_energy_j,eval_energy_j,total_energy_j,avg_power_w,test_acc_pct,SAM_a1_b1,SAM_a2_b2,SAM_a3_b3,SAM_a4_b4,SAM_a5_b5
0,0.0,1172.822,1172.822,0.0,0.0,0.0,0.0,7.82,nan,nan,nan,nan,nan
```

---

## Notes on Evaluation Methods

### Contrastive Models (OpenCLIP, SigLIP)
- **Zero-shot classification** via cosine similarity between image embeddings and class-name text embeddings.
- No training, no gradient updates — pure inference.
- OpenCLIP uses the CLIP-style softmax over dot products; SigLIP uses sigmoid-based loss (trained differently but inference is the same).

### Self-supervised Model (DINOv2)
- **Feature extraction:** All train and test images are passed through the frozen encoder to obtain feature vectors (dim=1024 for Large).
- **k-NN (k=20):** Nearest-neighbor search over training features — no parameters, no training.
- **Linear probe:** A single linear layer trained for 10 epochs on top of frozen features (lr=0.001). Very few parameters — only the classification head.
- Note: On DTD, k-NN (73.99%) outperforms linear probe (72.82%), possibly because 10 epochs was not enough for convergence on the small DTD training set.

### VLM Models (MobileVLM, PaLiGemma, SmolVLM)
- **Zero-shot VQA:** Each image is passed with a text prompt (e.g., "What is in this image? Answer with one of: [class_names]..."). The model generates a text answer which is matched against class names.
- These models were **not designed for classification** — they are generative chat/VQA models. The low accuracy reflects the mismatch between their training objective and the classification task.
- VQA evaluation is extremely slow due to autoregressive token generation per image.
- SmolVLM energy monitoring failed (pynvml not compatible with the venv used).

---

## Key Observations for Thesis

1. **Best accuracy (CIFAR-100):** DINOv2-Large with linear probe achieves 91.71%, highest among all models — benefiting from both powerful features and a supervised head.

2. **Best zero-shot accuracy (CIFAR-100):** OpenCLIP at 87.26%, ahead of SigLIP at 80.77%. Both use contrastive pretraining with natural language supervision.

3. **VLMs perform poorly on classification:** MobileVLM (27.40%), PaLiGemma (22.85%), SmolVLM (13.52%) — far below contrastive models. VLMs are not designed for this task.

4. **Energy efficiency:** OpenCLIP is remarkably efficient — 11,500J (3.19 Wh) for CIFAR-100, vs DINOv2's 255,424J (70.95 Wh). VLMs consume 57K–201K J just for CIFAR-100 inference.

5. **DTD results differ from CIFAR-100:** SigLIP (70.96%) outperforms OpenCLIP (67.77%) on texture data despite lower CIFAR-100 accuracy. DINOv2 k-NN (73.99%) remains the best.

6. **DINOv2 feature quality:** k-NN reaching 91.17% on CIFAR-100 with no training at all demonstrates the exceptional quality of DINOv2 representations.

7. **SAM scores:** OpenCLIP has the highest SAM_a1_b1 (0.2149 CIFAR / 0.2120 DTD), indicating the sharpest accuracy-efficiency boundary among contrastive models. DINOv2 and SigLIP are comparable (~0.177–0.191).
