import torch
print("torch:", torch.__version__)
print("CUDA build:", torch.version.cuda)      # должно быть что-то вроде '12.4', а не None
print("CUDA available:", torch.cuda.is_available())
print("device_count:", torch.cuda.device_count())