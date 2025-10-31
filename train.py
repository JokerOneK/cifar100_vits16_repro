import time, math, torch, timm, numpy as np
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100
from torch.optim import AdamW
from tqdm import tqdm
from transforms import make_transforms
from gpu_monitor import GPUMonitor
from sam_opt import SAM

def top1(logits, y):
    return (logits.argmax(1) == y).float().mean().item()

def make_loaders(bs=32, workers=None):
    train_tf, test_tf = make_transforms()
    import sys
    # Default workers: 0 on Windows (avoid spawn issues), else 4
    if workers is None:
        workers = 0 if sys.platform.startswith('win') else 4
    pin = torch.cuda.is_available()
    train_ds = CIFAR100(root='./data', train=True, download=True, transform=train_tf)
    test_ds  = CIFAR100(root='./data', train=False, download=True, transform=test_tf)
    train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=workers, pin_memory=pin)
    test_dl  = DataLoader(test_ds,  batch_size=bs, shuffle=False, num_workers=workers, pin_memory=pin)
    return train_dl, test_dl

def evaluate(model, loader, device):
    model.eval()
    n_correct=0; n_total=0
    with torch.no_grad():
        for x,y in loader:
            x,y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            logits = model(x)
            n_correct += (logits.argmax(1) == y).sum().item()
            n_total += y.numel()
    return n_correct / max(1, n_total)

def build_model(num_classes=100):
    m = timm.create_model('vit_small_patch16_224', pretrained=True)
    in_features = m.get_classifier().in_features
    m.reset_classifier(num_classes=num_classes)
    return m

def make_optimizer(model, lr, wd, use_sam=False):
    train_params = [p for p in model.parameters() if p.requires_grad]
    if use_sam:
        base_opt = AdamW
        opt = SAM(train_params, base_optimizer=base_opt, lr=lr, weight_decay=wd, rho=0.05)
    else:
        opt = AdamW(train_params, lr=lr, weight_decay=wd)
    return opt

def make_scheduler(optimizer, cfg, iters_per_epoch):
    """Build LR scheduler.
    Supports:
      - cosine with warmup (default): set cfg['scheduler']='cosine', optionally cfg['total_steps_override'], cfg['warmup_ratio']
      - StepLR in *iteration* units: set cfg['scheduler']='steplr', cfg['step_size'], cfg['gamma']
    """
    sched = cfg.get('scheduler', 'cosine')
    if sched == 'steplr':
        step_size = int(cfg.get('step_size', iters_per_epoch))  # in iterations (global)
        gamma = float(cfg.get('gamma', 0.1))
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    else:
        # cosine with warmup
        total_steps = cfg.get('total_steps_override')
        if total_steps is None:
            total_steps = cfg['epochs'] * iters_per_epoch
        warmup_ratio = float(cfg.get('warmup_ratio', 0.05))
        warmup_steps = max(1, int(warmup_ratio * total_steps))

        def lr_lambda(step):
            if step < warmup_steps:
                return (step + 1) / warmup_steps
            t = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.5 * (1.0 + math.cos(math.pi * t))
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
def train_one(model, train_dl, test_dl, cfg, seed=42, use_sam=False, gpu_index:int=0):
    torch.manual_seed(seed); np.random.seed(seed)
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Used device: {device}, cuda available: {torch.cuda.is_available()}')
    model = model.to(device)

    optimizer = make_optimizer(model, cfg['lr'], cfg['wd'], use_sam=use_sam)
    iters_per_epoch = (len(train_dl.dataset)//train_dl.batch_size + 1)
    scheduler = make_scheduler(optimizer, cfg, iters_per_epoch)
    criterion = nn.CrossEntropyLoss()

    start = time.perf_counter()
    mon = GPUMonitor(device_index=gpu_index, interval=0.5)
    mon.start()

    step = 0
    for epoch in range(cfg['epochs']):
        model.train()
        for x,y in tqdm(train_dl, desc=f"epoch {epoch+1}/{cfg['epochs']}"):
            x,y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            if isinstance(optimizer, SAM):
                # first step
                logits = model(x); loss = criterion(logits, y)
                loss.backward()
                optimizer.first_step(zero_grad=True)
                # second step
                logits = model(x); loss2 = criterion(logits, y)
                loss2.backward()
                optimizer.second_step(zero_grad=True)
            else:
                optimizer.zero_grad(set_to_none=True)
                logits = model(x); loss = criterion(logits, y)
                loss.backward()
                optimizer.step()
            scheduler.step()
            step += 1

    mon.stop()
    train_hours = (time.perf_counter() - start) / 3600.0
    avg_watts = mon.avg_power_W()
    energy_Wh = avg_watts * train_hours
    avg_mem_GB = mon.avg_mem_GB()
    peak_mem_GB = mon.peak_mem_GB()
    avg_gpu_util = mon.avg_gpu_util()
    avg_mem_util = mon.avg_mem_util()

    test_acc = evaluate(model, test_dl, device)
    return dict(
        test_acc=float(test_acc),
        hours=float(train_hours),
        avg_watts=float(avg_watts),
        energy_Wh=float(energy_Wh),
        avg_mem_GB=float(avg_mem_GB),
        peak_mem_GB=float(peak_mem_GB),
        avg_gpu_util_pct=float(avg_gpu_util),
        avg_mem_util_pct=float(avg_mem_util),
    )
