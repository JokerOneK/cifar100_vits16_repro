import os, pandas as pd
from train import build_model, make_loaders, train_one
from legacy.bitfit import prepare_bitfit
from lora_vit import prepare_lora

USE_SAM = bool(int(os.getenv("SWEEP_USE_SAM", "0")))

def run_mode(mode, seeds=(1,)):
    rows=[]
    for s in seeds:
        model = build_model()
        if mode=='ft':
            cfg=dict(lr=5e-4, wd=0.05, epochs=10, scheduler='cosine', total_steps_override=15640)
        elif mode=='bitfit':
            model = prepare_bitfit(model); cfg=dict(lr=2e-4, wd=0.0, epochs=10, total_steps_override=15640)
        elif mode=='lora8':
            model = prepare_lora(model, r=8, alpha=16); cfg=dict(lr=2e-4, wd=0.0, epochs=10, total_steps_override=15640)
        elif mode=='lora4':
            model = prepare_lora(model, r=4, alpha=8); cfg=dict(lr=2e-4, wd=0.0, epochs=10, total_steps_override=15640)
        else:
            raise ValueError(mode)
        train_dl, test_dl = make_loaders(32)
        res = train_one(model, train_dl, test_dl, cfg, seed=s, use_sam=USE_SAM)
        print(res)
        rows.append(dict(seed=s, **res))
    df = pd.DataFrame(rows)
    df['energy_Wh'] = df['avg_watts']*df['hours']
    out = f"results_{mode}.c   sv"
    df.to_csv(out, index=False)
    print(df.describe())
    print(f"Saved -> {out}")

if __name__ == "__main__":
    for m in ['lora8','lora4']:
        run_mode(m)
