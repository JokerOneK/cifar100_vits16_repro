import pandas as pd
import numpy as np

def compute_sam(df: pd.DataFrame, alpha: int, beta: int, formula: str = "acc_over_energy",
                scale: float = 1.0, electricity_col: str = "energy_Wh"):
    """Compute SAM_{alpha,beta} for each row of df.

    formula options:
      - 'acc_over_energy' -> scale * (acc**alpha) / (energy_Wh**beta)
      - 'acc_over_power'  -> scale * (acc**alpha) / (avg_watts**beta)
      - 'acc_over_time'   -> scale * (acc**alpha) / (hours**beta)
      - 'acc_over_energy_mem' -> scale * (acc**alpha) / ((energy_Wh * (peak_mem_GB+1e-6))**beta)
      - 'acc_over_log_energy' -> **Exact formula from user**
            SAM = beta * (acc**alpha) / log10(electricity)
            where electricity == df[electricity_col] (default: 'energy_Wh')
    """
    acc = df['test_acc'].astype(float).clip(1e-12, None)

    if formula == "acc_over_log_energy":
        electricity = df[electricity_col].astype(float).clip(1e-12, None)
        return beta * (acc ** alpha) / np.log10(electricity)
    elif formula == "acc_over_energy":
        denom = df['energy_Wh'].astype(float).clip(1e-12, None)
        return scale * (acc ** alpha) / (denom ** beta)
    elif formula == "acc_over_power":
        denom = df['avg_watts'].astype(float).clip(1e-12, None)
        return scale * (acc ** alpha) / (denom ** beta)
    elif formula == "acc_over_time":
        denom = df['hours'].astype(float).clip(1e-12, None)
        return scale * (acc ** alpha) / (denom ** beta)
    elif formula == "acc_over_energy_mem":
        denom = (df['energy_Wh'].astype(float) * df['peak_mem_GB'].astype(float).clip(1e-12, None)).clip(1e-12, None)
        return scale * (acc ** alpha) / (denom ** beta)
    else:
        raise ValueError(f"Unknown formula: {formula}")

def auto_scale_for_baseline(value, target):
    if value == 0: 
        return 1.0
    return float(target) / float(value)
