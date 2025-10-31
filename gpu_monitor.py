import time, threading
import numpy as np
import pynvml

class GPUMonitor:
    """Monitor power, memory, and utilization via NVML in a background thread."""
    def __init__(self, device_index:int=0, interval:float=0.5):
        self.device_index = device_index
        self.interval = interval
        self.power_W = []
        self.mem_used_MB = []
        self.mem_total_MB = None
        self.util_gpu_pct = []
        self.util_mem_pct = []
        self._stop = threading.Event()
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
        self.mem_total_MB = info.total / (1024*1024)

    def _loop(self):
        while not self._stop.is_set():
            try:
                mw = pynvml.nvmlDeviceGetPowerUsage(self.handle)  # milliwatts
                self.power_W.append(mw/1000.0)
                mem = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
                used_mb = mem.used/(1024*1024)
                self.mem_used_MB.append(used_mb)
                util = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
                self.util_gpu_pct.append(util.gpu)
                # util.mem is memory controller utilization (%)
                self.util_mem_pct.append(util.memory)
            except Exception:
                # In case of transient NVML error, don't crash the training loop
                pass
            time.sleep(self.interval)

    def start(self):
        self._stop.clear()
        self.t = threading.Thread(target=self._loop, daemon=True)
        self.t.start()

    def stop(self):
        self._stop.set()
        if hasattr(self, 't'):
            self.t.join()

    # Aggregates
    def avg_power_W(self): return float(np.mean(self.power_W)) if self.power_W else 0.0
    def avg_mem_GB(self):  return float(np.mean(self.mem_used_MB)/1024.0) if self.mem_used_MB else 0.0
    def peak_mem_GB(self): return float(np.max(self.mem_used_MB)/1024.0) if self.mem_used_MB else 0.0
    def avg_gpu_util(self): return float(np.mean(self.util_gpu_pct)) if self.util_gpu_pct else 0.0
    def avg_mem_util(self): return float(np.mean(self.util_mem_pct)) if self.util_mem_pct else 0.0
