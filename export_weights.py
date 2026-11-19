"""把 PyTorch 存檔轉成板子用的 numpy 權重：python export_weights.py models/a100_it1300.pt"""
import sys

import numpy as np
import torch

src = sys.argv[1] if len(sys.argv) > 1 else "models/a100_it1300.pt"
dst = sys.argv[2] if len(sys.argv) > 2 else "board-app/python/weights.npz"
ck = torch.load(src, map_location="cpu", weights_only=True)
arrays = {k: v.float().numpy() for k, v in ck["model"].items()}
arrays["res_blocks"] = np.array(ck["cfg"]["res_blocks"])
for k in ("height", "width", "start_len", "max_steps", "max_health", "n_food", "channels"):
    arrays[f"cfg_{k}"] = np.array(ck["cfg"][k])
np.savez(dst, **arrays)
n = sum(v.numel() for v in ck["model"].values())
print(f"wrote {dst}：{n:,} 個參數")
