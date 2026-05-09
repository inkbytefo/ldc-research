import os
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Force CPU + small thread count for deterministic, fast tests.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
torch.set_num_threads(2)
