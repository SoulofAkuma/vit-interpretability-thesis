import os

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import random
import torch
import numpy as np
import torch.backends.cudnn


def activate_determinism_with_seed(seed: int):
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True, warn_only=True)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def deactive_determinism():
    torch.backends.cudnn.benchmark=True
    torch.backends.cudnn.deterministic=False
    torch.use_deterministic_algorithms(False)
    del os.environ["CUBLAS_WORKSPACE_CONFIG"]