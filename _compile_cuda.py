import sys; sys.path.insert(0, '.')
import os; os.environ["TORCH_CUDA_ARCH_LIST"] = "8.6"

import mamba3_ssm.cuda_backend as cb
cb._CUDA_ATTEMPTED = False
cb._CUDA_MODULE = None

try:
    m = cb._compile_cuda_extension()
    print(f"CUDA module: {'LOADED' if m is not None else 'FAILED'}")
except Exception as e:
    print(f"Exception: {e}")
    import traceback; traceback.print_exc()
