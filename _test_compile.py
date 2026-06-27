import sys; sys.path.insert(0, '.')
import os
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.6")  # RTX 4060 is Ada Lovelace, sm_86

from torch.utils.cpp_extension import load_inline

cuda_src = """
__global__ void test_kernel(float* x, float val, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) x[i] = val;
}
extern "C" void launch_test(float* x, float val, int N) {
    test_kernel<<<(N+255)/256, 256>>>(x, val, N);
}
"""

cpp_src = """
#include <torch/extension.h>
extern "C" void launch_test(float* x, float val, int N);
torch::Tensor test_fn(torch::Tensor x, float val) {
    launch_test(x.data_ptr<float>(), val, x.numel());
    return x;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("test_fn", &test_fn, "Test CUDA kernel");
}
"""

print("Compiling test CUDA extension...", flush=True)
mod = load_inline(name="test_cuda_ext", cpp_sources=cpp_src, cuda_sources=cuda_src, functions=["test_fn"], verbose=True, extra_cuda_cflags=["-allow-unsupported-compiler"])
print(f"SUCCESS: {mod}", flush=True)

# Test it
import torch
x = torch.zeros(10, device="cuda")
mod.test_fn(x, 42.0)
print(f"Result: {x}", flush=True)
