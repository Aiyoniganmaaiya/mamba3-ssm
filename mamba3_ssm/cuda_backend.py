"""
CUDA-accelerated SSM scans for Mamba-3.

Provides fused CUDA kernels for the sequential SSM scan,
replacing the Python for-loop with a GPU-parallelized implementation.

SISO: fully fused (B×H blocks, P threads per block, all D in registers)
MIMO: split as x_mixed (PyTorch einsum) → state kernel → y (PyTorch einsum)
"""

import torch
import einops
from typing import Optional, Tuple

_CUDA_MODULE = None
_CUDA_ATTEMPTED = False

def _compile_cuda_extension():
    global _CUDA_MODULE, _CUDA_ATTEMPTED
    if _CUDA_ATTEMPTED:
        return _CUDA_MODULE
    _CUDA_ATTEMPTED = True

    cuda_source = r"""
#include <cuda_runtime.h>

// ================================================================
// SISO Scan Kernel
// Grid: (B, H)      — one block per (batch, head)
// Block: (P, 1, 1)  — one thread per head dimension
// Each thread holds h[D] and Bx_prev[D] in registers,
// shares B_t / C_t via shared memory each timestep.
// ================================================================

template<int D>
__global__ void siso_scan_kernel(
    const float* x,
    const float* B_proj,
    const float* C_proj,
    const float* decay,
    const float* dt,
    const float* tr,
    const float* D_param,
    float* y,
    int B, int L, int H, int P
) {
    int b = blockIdx.x;
    int h = blockIdx.y;
    int p = threadIdx.x;
    if (p >= P) return;

    // Strides for contiguous tensors (B, L, H, X)
    const int x_ld = H * P;      // L stride for x
    const int x_hd = P;          // H stride for x
    const int b_ld = H * D;      // L stride for B_proj / C_proj
    const int b_hd = D;          // H stride for B_proj / C_proj
    const int s_ld = H;          // L stride for scalars
    const int y_ld = H * P;      // L stride for y
    const int y_hd = P;          // H stride for y

    // Base pointers for this (b, h)
    const float* x_off   = x      + b * L * x_ld + h * x_hd;
    const float* B_off   = B_proj + b * L * b_ld + h * b_hd;
    const float* C_off   = C_proj + b * L * b_ld + h * b_hd;
    const float* decay_off = decay + b * L * s_ld + h;
    const float* dt_off    = dt    + b * L * s_ld + h;
    const float* tr_off    = tr    + b * L * s_ld + h;
    float* y_off        = y      + b * L * y_ld + h * y_hd;

    // h[D] and Bx_prev[D] in registers
    float h_reg[D];
    float prev[D];
    #pragma unroll
    for (int d = 0; d < D; d++) {
        h_reg[d] = 0.0f;
        prev[d] = 0.0f;
    }

    float d_val = D_param[h];

    // Shared memory for per-timestep B and C
    __shared__ float sB[D];
    __shared__ float sC[D];

    for (int l = 0; l < L; l++) {
        // Load B_t and C_t into shared (cooperatively over P threads for D elements)
        for (int i = p; i < D; i += blockDim.x) {
            sB[i] = B_off[l * b_ld + i];
            sC[i] = C_off[l * b_ld + i];
        }
        __syncthreads();

        float x_t = x_off[l * x_ld + p];
        float decay_t = decay_off[l * s_ld];
        float dt_t    = dt_off[l * s_ld];
        float tr_t    = tr_off[l * s_ld];

        float out_sum = 0.0f;

        #pragma unroll
        for (int d = 0; d < D; d++) {
            float Bx     = x_t * sB[d];
            float blend  = (1.0f - tr_t) * Bx + tr_t * 0.5f * (Bx + prev[d]);
            h_reg[d]     = decay_t * h_reg[d] + dt_t * blend;
            out_sum     += sC[d] * h_reg[d];
            prev[d]      = Bx;
        }

        y_off[l * y_ld + p] = out_sum + d_val * x_t;
    }
}


// ================================================================
// MIMO State Kernel
// Grid: (B, H)      — one block per (batch, head)
// Block: (D, 1, 1)  — one thread per state dimension
//
// Takes pre-mixed x_mixed (B, L, H, R) and produces y_r (B, L, H, R).
// The final output y = mimo_o @ y_r is computed in PyTorch.
// ================================================================

template<int R>
__global__ void mimo_state_kernel(
    const float* x_mixed,
    const float* B_proj,
    const float* C_proj,
    const float* decay,
    const float* dt,
    const float* tr,
    const float* D_param,
    float* y_r_out,
    int B, int L, int H, int Ds
) {
    int b = blockIdx.x;
    int h = blockIdx.y;
    int d = threadIdx.x;
    if (d >= Ds) return;

    // x_mixed: (B, L, H, R). For block (b,h): xm = &x_mixed[b,0,h,0], L stride = H*R
    // B_proj: (B, L, R, H, Ds). For block (b,h): B_off = &B_proj[b,0,0,h,0], L stride = R*H*Ds, R stride = H*Ds
    // y_r_out: (B, L, H, R). For block (b,h): y_out = &y_r_out[b,0,h,0], L stride = H*R
    const int L_stride_x = H * R;
    const float* xm    = x_mixed + b * L * L_stride_x + h * R;
    const float* B_off = B_proj  + b * L * R * H * Ds + h * Ds;
    const float* C_off = C_proj  + b * L * R * H * Ds + h * Ds;
    const float* dcy   = decay   + b * L * H + h;
    const float* dtt   = dt      + b * L * H + h;
    const float* trr   = tr      + b * L * H + h;
    float* y_out       = y_r_out + b * L * L_stride_x + h * R;

    const int L_stride_B = R * H * Ds;
    const int R_stride_B = H * Ds;

    // Register state
    float h_reg = 0.0f;
    float prev  = 0.0f;

    // Shared: x_mixed per timestep (R floats) + reduction buffer (R * D floats)
    __shared__ float s_xr[R];
    extern __shared__ float s_red[];  // [R * blockDim.x]

    for (int l = 0; l < L; l++) {
        // Load x_mixed into shared
        if (d < R) s_xr[d] = __ldg(&xm[l * L_stride_x + d]);
        __syncthreads();

        // Bx[d] = sum_r(x_r[r] * B_t[r, d])
        float Bx = 0.0f;
        for (int r = 0; r < R; r++) {
            Bx += s_xr[r] * __ldg(&B_off[l * L_stride_B + r * R_stride_B + d]);
        }

        float tr_t   = __ldg(&trr[l]);
        float decay_t = __ldg(&dcy[l]);
        float dt_t   = __ldg(&dtt[l]);

        float blend = (1.0f - tr_t) * Bx + tr_t * 0.5f * (Bx + prev);
        h_reg = decay_t * h_reg + dt_t * blend;
        prev  = Bx;

        // Partial y_r: C_t[r, d] * h_reg
        for (int r = 0; r < R; r++) {
            s_red[r * blockDim.x + d] = __ldg(&C_off[l * L_stride_B + r * R_stride_B + d]) * h_reg;
        }
        __syncthreads();

        // Tree-reduce over D for each r
        for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
            if (d < s) {
                for (int r = 0; r < R; r++) {
                    s_red[r * blockDim.x + d] += s_red[r * blockDim.x + d + s];
                }
            }
            __syncthreads();
        }

        if (d == 0) {
            for (int r = 0; r < R; r++) {
                y_out[l * L_stride_x + r] = s_red[r * blockDim.x];
            }
        }
    }
}


// ================================================================
// Explicit template instantiations
// ================================================================

#define INSTANTIATE_SISO(D)                                                               \
    template __global__ void siso_scan_kernel<D>(                                          \
        const float*, const float*, const float*, const float*,                            \
        const float*, const float*, const float*, float*, int, int, int, int)

INSTANTIATE_SISO(16);
INSTANTIATE_SISO(32);
INSTANTIATE_SISO(64);
INSTANTIATE_SISO(128);

#define INSTANTIATE_MIMO(R)                                                               \
    template __global__ void mimo_state_kernel<R>(                                          \
        const float*, const float*, const float*, const float*,                            \
        const float*, const float*, const float*, float*, int, int, int, int)

INSTANTIATE_MIMO(1);
INSTANTIATE_MIMO(2);
INSTANTIATE_MIMO(4);
INSTANTIATE_MIMO(8);

// ================================================================
// Raw-C launchers (compiled by NVCC so <<<>>> syntax works)
// No torch/STL headers — avoids MSVC 14.44 STL CUDA version assert
// ================================================================

void siso_scan_launcher(
    const float* x, const float* B_proj, const float* C_proj,
    const float* decay, const float* dt, const float* tr,
    const float* D_param, float* y,
    int B, int L, int H, int P, int D, cudaStream_t stream
) {
    dim3 grid(B, H);
    dim3 block(P);

    switch (D) {
        case 16:
            siso_scan_kernel<16><<<grid, block, 0, stream>>>(
                x, B_proj, C_proj, decay, dt, tr, D_param, y, B, L, H, P
            );
            break;
        case 32:
            siso_scan_kernel<32><<<grid, block, 0, stream>>>(
                x, B_proj, C_proj, decay, dt, tr, D_param, y, B, L, H, P
            );
            break;
        case 64:
            siso_scan_kernel<64><<<grid, block, 0, stream>>>(
                x, B_proj, C_proj, decay, dt, tr, D_param, y, B, L, H, P
            );
            break;
        case 128:
            siso_scan_kernel<128><<<grid, block, 0, stream>>>(
                x, B_proj, C_proj, decay, dt, tr, D_param, y, B, L, H, P
            );
            break;
    }
}

void mimo_state_launcher(
    const float* x_mixed, const float* B_proj, const float* C_proj,
    const float* decay, const float* dt, const float* tr,
    const float* D_param, float* y,
    int B, int L, int H, int Ds, int R, cudaStream_t stream
) {
    size_t shmem = R * Ds * sizeof(float);
    dim3 grid(B, H);
    dim3 block(Ds);

    switch (R) {
        case 1:
            mimo_state_kernel<1><<<grid, block, shmem, stream>>>(
                x_mixed, B_proj, C_proj, decay, dt, tr, D_param, y, B, L, H, Ds
            );
            break;
        case 2:
            mimo_state_kernel<2><<<grid, block, shmem, stream>>>(
                x_mixed, B_proj, C_proj, decay, dt, tr, D_param, y, B, L, H, Ds
            );
            break;
        case 4:
            mimo_state_kernel<4><<<grid, block, shmem, stream>>>(
                x_mixed, B_proj, C_proj, decay, dt, tr, D_param, y, B, L, H, Ds
            );
            break;
        case 8:
            mimo_state_kernel<8><<<grid, block, shmem, stream>>>(
                x_mixed, B_proj, C_proj, decay, dt, tr, D_param, y, B, L, H, Ds
            );
            break;
    }
}
"""

    cpp_source = r"""
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

// Forward declarations of raw-C launchers (linked from CUDA source)
void siso_scan_launcher(
    const float*, const float*, const float*, const float*,
    const float*, const float*, const float*, float*,
    int, int, int, int, int, cudaStream_t);
void mimo_state_launcher(
    const float*, const float*, const float*, const float*,
    const float*, const float*, const float*, float*,
    int, int, int, int, int, cudaStream_t);

// ---------- SISO torch wrapper ----------
torch::Tensor siso_scan(
    torch::Tensor x, torch::Tensor B_proj, torch::Tensor C_proj,
    torch::Tensor decay, torch::Tensor dt, torch::Tensor tr,
    torch::Tensor D_param
) {
    TORCH_CHECK(x.device().is_cuda());
    int B = x.size(0), L = x.size(1), H = x.size(2), P = x.size(3);
    int D = B_proj.size(-1);
    auto y = torch::empty({B, L, H, P}, x.options());
    auto stream = at::cuda::getCurrentCUDAStream();
    siso_scan_launcher(
        x.data_ptr<float>(), B_proj.data_ptr<float>(), C_proj.data_ptr<float>(),
        decay.data_ptr<float>(), dt.data_ptr<float>(), tr.data_ptr<float>(),
        D_param.data_ptr<float>(), y.data_ptr<float>(),
        B, L, H, P, D, stream
    );
    return y;
}

// ---------- MIMO torch wrapper ----------
torch::Tensor mimo_state(
    torch::Tensor x_mixed, torch::Tensor B_proj, torch::Tensor C_proj,
    torch::Tensor decay, torch::Tensor dt, torch::Tensor tr,
    torch::Tensor D_param, int64_t R
) {
    TORCH_CHECK(x_mixed.device().is_cuda());
    int B = x_mixed.size(0), L = x_mixed.size(1), H = x_mixed.size(2);
    int Ds = B_proj.size(-1);
    auto y = torch::empty({B, L, H, (int)R}, x_mixed.options());
    auto stream = at::cuda::getCurrentCUDAStream();
    mimo_state_launcher(
        x_mixed.data_ptr<float>(), B_proj.data_ptr<float>(), C_proj.data_ptr<float>(),
        decay.data_ptr<float>(), dt.data_ptr<float>(), tr.data_ptr<float>(),
        D_param.data_ptr<float>(), y.data_ptr<float>(),
        B, L, H, Ds, (int)R, stream
    );
    return y;
}
"""

    try:
        import os, subprocess, sys
        os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.6")
        vcvars = r"C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
        if os.path.exists(vcvars) and not os.environ.get("VSCMD_VER"):
            env = os.environ.copy()
            result = subprocess.run(
                f'cmd /c "call "{vcvars}" >nul 2>nul & set"',
                capture_output=True, text=True, shell=True, env=env
            )
            for line in result.stdout.splitlines():
                if "=" in line:
                    k, v = line.split("=", 1)
                    os.environ[k] = v
        from torch.utils.cpp_extension import load_inline
        _CUDA_MODULE = load_inline(
            name="mamba3_scan_cuda",
            cpp_sources=cpp_source,
            cuda_sources=cuda_source,
            functions=["siso_scan", "mimo_state"],
            verbose=True,
            extra_cuda_cflags=["--use_fast_math", "-O3", "-allow-unsupported-compiler", "-D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH"],
        )
    except Exception as _cuda_err:
        import traceback; traceback.print_exc()
        _CUDA_MODULE = None

    return _CUDA_MODULE


# =====================================================================
# Python wrappers with fallback to pure-Python ops
# =====================================================================

def siso_scan_cuda(
    x: torch.Tensor,
    B_proj: torch.Tensor,
    C_proj: torch.Tensor,
    decay: torch.Tensor,
    dt: torch.Tensor,
    tr: torch.Tensor,
    D_param: torch.Tensor,
) -> Optional[torch.Tensor]:
    """Run SISO scan via CUDA. Returns None if CUDA unavailable."""
    mod = _compile_cuda_extension()
    if mod is None:
        return None

    # Ensure contiguous
    x = x.contiguous()
    B_proj = B_proj.contiguous()
    C_proj = C_proj.contiguous()
    decay = decay.contiguous()
    dt = dt.contiguous()
    tr = tr.contiguous()
    D_param = D_param.contiguous()

    return mod.siso_scan(x, B_proj, C_proj, decay, dt, tr, D_param)


def mimo_state_cuda(
    x: torch.Tensor,
    B_proj: torch.Tensor,
    C_proj: torch.Tensor,
    decay: torch.Tensor,
    dt: torch.Tensor,
    tr: torch.Tensor,
    D_param: torch.Tensor,
    mimo_x: torch.Tensor,
    mimo_o: torch.Tensor,
    R: int,
) -> Optional[torch.Tensor]:
    """Run MIMO scan via CUDA (split design). Returns None if unavailable.

    Steps:
      1. x_mixed = einsum("blhp,hrp->blhr", x, mimo_x)     — PyTorch
      2. y_r     = CUDA state kernel                         — fused scan
      3. y       = einsum("blhr,hrp->blhp", y_pre, mimo_o)  — PyTorch
    """
    mod = _compile_cuda_extension()
    if mod is None:
        return None

    # Step 1: pre-mix x with mimo_x
    x_mixed = torch.einsum("blhp,hrp->blhr", x.float(), mimo_x.float())

    # Step 2: CUDA state kernel
    B_proj = B_proj.contiguous()
    C_proj = C_proj.contiguous()
    decay = decay.contiguous()
    dt = dt.contiguous()
    tr = tr.contiguous()
    D_param = D_param.contiguous()

    y_r = mod.mimo_state(x_mixed, B_proj, C_proj, decay, dt, tr, D_param, R)

    # Add skip connection
    skip = D_param.unsqueeze(0).unsqueeze(0).unsqueeze(-1) * x_mixed  # (B, L, H, R)
    y_pre = y_r + skip

    # Step 3: post-mix with mimo_o
    y_out = torch.einsum("blhr,hrp->blhp", y_pre, mimo_o.float())
    return y_out.to(x.dtype)
