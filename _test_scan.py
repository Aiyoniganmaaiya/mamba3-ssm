import os, sys, time
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
sys.path.insert(0, r'C:\Users\Chang\OneDrive\Desktop\mamba3-reproduce')
for m in list(sys.modules):
    if 'mamba3_ssm' in m: del sys.modules[m]

import torch
from mamba3_ssm.ops import ssm_scan_siso, ssm_scan_mimo

def siso_ref(x, Bp, Cp, ADT, DT, trap, D):
    B,L,H,P = x.shape; Ds=Bp.shape[-1]
    h=torch.zeros(B,H,P,Ds,dtype=torch.float64)
    Bxp=torch.zeros(B,H,P,Ds,dtype=torch.float64)
    outs=[]
    for t in range(L):
        xt=x[:,t].double(); Bt=Bp[:,t].double(); Ct=Cp[:,t].double()
        adt=ADT[:,t].double(); dt=DT[:,t].double(); tr=trap[:,t].double().sigmoid()
        decay=torch.exp(adt).unsqueeze(-1).unsqueeze(-1)
        Bx=torch.einsum("bhp,bhd->bhpd",xt,Bt)
        blend=(1-tr.unsqueeze(-1).unsqueeze(-1))*Bx+tr.unsqueeze(-1).unsqueeze(-1)*0.5*(Bx+Bxp)
        h=decay*h+dt.unsqueeze(-1).unsqueeze(-1)*blend
        y=torch.einsum("bhd,bhpd->bhp",Ct,h)+D.double().unsqueeze(0).unsqueeze(-1)*xt
        outs.append(y); Bxp=Bx
    return torch.stack(outs,dim=1)

torch.manual_seed(42)
B,L,H,P,D=2,128,4,32,64
x=torch.randn(B,L,H,P); Bp=torch.randn(B,L,H,D); Cp=torch.randn(B,L,H,D)
ADT=-torch.rand(B,L,H)*0.1; DT=torch.rand(B,L,H)*0.01+0.001
trap=torch.sigmoid(torch.randn(B,L,H)); Dw=torch.ones(H)

for cs in [64, 32, 16]:
    y=ssm_scan_siso(x,Bp,Cp,ADT,DT,trap,Dw,chunk_size=cs)
    y_ref=siso_ref(x,Bp,Cp,ADT,DT,trap,Dw)
    diff=(y.double()-y_ref).abs().max().item()
    print(f'chunk={cs}: max diff={diff:.8f} {"PASS" if diff<1e-4 else "FAIL"}')

# Speed
for _ in range(5): _=ssm_scan_siso(x,Bp,Cp,ADT,DT,trap,Dw,chunk_size=64)
t0=time.time()
for _ in range(50): _=ssm_scan_siso(x,Bp,Cp,ADT,DT,trap,Dw,chunk_size=64)
print(f'Chunked scan (L=128): {(time.time()-t0)/50*1000:.1f}ms')

# 306M-scale
x_t=torch.randn(1,256,48,64).cuda()
Bp_t=torch.randn(1,256,48,64).cuda()
Cp_t=torch.randn(1,256,48,64).cuda()
ADT_t=-torch.rand(1,256,48).cuda()*0.1
DT_t=torch.rand(1,256,48).cuda()*0.01+0.001
trap_t=torch.sigmoid(torch.randn(1,256,48).cuda())
Dw_t=torch.ones(48).cuda()

for _ in range(3): _=ssm_scan_siso(x_t,Bp_t,Cp_t,ADT_t,DT_t,trap_t,Dw_t,chunk_size=64)
t0=time.time()
for _ in range(20): _=ssm_scan_siso(x_t,Bp_t,Cp_t,ADT_t,DT_t,trap_t,Dw_t,chunk_size=64)
torch.cuda.synchronize()
t_scan=(time.time()-t0)/20*1000
print(f'306M scan (L=256): {t_scan:.1f}ms/step')
steps_per_epoch = 42000  # approximate for TinyStories with medium preset
print(f'Epoch time (grad_accum=16): {t_scan*16*steps_per_epoch/3600:.1f}s = {t_scan*16*steps_per_epoch/3600/60:.1f}min')
