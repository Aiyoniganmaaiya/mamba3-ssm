"""
RTX 4060 Laptop 8GB — Recommended training configs

Benchmark results (bf16 + AdamW, forward+backward):
===========================================================================
Config                                       Params     Peak     Total Status
---------------------------------------------------------------------------
112M  d1024 nl16 ds64  bs1 sl256              112M    1.59GB    2.94GB OK
165M  d1024 nl24 ds64  bs1 sl256              168M    2.36GB    4.38GB OK
190M  d1280 nl20 ds64  bs1 sl256              215M    2.72GB    5.30GB OK
166M  d1024 nl24 ds128 bs1 sl256              182M    3.48GB    5.65GB OK
214M  d1536 nl16 ds64  bs1 sl256              245M    2.83GB    5.78GB OK
265M  d1536 nl20 ds64  bs1 sl256              306M    3.52GB    7.19GB FIT ← RECOMMENDED
317M  d1536 nl24 ds64  bs1 sl256              367M    4.21GB    8.61GB TIGHT
387M  d1536 nl24 ds128 bs1 sl256              387M    5.94GB   10.59GB TIGHT
===========================================================================

RECOMMENDED 380M-class config for RTX 4060 Laptop:
  d_model=1536, n_layer=20, d_state=64, bs=1, seq_len=256
  306M params, ~7.2GB total (fits with ~300MB headroom)
  
  Gradient accumulation (grad_accum=16) gives effective_batch=16

If you need seq_len=512:
  d_model=1024, n_layer=16, d_state=64, bs=2, seq_len=512
  112M params, ~5.6GB total — comfortable

If you want maximum quality (pushing the limit):
  d_model=1536, n_layer=24, d_state=64, bs=1, seq_len=256, grad_accum=16
  367M params, ~8.6GB — TIGHT, may OOM without cache clearing
"""

CONFIGS = {
    "small": {
        "d_model": 1024, "n_layer": 16, "d_state": 64,
        "batch_size": 2, "seq_len": 512, "grad_accum": 8,
        "desc": "112M params, ~5.6GB, seq_len=512"
    },
    "medium": {
        "d_model": 1536, "n_layer": 20, "d_state": 64,
        "batch_size": 1, "seq_len": 256, "grad_accum": 16,
        "desc": "306M params, ~7.2GB, seq_len=256"
    },
    "large": {
        "d_model": 1536, "n_layer": 24, "d_state": 64,
        "batch_size": 1, "seq_len": 256, "grad_accum": 16,
        "desc": "367M params, ~8.6GB (TIGHT)"
    },
}
