"""cuBLAS fp16 accumulation for every Python process of the BGE-M3 vLLM server.

bge_m3_service.sh always puts this directory first on PYTHONPATH, so the API server and the
engine-core process it spawns both import it at start-up, before vLLM builds or captures anything.
GeForce cards run fp16 tensor-core math with FP32 accumulation at half the fp16-accumulation rate;
this flag (PyTorch >= 2.7) makes torch's half GEMMs accumulate in fp16. FlashAttention keeps its
own FP32 accumulation.

Measured on the arc99 RTX 3090 (PLAN-2026-11-14-vllm-embed-host.md §5): 123 -> 171 chunks/s on
real code chunks; against FlagEmbedding fp16 over 25,000 chunks, dense cos mean 0.999956 / min
0.998087, sparse top-20 overlap mean 0.9937. The user chose it for production on 2026-11-14.
"""
import os
import sys

import torch

torch.backends.cuda.matmul.allow_fp16_accumulation = True
print(f"[fp16_accumulation] pid {os.getpid()}: torch.backends.cuda.matmul.allow_fp16_accumulation="
      f"{torch.backends.cuda.matmul.allow_fp16_accumulation}", file=sys.stderr, flush=True)
