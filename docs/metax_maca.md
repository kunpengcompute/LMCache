# MetaX MACA build notes

This document describes the initial LMCache build path for MetaX MACA
environments such as MetaX C500 with `torch+metax`.

## Scope

The first supported target is the minimal vLLM KV round-trip path:

```text
vLLM GPU KV cache
  -> LMCache GPU connector
  -> LMCache CPU memory
  -> LMCache GPU connector
  -> vLLM GPU KV cache
```

Mooncake TCP can be validated after this path works. RDMA, GPU Direct, P2P,
CacheGen, CacheBlend, and model-specific advanced paths should be validated in
later phases.

## Prerequisites

Use an environment that already contains the MetaX PyTorch build. Do not let
pip replace it with a regular PyPI PyTorch wheel.

Check the environment first:

```bash
python - <<'PY'
import os
import platform
import torch

print("python", platform.python_version())
print("machine", platform.machine())
print("torch", torch.__version__)
print("torch.version.cuda", getattr(torch.version, "cuda", None))
print("torch.version.maca", getattr(torch.version, "maca", None))
print("cuda available", torch.cuda.is_available())
print("device count", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device name", torch.cuda.get_device_name(0))
    print("capability", torch.cuda.get_device_capability(0))
print("MACA_HOME", os.getenv("MACA_HOME"))
print("LMCACHE_MACA_HOME", os.getenv("LMCACHE_MACA_HOME"))
PY
```

## Build command

Use the MACA build branch explicitly:

```bash
BUILD_WITH_MACA=1 \
pip install -e . --no-build-isolation --no-deps
```

`--no-build-isolation` avoids creating a temporary build environment that may
install regular `torch==2.8.0`. `--no-deps` avoids replacing the preinstalled
`torch+metax` package or installing CUDA-only dependencies such as
`cupy-cuda12x`.

## Optional environment variables

The MACA branch works with explicit overrides when the runtime image does not
export standard paths:

```bash
export LMCACHE_MACA_HOME=/path/to/maca
export LMCACHE_MACA_COMPILER=/path/to/maca/compiler
export LMCACHE_MACA_INCLUDE_DIRS=/path/to/include1:/path/to/include2
export LMCACHE_MACA_LIBRARY_DIRS=/path/to/lib1:/path/to/lib2
```

`MACA_HOME` and `MACA_COMPILER` are also recognized. If these variables are not
set, the build relies on `torch.utils.cpp_extension` and compiler default search
paths.

When `LMCACHE_MACA_HOME` or `MACA_HOME` is set, the build branch also exposes it
as `CUDA_HOME` before importing `torch.utils.cpp_extension`. This keeps the
CUDA-compatible extension discovery path usable for torch+metax builds.

## Post-build smoke checks

```bash
python - <<'PY'
import lmcache.c_ops as ops

print("lmcache.c_ops ok")
print("has multi_layer_kv_transfer", hasattr(ops, "multi_layer_kv_transfer"))
PY
```

Then validate PyTorch stream and copy behavior:

```bash
python - <<'PY'
import torch

stream = torch.cuda.Stream()
x = torch.empty((16, 16), device="cuda")
y = torch.empty_like(x, device="cpu", pin_memory=True)
with torch.cuda.stream(stream):
    y.copy_(x, non_blocking=True)
torch.cuda.synchronize()
print("stream and pinned non_blocking copy ok")
PY
```

Only after these checks pass should vLLM be started with
`kv_connector="LMCacheConnectorV1"` and `kv_role="kv_both"`.
