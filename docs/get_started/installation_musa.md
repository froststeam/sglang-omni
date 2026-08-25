# 🚀 Installation — MUSA

This installs `sglang-omni` on Moore Threads GPUs.

The flow matches other accelerator builds: SGLang owns the MUSA runtime image
and base Python stack; SGLang-Omni installs on top as the overlay.

## Prerequisites

Install the Moore Threads driver/runtime on the host first.

Install SGLang's MUSA environment first. The current MUSA stack uses the Moore
Threads torch 2.9.1 wheels, MUSA Triton 3.2.0, and MUSA TileLang:

```bash
python -m pip install --index-url https://dl.mthreads.com/repo/api/pypi/pypi/simple \
  --trusted-host dl.mthreads.com \
  torch==2.9.1.post1+musa5.2.0 \
  torch_musa==2.9.1.post1+musa5.2.0 \
  torchaudio==2.9.1+musa5.2.0 \
  torchvision==0.24.1.post1+musa5.2.0 \
  triton==3.2.0 \
  tilelang_musa==0.1.8+musa.3
```

Then install SGLang from source with the MUSA extras:

```bash
git clone https://github.com/sgl-project/sglang.git
cd sglang/python
cp pyproject_other.toml pyproject.toml
python -m pip install -e ".[all_musa]" --no-build-isolation \
  --index-url https://pypi.org/simple \
  --extra-index-url https://dl.mthreads.com/repo/api/pypi/pypi/simple \
  --trusted-host dl.mthreads.com
```

## 🐳 Option A: Docker

Build the SGLang MUSA image first:

```bash
git clone https://github.com/sgl-project/sglang.git
cd sglang
docker build -f docker/musa.Dockerfile -t sglang:musa .
```

Then build SGLang-Omni on top of it:

```bash
git clone https://github.com/sgl-project/sglang-omni.git
cd sglang-omni
docker build -f docker/musa.Dockerfile -t sglang-omni:musa .
```

Use a different SGLang base image or SGLang-Omni revision when needed:

```bash
docker build -f docker/musa.Dockerfile \
  --build-arg SGLANG_MUSA_IMAGE=<your-sglang-musa-image> \
  --build-arg OMNI_REF=<sglang-omni commit-or-tag> \
  -t sglang-omni:musa .
```

Run it with the MUSA devices exposed:

```bash
docker run -it \
  --shm-size 32g \
  --ipc host \
  --network host \
  --privileged \
  sglang-omni:musa
```

## 🛠️ Option B: Install from source in an existing SGLang MUSA environment

```bash
git clone https://github.com/sgl-project/sglang-omni.git
cd sglang-omni

python -m pip install --upgrade pip setuptools wheel
cp pyproject.toml /tmp/pyproject_omni_cuda.toml
cp pyproject_musa.toml pyproject.toml
python -m pip install --no-build-isolation -e . \
  --index-url https://pypi.org/simple \
  --extra-index-url https://dl.mthreads.com/repo/api/pypi/pypi/simple \
  --trusted-host dl.mthreads.com
cp /tmp/pyproject_omni_cuda.toml pyproject.toml
rm /tmp/pyproject_omni_cuda.toml
```

`pyproject_musa.toml` does not install `torch`, `torch_musa`, Triton, TileLang,
MATE, FlashAttention, or SGLang. Those are inherited from the SGLang MUSA
environment. It keeps CUDA-only Omni dependencies out of the base MUSA install.

If a model needs TorchCodec, install the Moore Threads MUSA TorchCodec build:

```bash
cp pyproject.toml /tmp/pyproject_omni_cuda.toml
cp pyproject_musa.toml pyproject.toml
python -m pip install --no-build-isolation -e ".[torchcodec-musa]" \
  --index-url https://pypi.org/simple \
  --extra-index-url https://dl.mthreads.com/repo/api/pypi/pypi/simple \
  --trusted-host dl.mthreads.com
cp /tmp/pyproject_omni_cuda.toml pyproject.toml
rm /tmp/pyproject_omni_cuda.toml
```

If you start from a clean container, install SGLang first:

```bash
python -m pip install --index-url https://dl.mthreads.com/repo/api/pypi/pypi/simple \
  --trusted-host dl.mthreads.com \
  torch==2.9.1.post1+musa5.2.0 \
  torch_musa==2.9.1.post1+musa5.2.0 \
  torchaudio==2.9.1+musa5.2.0 \
  torchvision==0.24.1.post1+musa5.2.0 \
  triton==3.2.0 \
  tilelang_musa==0.1.8+musa.3

python -m pip install "torchada>=0.1.84"

git clone https://github.com/sgl-project/sglang.git /tmp/sglang
cd /tmp/sglang/python
cp pyproject_other.toml pyproject.toml
python -m pip install -e ".[all_musa]" --no-build-isolation \
  --index-url https://pypi.org/simple \
  --extra-index-url https://dl.mthreads.com/repo/api/pypi/pypi/simple \
  --trusted-host dl.mthreads.com

cd /path/to/sglang-omni
cp pyproject.toml /tmp/pyproject_omni_cuda.toml
cp pyproject_musa.toml pyproject.toml
python -m pip install --no-build-isolation -e . \
  --index-url https://pypi.org/simple \
  --extra-index-url https://dl.mthreads.com/repo/api/pypi/pypi/simple \
  --trusted-host dl.mthreads.com
cp /tmp/pyproject_omni_cuda.toml pyproject.toml
rm /tmp/pyproject_omni_cuda.toml
```

## Verify

```bash
python - <<'PY'
import torch
import torchada  # noqa: F401
import triton
import triton.backends.mtgpu  # noqa: F401
import sglang
import sglang_omni

print("torch:", torch.__version__)
print("musa:", torch.version.musa)
print("triton:", triton.__version__)
print("devices:", torch.musa.device_count())
print("sglang:", sglang.__version__)
print("sglang_omni:", sglang_omni.__file__)
PY
```
