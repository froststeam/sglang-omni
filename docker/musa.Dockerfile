# syntax=docker/dockerfile:1.7

# SGLang-Omni MUSA image. Build SGLang's MUSA image first:
#   docker build -f docker/musa.Dockerfile -t sglang:musa <sglang-repo>
# Then build this image:
#   docker build -f docker/musa.Dockerfile -t sglang-omni:musa .

ARG SGLANG_MUSA_IMAGE=sglang:musa

FROM ${SGLANG_MUSA_IMAGE} AS runtime

ARG OMNI_REPO=https://github.com/sgl-project/sglang-omni.git
ARG OMNI_REF=main

ENV SGLANG_OMNI_REPO_DIR=/workspace/sglang-omni

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ffmpeg \
        git \
        libsndfile1 \
        sox \
    && rm -rf /var/lib/apt/lists/*

RUN git clone "${OMNI_REPO}" "${SGLANG_OMNI_REPO_DIR}" \
    && git -C "${SGLANG_OMNI_REPO_DIR}" checkout --detach "${OMNI_REF}"

WORKDIR ${SGLANG_OMNI_REPO_DIR}

COPY pyproject_musa.toml /tmp/pyproject_musa.toml
RUN python3 -m pip install --upgrade pip setuptools wheel \
    && cp pyproject.toml /tmp/pyproject_cuda.toml \
    && cp /tmp/pyproject_musa.toml pyproject.toml \
    && python3 -m pip install --no-build-isolation -e . \
        --index-url https://pypi.org/simple \
        --extra-index-url https://dl.mthreads.com/repo/api/pypi/pypi/simple \
        --trusted-host dl.mthreads.com \
    && cp /tmp/pyproject_cuda.toml pyproject.toml \
    && rm /tmp/pyproject_musa.toml /tmp/pyproject_cuda.toml

RUN python3 - <<'PY'
import torch

assert getattr(torch.version, "musa", None), "the inherited PyTorch build is not MUSA-enabled"
assert hasattr(torch, "musa") and torch.musa.is_available(), "torch.musa is unavailable"
import torchada  # noqa: F401
import triton
assert triton.__version__ == "3.2.0", triton.__version__
import triton.backends.mtgpu  # noqa: F401
import sglang  # noqa: F401
import sglang_omni  # noqa: F401
PY

CMD ["/bin/bash"]
