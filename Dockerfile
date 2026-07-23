# GPU experiment image for llm-reasoners (demo.ipynb: CoT vs ToT vs RAP on Blocksworld).
# Built and run on a CUDA-capable Linux host with the NVIDIA Container Toolkit
# (`docker run --gpus all ...` / `docker compose` with a `gpus` reservation).
#
# Base: CUDA 12.6 devel (not runtime-only) because bitsandbytes builds CUDA kernels
# at install time. Ubuntu 22.04 ships Python 3.10 by default, matching pyproject.toml's floor.
FROM nvidia/cuda:12.6.3-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-venv python-is-python3 \
        git \
        cmake build-essential g++ \
        curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# --- VAL (Blocksworld plan validator) ---
# Built from source from the canonical, actively-attributed upstream (KCL-Planning/VAL,
# BSD-3-Clause: Fox, Long, Howey, Cresswell) rather than trusting the unaudited precompiled
# binaries vendored inside the LLMs-Planning submodule (see llm-prompts/context/llm-reasoners
# planning notes for the full reasoning). Independent of repo content, so this layer caches
# regardless of source changes below.
RUN git clone --depth 1 https://github.com/KCL-Planning/VAL.git /opt/VAL-src \
    && cd /opt/VAL-src && mkdir build && cd build \
    && cmake -DCMAKE_BUILD_TYPE=Release .. \
    && cmake --build . --target Validate -- -j"$(nproc)" \
    && mkdir -p /opt/VAL \
    && cp bin/Validate bin/libVAL.so /opt/VAL/ \
    && ln -s Validate /opt/VAL/validate \
    && rm -rf /opt/VAL-src
ENV VAL=/opt/VAL

WORKDIR /workspace

# Install dependencies before copying the rest of the source, so this layer is cached
# across source-only changes (poetry.toml sets virtualenvs.create=false: installs
# straight into this image's system Python, no nested venv).
COPY pyproject.toml poetry.lock poetry.toml ./
RUN pip install --no-cache-dir poetry==2.4.1 \
    && poetry install --with notebook,dev --no-root

COPY . .

# LLMs-Planning must be checked out on the host before `docker build` (`git submodule
# update --init`) — Docker's build context is a plain file copy, it won't fetch submodules.
RUN test -f LLMs-Planning/LICENSE || \
    (echo "ERROR: LLMs-Planning submodule not initialized on the host. Run:" && \
     echo "  git submodule update --init" && exit 1)

RUN poetry install --with notebook,dev

# Model weights are not baked into the image (large, license-gated) — downloaded at
# container runtime into this mounted cache dir instead. Pass HF_TOKEN at `docker run`.
ENV HF_HOME=/workspace/.cache/huggingface
VOLUME ["/workspace/.cache/huggingface"]

EXPOSE 8888
CMD ["poetry", "run", "jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
