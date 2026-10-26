# Shared settings for the serve scripts. Override any of these from the environment.
IMAGE=${IMAGE:-vllm-gfx906-kintegrated:0.30.0-rocm7.14-opt}
MODELS=${MODELS:-$HOME/models}               # holds Qwen3.8-27B-AWQ-INT4 (+ Qwen3.8-27B-DFlash2)
CACHE_DIR=${CACHE_DIR:-$HOME/.cache/vllm-gfx906}  # HF + torch.compile cache, mounted at /root/.cache
RENDER_GID=${RENDER_GID:-$(getent group render | cut -d: -f3)}
VIDEO_GID=${VIDEO_GID:-$(getent group video | cut -d: -f3)}
mkdir -p "$CACHE_DIR"
