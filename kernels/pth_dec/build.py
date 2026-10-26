"""Build _gfx906_pth_dec.so: split-K decode attention over vLLM's int8 per-token-head KV cache (gfx906)."""
import os
import shutil

from torch.utils.cpp_extension import load

here = os.path.dirname(os.path.abspath(__file__))
build = os.path.join(here, "build")
shutil.rmtree(build, ignore_errors=True)  # torch does not track header changes: always rebuild
os.makedirs(build, exist_ok=True)
os.environ.setdefault("PYTORCH_ROCM_ARCH", "gfx906")
flags = ["-O3", "-std=c++17", "--offload-arch=gfx906", "-D__HIP_PLATFORM_AMD__",
         "-U__HIP_NO_HALF_OPERATORS__", "-U__HIP_NO_HALF_CONVERSIONS__"]
if os.environ.get("SAVE_TEMPS"):
    flags += ["-save-temps=obj"]
load(name="_gfx906_pth_dec", sources=[os.path.join(here, f) for f in ("pth_dec.cpp", "pth_dec.cu")],
     extra_cflags=["-O3", "-std=c++17"], extra_cuda_cflags=flags, build_directory=build,
     is_python_module=False, verbose=bool(os.environ.get("VERBOSE")))
print("built:", [f for f in os.listdir(build) if f.endswith(".so")])
