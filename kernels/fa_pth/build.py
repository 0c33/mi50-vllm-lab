"""Build the gfx906 int8-per-token-head FA extension (_gfx906_fa_pth) as a torch extension."""
import os

from torch.utils.cpp_extension import load

here = os.path.dirname(os.path.abspath(__file__))
build = os.path.join(here, "build")
import shutil
shutil.rmtree(build, ignore_errors=True)  # headers are not tracked: always rebuild
os.makedirs(build, exist_ok=True)
os.environ.setdefault("PYTORCH_ROCM_ARCH", "gfx906")
flags = ["-O3", "-std=c++17", "--offload-arch=gfx906", "-DGGML_USE_HIP", "-DGGML_HIP_GFX906",
         "-DFLASH_ATTN_AVAILABLE", "-D__HIP_PLATFORM_AMD__", "-U__HIP_NO_HALF_OPERATORS__",
         "-U__HIP_NO_HALF_CONVERSIONS__", f"-I{here}", f"-I{here}/kernel"]
load(name="_gfx906_fa_pth", sources=[os.path.join(here, f) for f in ("pth_fa.cpp", "gfx906_fa_launcher.cu")],
     extra_cflags=["-O3", "-std=c++17"], extra_cuda_cflags=flags, extra_include_paths=[here, f"{here}/kernel"],
     build_directory=build, is_python_module=False, verbose=False)
print("built:", [f for f in os.listdir(build) if f.endswith(".so")])
