"""Build gfx906_skinny.so (skinny fp16 GEMM, W4 GEMM, host all-reduce) as a torch extension."""
import os

from torch.utils.cpp_extension import load

here = os.path.dirname(os.path.abspath(__file__))
build = os.path.join(here, "build")
os.makedirs(build, exist_ok=True)
os.environ.setdefault("PYTORCH_ROCM_ARCH", "gfx906")
load(name="gfx906_skinny", sources=[os.path.join(here, f) for f in ("skinny.cu", "w4.cu", "hostar.cu")],
     extra_cuda_cflags=["-O3", "--offload-arch=gfx906", "-munsafe-fp-atomics"],
     build_directory=build, is_python_module=False, verbose=False)
print("built:", [f for f in os.listdir(build) if f.endswith(".so")])
