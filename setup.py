from setuptools import setup
import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='w2a16_gemm',
    ext_modules=[
        CUDAExtension(
            name='w2a16_gemm',
            sources=['w2a16_gemm.cu'],
            extra_compile_args={'cxx': ['-O3'], 'nvcc': ['-O3']}
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)
