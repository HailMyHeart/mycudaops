from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os
from glob import glob

# 自动获取 csrc 目录下所有的源文件（包含 .cu、.cpp、.cc、.c），
# 否则像 bindings.cpp 这样的 C++ 绑定文件不会被编入扩展，导致缺少 PyInit_* 导出。
sources = []
for pattern in ('*.cu', '*.cpp', '*.cc', '*.c'):
    sources.extend(glob(os.path.join('csrc', pattern)))

setup(
    name='my_lib',
    ext_modules=[
        CUDAExtension(
            name='my_lib', # 统一叫 my_lib
            sources=sources,
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': [
                    '-O3',
                    '--use_fast_math', # 开启快速数学运算，提升带宽类算子性能
                    '-arch=sm_89'      # 针对 4090 (Ada架构) 进行指令优化
                ]
            }
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })