from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='emd',
    ext_modules=[
        CUDAExtension('emd', [
            'emd.cpp',
            'emd_cuda.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })

# setup(
#     name='chamfer_distance',
#     ext_modules=[
#         CUDAExtension('chamfer_distance', [
#             'chamfer_distance.cpp',
#             'chamfer_distance.cu',
#         ]),
#     ],
#     cmdclass={
#         'build_ext': BuildExtension
#     })