# SPDX-License-Identifier: Apache-2.0
# Standard
from pathlib import Path
import os
import sys

# Third Party
from setuptools import find_packages, setup

ROOT_DIR = Path(__file__).parent
HIPIFY_DIR = os.path.join(ROOT_DIR, "csrc/")
HIPIFY_OUT_DIR = os.path.join(ROOT_DIR, "csrc_hip/")

# python -m build --sdist
# will run python setup.py sdist --dist-dir dist
BUILDING_SDIST = "sdist" in sys.argv or os.environ.get("NO_CUDA_EXT", "0") == "1"

# New environment variable to choose between CUDA and HIP
BUILD_WITH_HIP = os.environ.get("BUILD_WITH_HIP", "0") == "1"

BUILD_WITH_MACA = os.environ.get("BUILD_WITH_MACA", "0") == "1"

ENABLE_CXX11_ABI = os.environ.get("ENABLE_CXX11_ABI", "1") == "1"


def cxx_abi_flag() -> str:
    global ENABLE_CXX11_ABI
    if ENABLE_CXX11_ABI:
        return "-D_GLIBCXX_USE_CXX11_ABI=1"
    return "-D_GLIBCXX_USE_CXX11_ABI=0"


def cuda_sources() -> list[str]:
    return [
        "csrc/pybind.cpp",
        "csrc/mem_kernels.cu",
        "csrc/cal_cdf.cu",
        "csrc/ac_enc.cu",
        "csrc/ac_dec.cu",
        "csrc/pos_kernels.cu",
        "csrc/mem_alloc.cpp",
        "csrc/utils.cpp",
    ]


def storage_manager_sources() -> list[str]:
    return [
        "csrc/storage_manager/pybind.cpp",
        "csrc/storage_manager/ttl_lock.cpp",
    ]


def split_env_paths(name: str) -> list[str]:
    value = os.environ.get(name)
    if not value:
        return []
    return [path for path in value.split(os.pathsep) if path]


def hipify_wrapper() -> None:
    # Third Party
    from torch.utils.hipify.hipify_python import hipify

    print("Hipifying sources ")

    # Get absolute path for all source files.
    extra_files = [
        os.path.abspath(os.path.join(HIPIFY_DIR, item))
        for item in os.listdir(HIPIFY_DIR)
        if os.path.isfile(os.path.join(HIPIFY_DIR, item))
    ]

    hipify_result = hipify(
        project_directory=HIPIFY_DIR,
        output_directory=HIPIFY_OUT_DIR,
        header_include_dirs=[],
        includes=[],
        extra_files=extra_files,
        show_detailed=True,
        is_pytorch_extension=True,
        hipify_extra_files_only=True,
    )
    hipified_sources = []
    for source in extra_files:
        s_abs = os.path.abspath(source)
        hipified_s_abs = (
            hipify_result[s_abs].hipified_path
            if (
                s_abs in hipify_result
                and hipify_result[s_abs].hipified_path is not None
            )
            else s_abs
        )
        hipified_sources.append(hipified_s_abs)

    assert len(hipified_sources) == len(extra_files)


def cuda_extension() -> tuple[list, dict]:
    # Third Party
    from torch.utils import cpp_extension  # Import here

    print("Building CUDA extensions")
    flag_cxx_abi = cxx_abi_flag()
    ext_modules = [
        cpp_extension.CUDAExtension(
            "lmcache.c_ops",
            sources=cuda_sources(),
            extra_compile_args={
                "cxx": [flag_cxx_abi],
                "nvcc": [flag_cxx_abi],
            },
        ),
        cpp_extension.CppExtension(
            "lmcache.native_storage_ops",
            sources=storage_manager_sources(),
            include_dirs=["csrc/storage_manager"],
            extra_compile_args={
                "cxx": [flag_cxx_abi, "-O3"],
            },
        ),
    ]
    cmdclass = {"build_ext": cpp_extension.BuildExtension}
    return ext_modules, cmdclass


def maca_extension() -> tuple[list, dict]:
    import torch

    print("Building MetaX MACA extensions")
    print(f"torch version: {torch.__version__}")
    print(f"torch.version.cuda: {getattr(torch.version, 'cuda', None)}")
    print(f"torch.version.maca: {getattr(torch.version, 'maca', None)}")

    flag_cxx_abi = cxx_abi_flag()
    maca_home = os.environ.get("MACA_HOME") or os.environ.get("LMCACHE_MACA_HOME")
    maca_compiler = os.environ.get("MACA_COMPILER") or os.environ.get(
        "LMCACHE_MACA_COMPILER"
    )
    include_dirs = split_env_paths("LMCACHE_MACA_INCLUDE_DIRS")
    library_dirs = split_env_paths("LMCACHE_MACA_LIBRARY_DIRS")

    if maca_home and "CUDA_HOME" not in os.environ:
        os.environ["CUDA_HOME"] = maca_home

    if maca_compiler:
        os.environ.setdefault("CUDACXX", maca_compiler)

    from torch.utils import cpp_extension  # Import here

    if maca_home:
        include_dirs.append(os.path.join(maca_home, "include"))
        library_dirs.extend(
            [
                os.path.join(maca_home, "lib"),
                os.path.join(maca_home, "lib64"),
            ]
        )
    else:
        print(
            "MACA_HOME/LMCACHE_MACA_HOME is not set; relying on torch+metax "
            "and compiler default search paths."
        )

    if maca_compiler:
        print(f"Using MACA compiler from environment: {maca_compiler}")
    else:
        print(
            "MACA_COMPILER/LMCACHE_MACA_COMPILER is not set; relying on "
            "torch.utils.cpp_extension compiler discovery."
        )

    print(f"torch.utils.cpp_extension.CUDA_HOME: {cpp_extension.CUDA_HOME}")
    print(f"MACA include dirs: {include_dirs}")
    print(f"MACA library dirs: {library_dirs}")

    define_macros = [("USE_MACA", "1")]
    ext_modules = [
        cpp_extension.CUDAExtension(
            "lmcache.c_ops",
            sources=cuda_sources(),
            include_dirs=include_dirs,
            library_dirs=library_dirs,
            define_macros=define_macros,
            extra_compile_args={
                "cxx": [flag_cxx_abi, "-DUSE_MACA"],
                "nvcc": [flag_cxx_abi, "-DUSE_MACA"],
            },
        ),
        cpp_extension.CppExtension(
            "lmcache.native_storage_ops",
            sources=storage_manager_sources(),
            include_dirs=["csrc/storage_manager"],
            extra_compile_args={
                "cxx": [flag_cxx_abi, "-O3"],
            },
        ),
    ]
    cmdclass = {"build_ext": cpp_extension.BuildExtension}
    return ext_modules, cmdclass


def rocm_extension() -> tuple[list, dict]:
    # Third Party
    from torch.utils import cpp_extension  # Import here

    print("Building ROCM extensions")
    hipify_wrapper()
    hip_sources = [
        "csrc/pybind_hip.cpp",  # Use the hipified pybind
        "csrc/mem_kernels.hip",
        "csrc/cal_cdf.hip",
        "csrc/ac_enc.hip",
        "csrc/ac_dec.hip",
        "csrc/pos_kernels.hip",
        "csrc/mem_alloc_hip.cpp",
        "csrc/utils_hip.cpp",
    ]
    # For HIP, we generally use CppExtension and let hipcc handle things.
    # Ensure CXX environment variable is set to hipcc when running this build.
    # e.g., CXX=hipcc python setup.py install
    define_macros = [("__HIP_PLATFORM_HCC__", "1"), ("USE_ROCM", "1")]
    ext_modules = [
        cpp_extension.CppExtension(
            "lmcache.c_ops",
            sources=hip_sources,
            extra_compile_args={
                "cxx": [  # hipcc is typically invoked as a C++ compiler
                    # '-D_GLIBCXX_USE_CXX11_ABI=0',
                    "-O3"
                    # Add any HIP specific flags if needed.
                    # For example, if you need to specify ROCm architecture:
                    # '--offload-arch=gfx942' # (replace with your target arch)
                    # '-x hip' # Sometimes needed to explicitly treat files as HIP
                ],
                # No 'nvcc' key for hipcc with CppExtension
            },
            # You might need to specify include paths for ROCm if not found
            # automatically
            include_dirs=[
                os.path.join(os.environ.get("ROCM_PATH", "/opt/rocm"), "include")
            ],
            library_dirs=[
                os.path.join(os.environ.get("ROCM_PATH", "/opt/rocm"), "lib")
            ],
            # libraries=['amdhip64'] # Or other relevant HIP libs if needed
            define_macros=define_macros,
        ),
        cpp_extension.CppExtension(
            "lmcache.native_storage_ops",
            sources=storage_manager_sources(),
            include_dirs=["csrc/storage_manager"],
            extra_compile_args={
                "cxx": ["-O3"],
            },
        ),
    ]
    cmdclass = {"build_ext": cpp_extension.BuildExtension}
    return ext_modules, cmdclass


def source_dist_extension() -> tuple[list, dict]:
    print("Not building CUDA/HIP/MACA extensions for sdist")
    return [], {}


if __name__ == "__main__":
    if BUILDING_SDIST:
        get_extension = source_dist_extension
    elif BUILD_WITH_MACA:
        get_extension = maca_extension
    elif BUILD_WITH_HIP:
        get_extension = rocm_extension
    else:
        get_extension = cuda_extension

    ext_modules, cmdclass = get_extension()

    setup(
        packages=find_packages(
            exclude=("csrc",)
        ),  # Ensure csrc is excluded if it only contains sources
        ext_modules=ext_modules,
        cmdclass=cmdclass,
        include_package_data=True,
    )
