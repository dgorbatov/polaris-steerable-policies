#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#

import torch

# ==============================================================================
# JIT COMPILATION SUPPORT
# ==============================================================================
# Try to import pre-compiled _C module, if not available, compile via JIT
# ==============================================================================

try:
    # Try to import pre-compiled extension (from setup.py build)
    from . import _simple_knn as _C
except ImportError:
    # If not available, compile via JIT on first import
    import os
    from pathlib import Path

    def _load_extension_jit():
        """JIT compile the CUDA extension if pre-built version not available."""
        import importlib.util
        from torch.utils.cpp_extension import load

        # Build directory
        cuda_ver = (
            torch.version.cuda.replace(".", "_") if torch.cuda.is_available() else "cpu"
        )
        build_dir = os.path.join(
            os.path.expanduser("~"),
            ".cache",
            "torch_extensions",
            f"simple_knn_cu{cuda_ver}",
        )
        so_name = "simple_knn_cuda"
        so_path = os.path.join(build_dir, f"{so_name}.so")

        # If pre-compiled .so already exists, load it directly without
        # re-triggering ninja (torch.utils.cpp_extension.load always re-runs
        # ninja even when the .so is up-to-date, which requires nvcc).
        if os.path.exists(so_path):
            spec = importlib.util.spec_from_file_location(so_name, so_path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod

        # .so not found — compile via JIT (requires nvcc / CUDA_HOME)
        _pkg_path = Path(__file__).parent
        _src_path = _pkg_path.parent  # repo root where .cu files are

        sources = []
        for f in [_src_path / "ext.cpp", _src_path / "spatial.cu", _src_path / "simple_knn.cu"]:
            if f.exists():
                sources.append(str(f))
        for ext in ["*.cu", "*.cpp"]:
            for p in _src_path.rglob(ext):
                p_str = str(p)
                if p_str not in sources and "test" not in p_str.lower():
                    sources.append(p_str)

        if not sources:
            raise FileNotFoundError(
                f"No source files found in {_src_path}. "
                "Make sure simple-knn is properly installed."
            )

        extra_cuda_cflags = ["-O3", "--use_fast_math", "-std=c++17", "--expt-relaxed-constexpr"]
        extra_cflags = ["-O3", "-std=c++17"]
        include_dirs = [str(_src_path)]

        os.makedirs(build_dir, exist_ok=True)

        print("\n" + "=" * 70)
        print("Compiling simple-knn (first time only)...")
        print("This will take 1-2 minutes.")
        print("=" * 70 + "\n")

        try:
            extension = load(
                name=so_name,
                sources=sources,
                extra_cflags=extra_cflags,
                extra_cuda_cflags=extra_cuda_cflags,
                extra_include_paths=include_dirs,
                build_directory=build_dir,
                verbose=True,
                with_cuda=True,
            )
            print("\n✓ Compilation successful! Cached for future use.\n")
            return extension

        except Exception as e:
            print("\n" + "=" * 70)
            print("ERROR: Failed to compile simple-knn")
            print("=" * 70)
            print(f"\n{e}\n")
            print("Requirements:")
            print("  - CUDA toolkit installed")
            print("  - Compatible C++ compiler (gcc 7-12)")
            print("  - PyTorch with CUDA support")
            print("=" * 70 + "\n")
            raise

    # Load via JIT
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA not available. simple-knn requires CUDA.\n"
            f"PyTorch version: {torch.__version__}"
        )

    _C = _load_extension_jit()

# ==============================================================================
# Export the distCUDA2 function (main API)
# ==============================================================================


def distCUDA2(points):
    """
    Compute KNN distances for points using CUDA.

    Args:
        points: Tensor of shape (N, 3) containing 3D points

    Returns:
        Tensor of shape (N,) containing squared distances to nearest neighbors
    """
    return _C.distCUDA2(points)
