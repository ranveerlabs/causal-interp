"""Verify the environment is ready: CUDA-enabled PyTorch, a visible GPU, transformer_lens.

Run after setup:
    python scripts/check_env.py
"""

import sys
from importlib.metadata import version


def main() -> int:
    ok = True

    try:
        import torch
    except ImportError:
        print("FAIL  torch is not installed")
        return 1

    print(f"torch            {torch.__version__}")
    print(f"cuda build       {torch.version.cuda or 'CPU-ONLY BUILD'}")

    if torch.version.cuda is None:
        print("FAIL  this is a CPU-only PyTorch build; reinstall from the cu128 index")
        ok = False

    if not torch.cuda.is_available():
        print("FAIL  torch.cuda.is_available() is False; the GPU will not be used")
        ok = False
    else:
        name = torch.cuda.get_device_name(0)
        major, minor = torch.cuda.get_device_capability(0)
        total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"gpu              {name}")
        print(f"compute cap      sm_{major}{minor}")
        print(f"vram             {total_gb:.1f} GiB")

        # Blackwell (sm_120) needs a cu128+ build; older wheels compile no matching kernels.
        arch_list = torch.cuda.get_arch_list()
        if f"sm_{major}{minor}" not in arch_list:
            print(f"FAIL  no kernels for sm_{major}{minor} in this build (has: {', '.join(arch_list)})")
            ok = False
        else:
            # An actual allocation is the only proof the kernels run.
            x = torch.randn(1024, 1024, device="cuda")
            torch.mm(x, x)
            torch.cuda.synchronize()
            print("matmul on cuda   OK")

    try:
        import transformer_lens  # noqa: F401
    except ImportError:
        print("FAIL  transformer_lens is not installed")
        ok = False
    else:
        # The package exposes no __version__ attribute; read the installed metadata.
        print(f"transformer_lens {version('transformer_lens')}")

    print("\n" + ("ENVIRONMENT OK" if ok else "ENVIRONMENT NOT READY"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
