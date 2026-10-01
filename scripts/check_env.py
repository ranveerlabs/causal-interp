import argparse
import sys
from importlib.metadata import version


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--require-arch', action='append', default=[])
    parser.add_argument('--metadata-only', action='store_true')
    args = parser.parse_args()
    try:
        import torch
        import transformer_lens
        print('torch', torch.__version__)
        print('CUDA', torch.version.cuda)
        print('transformer_lens', version('transformer_lens'))
        arch = torch.cuda.get_arch_list()
        print('architectures', arch)
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable')
        if any(a not in arch for a in args.require_arch):
            raise RuntimeError('required architecture missing')
        compatible = []
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            cap = f'sm_{p.major}{p.minor}'
            print(i, p.name, round(p.total_memory/1024**3, 2), 'GiB', cap)
            if cap in arch:
                compatible.append(i)
        if not compatible:
            raise RuntimeError('no visible device has a compiled architecture')
        if not args.metadata_only:
            for i in compatible:
                with torch.cuda.device(i):
                    x = torch.ones((32, 32), device=f'cuda:{i}', dtype=torch.float32)
                    if not torch.equal(x @ x, x * 32):
                        raise RuntimeError('fp32 CUDA matrix check failed')
                    torch.cuda.synchronize()
            print('small CUDA matrix check passed')
        return 0
    except Exception as e:
        print('ENVIRONMENT NOT READY:', e)
        return 1


if __name__ == '__main__':
    sys.exit(main())
