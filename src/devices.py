import sys
import torch

# /d:/GitHub/dl-ecg/src/devices.py
"""
Simple script to detect CUDA-capable GPUs using PyTorch.

Usage:
    python devices.py
"""

def _bytes_to_gb(b: int) -> str:
    return f"{b / (1024 ** 3):.2f} GB"


def main():
    print(f"PyTorch version: {torch.__version__}")
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")

    if not cuda_available:
        sys.exit(0)

    device_count = torch.cuda.device_count()
    print(f"CUDA device count: {device_count}")

    for i in range(device_count):
        try:
            props = torch.cuda.get_device_properties(i)
        except Exception as e:
            print(f"  [{i}] Error getting properties: {e}")
            continue

        current = "(current)" if torch.cuda.current_device() == i else ""
        name = props.name
        capability = f"{props.major}.{props.minor}"
        total_mem = _bytes_to_gb(props.total_memory)
        mp_count = getattr(props, "multi_processor_count", "N/A")

        print(f"  [{i}] {name} {current}")
        print(f"       Compute capability: {capability}")
        print(f"       Total memory: {total_mem}")
        print(f"       Multiprocessors: {mp_count}")

    # Example: print default device index and name
    try:
        default_idx = torch.cuda.current_device()
        print(f"Default CUDA device: {default_idx} - {torch.cuda.get_device_name(default_idx)}")
    except Exception:
        pass


if __name__ == "__main__":
    main()