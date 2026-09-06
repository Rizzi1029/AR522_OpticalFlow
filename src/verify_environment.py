import torch


def verify_environment():
    print(f"PyTorch Version: {torch.__version__}")
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_available}")

    if cuda_available:
        gpu_name = torch.cuda.get_device_name(0)
        print(f"GPU Name: {gpu_name}")

        # Create a small tensor on the GPU and perform a simple operation
        tensor_a = torch.tensor([1.0, 2.0, 3.0, 4.0], device="cuda")
        tensor_b = tensor_a * 2.0 + 1.0
        torch.cuda.synchronize()
        print(f"CUDA Computation Result: {tensor_b.tolist()}")
        print("Success: PyTorch and CUDA environment verified successfully!")
    else:
        print("Notice: CUDA is not available on this device.")
        print("Success: PyTorch environment verified (CPU only).")


if __name__ == "__main__":
    verify_environment()
