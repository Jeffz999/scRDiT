import torch
import torch.nn.functional as F
import time

def benchmark_attention(name, attention_fn, q, k, v, num_repeats=100):
    """Benchmarks the given attention function."""
    # Warm-up
    for _ in range(10):
        _ = attention_fn(q, k, v)
    torch.cuda.synchronize()

    start_time = time.time()
    for _ in range(num_repeats):
        _ = attention_fn(q, k, v)
    torch.cuda.synchronize()
    end_time = time.time()

    print(f"{name} | Average execution time: {(end_time - start_time) / num_repeats:.6f} seconds")
    # A simple way to observe memory is to use torch.cuda.max_memory_allocated()
    # before and after the benchmark loop, but for simplicity, this is omitted here.


def main():
    if not torch.cuda.is_available():
        print("CUDA is not available. This test requires a GPU.")
        return

    device = torch.device("cuda")
    dtype = torch.float16 if torch.cuda.get_device_capability()[0] >= 7 else torch.float32
    if dtype == torch.float32:
        print("Warning: Flash Attention 2 requires a GPU with compute capability >= 8.0 and float16/bfloat16 dtype.")


    batch_size = 8
    num_heads = 12
    seq_length = 4096
    head_dim = 64

    # Create dummy input tensors
    query = torch.randn(batch_size, num_heads, seq_length, head_dim, device=device, dtype=dtype)
    key = torch.randn(batch_size, num_heads, seq_length, head_dim, device=device, dtype=dtype)
    value = torch.randn(batch_size, num_heads, seq_length, head_dim, device=device, dtype=dtype)

    print("--- Verifying Flash Attention 2 Usage ---")

    # 1. Check if Flash Attention is enabled by default (should be False)
    print(f"Flash Attention enabled by default: {torch.backends.cuda.flash_sdp_enabled()}")

    # 2. Use the context manager to enable Flash Attention
    with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False) as a:
        print(f"Inside context manager: Flash Attention enabled: {torch.backends.cuda.flash_sdp_enabled()}")
        is_flash_in_use = F.scaled_dot_product_attention(query, key, value).is_contiguous()
        # The specific kernel used is not directly exposed, but we can infer
        # its use through the performance benchmark.


    # 3. Check if Flash Attention is enabled after the context (should be False)
    print(f"After context manager: Flash Attention enabled: {torch.backends.cuda.flash_sdp_enabled()}\n")


    print("--- Benchmarking Attention Implementations ---")

    # Benchmark Standard Attention
    def standard_attention(q, k, v):
        # Manually implement scaled dot-product attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / (head_dim ** 0.5)
        attn_weights = F.softmax(attn_weights, dim=-1)
        return torch.matmul(attn_weights, v)

    benchmark_attention("Standard Attention", standard_attention, query, key, value)


    # Benchmark with Flash Attention 2 enabled
    def flash_attention_2(q, k, v):
         with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
              return F.scaled_dot_product_attention(q, k, v)

    if torch.cuda.get_device_capability()[0] >= 8:
        benchmark_attention("Flash Attention 2", flash_attention_2, query, key, value)
    else:
        print("Skipping Flash Attention 2 benchmark: GPU compute capability < 8.0")


if __name__ == "__main__":
    main()