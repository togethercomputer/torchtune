import torch
import time
import gc
import numpy as np
from typing import Tuple, List, Dict
import psutil
import os


def warmup_memory_allocation(L: int, E: int, R: int, C: int, r: int) -> None:
    """
    Warmup function to initialize GPU memory and ensure consistent benchmarking.
    """
    # Create some dummy tensors to warmup GPU memory allocation
    dummy_tensors = []
    for _ in range(3):
        dummy_a = torch.randn(E, R, r, device='cuda' if torch.cuda.is_available() else 'cpu')
        dummy_b = torch.randn(E, r, C, device='cuda' if torch.cuda.is_available() else 'cpu')
        dummy_tensors.extend([dummy_a, dummy_b])

    # Clean up
    del dummy_tensors
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def allocate_individual_adapters(L: int, E: int, R: int, C: int, r: int) -> List[
    List[Tuple[torch.Tensor, torch.Tensor]]]:
    """
    Allocate individual LoRA adapters A and B for each expert in each layer.

    Args:
        L: Number of layers
        E: Number of experts per layer
        R: Input dimension
        C: Output dimension
        r: Rank of LoRA adaptation

    Returns:
        List of layers, each containing list of (A, B) adapter pairs for each expert
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    layers = []

    for layer_idx in range(L):
        experts = []
        for expert_idx in range(E):
            # Allocate A matrix of shape (R, r) and B matrix of shape (r, C)
            A = torch.randn(R, r, device=device, dtype=torch.float32)
            B = torch.randn(r, C, device=device, dtype=torch.float32)
            experts.append((A, B))
        layers.append(experts)

    return layers


def allocate_3D_adapters(L: int, E: int, R: int, C: int, r: int) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Allocate two 3D adapters A and B per layer for all experts.

    Args:
        L: Number of layers
        E: Number of experts per layer
        R: Input dimension
        C: Output dimension
        r: Rank of LoRA adaptation

    Returns:
        List of layers, each containing (A, B) 3D adapter pair
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    layers = []

    for layer_idx in range(L):
        # Allocate 3D A matrix of shape (E, R, r) and 3D B matrix of shape (E, r, C)
        A = torch.randn(E, R, r, device=device, dtype=torch.float32)
        B = torch.randn(E, r, C, device=device, dtype=torch.float32)
        layers.append((A, B))

    return layers


def get_memory_usage() -> float:
    """Get current memory usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024 * 1024)  # Convert to MB
    else:
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024)  # Convert to MB


def benchmark_allocation_method(allocation_func, L: int, E: int, R: int, C: int, r: int, T: int = 10) -> Dict[
    str, float]:
    """
    Benchmark a specific allocation method.

    Args:
        allocation_func: Function to benchmark
        L, E, R, C, r: Parameters for allocation
        T: Number of trials

    Returns:
        Dictionary containing timing and memory statistics
    """
    times = []
    memory_usage = []

    for trial in range(T):
        # Clear memory before each trial
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        # Measure initial memory
        initial_memory = get_memory_usage()

        # Time the allocation
        start_time = time.perf_counter()

        result = allocation_func(L, E, R, C, r)

        # Ensure allocation is complete
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        end_time = time.perf_counter()

        # Measure final memory
        final_memory = get_memory_usage()

        times.append(end_time - start_time)
        memory_usage.append(final_memory - initial_memory)

        # Clean up result
        del result
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return {
        'mean_time': np.mean(times),
        'median_time': np.median(times),
        'std_time': np.std(times),
        'mean_memory': np.mean(memory_usage),
        'median_memory': np.median(memory_usage),
        'std_memory': np.std(memory_usage)
    }


def run_comprehensive_benchmark():
    """Run comprehensive benchmarks with different parameter configurations."""

    # Test configurations
    configs = [
        {'L': 4, 'E': 8, 'R': 512, 'C': 512, 'r': 16},
        {'L': 8, 'E': 16, 'R': 1024, 'C': 1024, 'r': 32},
        {'L': 12, 'E': 32, 'R': 2048, 'C': 2048, 'r': 64},
        {'L': 6, 'E': 64, 'R': 1024, 'C': 1024, 'r': 32},
    ]

    T = 20  # Number of trials per configuration

    print("LoRA Adapter Memory Allocation Benchmark")
    print("=" * 60)
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print(f"Number of trials per configuration: {T}")
    print()

    for i, config in enumerate(configs):
        L, E, R, C, r = config['L'], config['E'], config['R'], config['C'], config['r']

        print(f"Configuration {i + 1}: L={L}, E={E}, R={R}, C={C}, r={r}")
        print(f"Total parameters per method: {L * E * (R * r + r * C):,}")
        print("-" * 40)

        # Warmup
        warmup_memory_allocation(L, E, R, C, r)

        # Benchmark individual adapters
        print("Benchmarking individual adapters...")
        individual_stats = benchmark_allocation_method(
            allocate_individual_adapters, L, E, R, C, r, T
        )

        # Benchmark 3D adapters
        print("Benchmarking 3D adapters...")
        tensor_3d_stats = benchmark_allocation_method(
            allocate_3D_adapters, L, E, R, C, r, T
        )

        # Display results
        print("\nResults:")
        print(f"Individual Adapters:")
        print(
            f"  Time: {individual_stats['mean_time']:.6f}s ± {individual_stats['std_time']:.6f}s (median: {individual_stats['median_time']:.6f}s)")
        print(f"  Memory: {individual_stats['mean_memory']:.2f}MB ± {individual_stats['std_memory']:.2f}MB")

        print(f"3D Adapters:")
        print(
            f"  Time: {tensor_3d_stats['mean_time']:.6f}s ± {tensor_3d_stats['std_time']:.6f}s (median: {tensor_3d_stats['median_time']:.6f}s)")
        print(f"  Memory: {tensor_3d_stats['mean_memory']:.2f}MB ± {tensor_3d_stats['std_memory']:.2f}MB")

        # Calculate speedup
        speedup = individual_stats['mean_time'] / tensor_3d_stats['mean_time']
        memory_ratio = individual_stats['mean_memory'] / tensor_3d_stats['mean_memory'] if tensor_3d_stats[
                                                                                               'mean_memory'] > 0 else 1.0

        print(f"\nPerformance Comparison:")
        print(f"  3D Adapters are {speedup:.2f}x {'faster' if speedup > 1 else 'slower'} than Individual Adapters")
        print(f"  Memory usage ratio (Individual/3D): {memory_ratio:.2f}x")
        print()
        print("=" * 60)
        print()


if __name__ == "__main__":
    run_comprehensive_benchmark()