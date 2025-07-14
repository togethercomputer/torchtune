import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed._tensor import DTensor, DeviceMesh, Shard, Replicate
from torch.distributed.tensor.parallel import parallelize_module
import time
import gc
import numpy as np
from typing import Tuple, List, Dict, Optional, Union
import psutil
import os
import functools
import argparse


def setup_distributed(rank: int, world_size: int, backend: str = 'nccl'):
    """Initialize distributed training setup."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    if backend == 'nccl' and not torch.cuda.is_available():
        backend = 'gloo'

    dist.init_process_group(backend, rank=rank, world_size=world_size)

    if torch.cuda.is_available():
        torch.cuda.set_device(rank % torch.cuda.device_count())


def cleanup_distributed():
    """Clean up distributed training."""
    dist.destroy_process_group()


def create_device_mesh(world_size: int) -> DeviceMesh:
    """Create a device mesh for FSDP-2."""
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    return DeviceMesh(device_type, list(range(world_size)))


def warmup_memory_allocation(L: int, E: int, R: int, C: int, r: int, device_mesh: DeviceMesh) -> None:
    """
    Warmup function to initialize distributed memory allocation.
    """
    device = device_mesh.device_type

    # Create dummy 3D tensors with sharding
    dummy_a = torch.randn(E, R, r, device=device)
    dummy_b = torch.randn(E, r, C, device=device)

    # Create distributed tensors for warmup
    dtensor_a = DTensor.from_local(dummy_a, device_mesh, [Replicate(), Shard(1), Replicate()])
    dtensor_b = DTensor.from_local(dummy_b, device_mesh, [Replicate(), Shard(1), Replicate()])

    # Create dummy 2D tensors
    dummy_2d = torch.randn(R, r, device=device)
    dtensor_2d = DTensor.from_local(dummy_2d, device_mesh, [Shard(0), Replicate()])

    # Clean up
    del dummy_a, dummy_b, dummy_2d, dtensor_a, dtensor_b, dtensor_2d
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def allocate_individual_adapters_fsdp2(L: int, E: int, R: int, C: int, r: int, device_mesh: DeviceMesh) -> List[
    List[Tuple[DTensor, DTensor]]]:
    """
    Allocate individual LoRA adapters with FSDP-2 sharding on first dimension.

    Args:
        L: Number of layers
        E: Number of experts per layer
        R: Input dimension
        C: Output dimension
        r: Rank of LoRA adaptation
        device_mesh: Device mesh for distributed tensor placement

    Returns:
        List of layers, each containing list of (A, B) distributed adapter pairs
    """
    device = device_mesh.device_type
    layers = []

    for layer_idx in range(L):
        experts = []
        for expert_idx in range(E):
            # Create local tensors
            A_local = torch.randn(R, r, device=device, dtype=torch.float32)
            B_local = torch.randn(r, C, device=device, dtype=torch.float32)

            # Create distributed tensors with sharding on first dimension (rows)
            # A matrix: shard on rows (dimension 0)
            A_dtensor = DTensor.from_local(A_local, device_mesh, [Shard(0), Replicate()])
            # B matrix: shard on rows (dimension 0)
            B_dtensor = DTensor.from_local(B_local, device_mesh, [Shard(0), Replicate()])

            experts.append((A_dtensor, B_dtensor))
        layers.append(experts)

    return layers


def allocate_3D_adapters_fsdp2(L: int, E: int, R: int, C: int, r: int, device_mesh: DeviceMesh) -> List[
    Tuple[DTensor, DTensor]]:
    """
    Allocate 3D LoRA adapters with FSDP-2 sharding on second dimension.

    Args:
        L: Number of layers
        E: Number of experts per layer
        R: Input dimension
        C: Output dimension
        r: Rank of LoRA adaptation
        device_mesh: Device mesh for distributed tensor placement

    Returns:
        List of layers, each containing (A, B) 3D distributed adapter pair
    """
    device = device_mesh.device_type
    layers = []

    for layer_idx in range(L):
        # Create local 3D tensors
        A_local = torch.randn(E, R, r, device=device, dtype=torch.float32)
        B_local = torch.randn(E, r, C, device=device, dtype=torch.float32)

        # Create distributed tensors with sharding on second dimension
        # A matrix (E, R, r): shard on R dimension (dimension 1)
        A_dtensor = DTensor.from_local(A_local, device_mesh, [Replicate(), Shard(1), Replicate()])
        # B matrix (E, r, C): shard on r dimension (dimension 1)
        B_dtensor = DTensor.from_local(B_local, device_mesh, [Replicate(), Shard(1), Replicate()])

        layers.append((A_dtensor, B_dtensor))

    return layers


def get_distributed_memory_usage() -> float:
    """Get current distributed memory usage in MB."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        return torch.cuda.memory_allocated() / (1024 * 1024)
    else:
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024)


def benchmark_allocation_method_fsdp2(allocation_func, L: int, E: int, R: int, C: int, r: int,
                                      device_mesh: DeviceMesh, T: int = 10) -> Dict[str, float]:
    """
    Benchmark a specific allocation method with FSDP-2.

    Args:
        allocation_func: Function to benchmark
        L, E, R, C, r: Parameters for allocation
        device_mesh: Device mesh for distributed operations
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

        # Synchronize all processes
        dist.barrier()

        # Measure initial memory
        initial_memory = get_distributed_memory_usage()

        # Time the allocation
        start_time = time.perf_counter()

        result = allocation_func(L, E, R, C, r, device_mesh)

        # Ensure allocation is complete across all processes
        dist.barrier()
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        end_time = time.perf_counter()

        # Measure final memory
        final_memory = get_distributed_memory_usage()

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


def run_distributed_benchmark(rank: int, world_size: int):
    """Run distributed benchmarks on a single process."""

    # Setup distributed environment
    setup_distributed(rank, world_size)

    # Create device mesh for FSDP-2
    device_mesh = create_device_mesh(world_size)

    # Test configurations
    configs = [
        {'L': 4, 'E': 8, 'R': 512, 'C': 512, 'r': 16},
        {'L': 8, 'E': 16, 'R': 1024, 'C': 1024, 'r': 32},
        {'L': 12, 'E': 32, 'R': 2048, 'C': 2048, 'r': 64},
        {'L': 6, 'E': 64, 'R': 1024, 'C': 1024, 'r': 32},
    ]

    T = 15  # Number of trials per configuration

    if rank == 0:
        print("LoRA Adapter FSDP-2 Distributed Memory Allocation Benchmark")
        print("=" * 70)
        print(f"World Size: {world_size}")
        print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
        print(f"Number of trials per configuration: {T}")
        print()

    for i, config in enumerate(configs):
        L, E, R, C, r = config['L'], config['E'], config['R'], config['C'], config['r']

        if rank == 0:
            print(f"Configuration {i + 1}: L={L}, E={E}, R={R}, C={C}, r={r}")
            print(f"Total parameters per method: {L * E * (R * r + r * C):,}")
            print(f"Parameters per rank: {L * E * (R * r + r * C) // world_size:,}")
            print("-" * 50)

        # Warmup
        warmup_memory_allocation(L, E, R, C, r, device_mesh)

        # Benchmark individual adapters
        if rank == 0:
            print("Benchmarking individual adapters with FSDP-2...")

        individual_stats = benchmark_allocation_method_fsdp2(
            allocate_individual_adapters_fsdp2, L, E, R, C, r, device_mesh, T
        )

        # Benchmark 3D adapters
        if rank == 0:
            print("Benchmarking 3D adapters with FSDP-2...")

        tensor_3d_stats = benchmark_allocation_method_fsdp2(
            allocate_3D_adapters_fsdp2, L, E, R, C, r, device_mesh, T
        )

        # Gather results from all ranks
        all_individual_times = [None] * world_size
        all_3d_times = [None] * world_size
        all_individual_memory = [None] * world_size
        all_3d_memory = [None] * world_size

        dist.all_gather_object(all_individual_times, individual_stats['mean_time'])
        dist.all_gather_object(all_3d_times, tensor_3d_stats['mean_time'])
        dist.all_gather_object(all_individual_memory, individual_stats['mean_memory'])
        dist.all_gather_object(all_3d_memory, tensor_3d_stats['mean_memory'])

        # Display results from rank 0
        if rank == 0:
            print(f"\nResults (averaged across {world_size} ranks):")

            avg_individual_time = np.mean(all_individual_times)
            avg_3d_time = np.mean(all_3d_times)
            avg_individual_memory = np.mean(all_individual_memory)
            avg_3d_memory = np.mean(all_3d_memory)

            print(f"Individual Adapters (2D, sharded on rows):")
            print(f"  Time: {avg_individual_time:.6f}s ± {np.std(all_individual_times):.6f}s")
            print(f"  Memory per rank: {avg_individual_memory:.2f}MB ± {np.std(all_individual_memory):.2f}MB")
            print(f"  Total memory: {avg_individual_memory * world_size:.2f}MB")

            print(f"3D Adapters (sharded on 2nd dimension):")
            print(f"  Time: {avg_3d_time:.6f}s ± {np.std(all_3d_times):.6f}s")
            print(f"  Memory per rank: {avg_3d_memory:.2f}MB ± {np.std(all_3d_memory):.2f}MB")
            print(f"  Total memory: {avg_3d_memory * world_size:.2f}MB")

            # Calculate speedup
            speedup = avg_individual_time / avg_3d_time if avg_3d_time > 0 else 1.0
            memory_ratio = avg_individual_memory / avg_3d_memory if avg_3d_memory > 0 else 1.0

            print(f"\nDistributed Performance Comparison:")
            print(f"  3D Adapters are {speedup:.2f}x {'faster' if speedup > 1 else 'slower'} than Individual Adapters")
            print(f"  Memory usage ratio (Individual/3D): {memory_ratio:.2f}x")
            print(f"  Communication efficiency: 3D adapters have better locality")
            print()
            print("=" * 70)
            print()

    # Cleanup
    cleanup_distributed()


def run_single_machine_benchmark(world_size: int = None):
    """Run benchmark on a single machine with multiple processes."""
    if world_size is None:
        world_size = torch.cuda.device_count() if torch.cuda.is_available() else 2

    print(f"Starting single machine benchmark with {world_size} processes...")

    # Use torch.multiprocessing to spawn processes
    mp.spawn(run_distributed_benchmark, args=(world_size,), nprocs=world_size, join=True)


def run_multi_machine_benchmark():
    """
    Run benchmark across multiple machines.
    This function should be called on each machine with appropriate environment variables set.
    """
    # These should be set via environment variables or configuration
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))

    print(f"Starting multi-machine benchmark: rank {rank}/{world_size}")

    # For multi-machine, you need to set MASTER_ADDR and MASTER_PORT appropriately
    # Example environment setup for multi-machine:
    # export MASTER_ADDR=<IP_OF_RANK_0_MACHINE>
    # export MASTER_PORT=12355
    # export RANK=<CURRENT_MACHINE_RANK>
    # export WORLD_SIZE=<TOTAL_NUMBER_OF_MACHINES>

    run_distributed_benchmark(rank, world_size)


class DistributedLoRABenchmark:
    """Class to manage distributed LoRA adapter benchmarking."""

    def __init__(self, world_size: int = None):
        self.world_size = world_size or (torch.cuda.device_count() if torch.cuda.is_available() else 2)

    def run_single_machine(self):
        """Run benchmark on single machine."""
        run_single_machine_benchmark(self.world_size)

    def run_multi_machine(self):
        """Run benchmark across multiple machines."""
        run_multi_machine_benchmark()

    @staticmethod
    def get_optimal_sharding_config(L: int, E: int, R: int, C: int, r: int, world_size: int) -> Dict[str, any]:
        """
        Get optimal sharding configuration recommendations.
        """
        total_params_2d = L * E * (R * r + r * C)

        return {
            'total_parameters': total_params_2d,
            'parameters_per_rank': total_params_2d // world_size,
            'recommended_method': '3D' if E * R > world_size else '2D',
            'memory_per_rank_mb': (total_params_2d * 4) / (world_size * 1024 * 1024),  # 4 bytes per float32
            'sharding_efficiency': min(1.0, (E * R) / world_size),
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run distributed LoRA adapter benchmark')
    parser.add_argument('--mode', choices=['single', 'multi'], default='single', help='Run on single machine or multiple machines')
    parser.add_argument('--world-size', type=int, default=None, help='Number of processes/machines to use')
    args = parser.parse_args()

    benchmark = DistributedLoRABenchmark(args.world_size)

    if args.mode == 'single':
        benchmark.run_single_machine()
    else:
        benchmark.run_multi_machine()

    # Example usage for getting optimal configuration
    if dist.is_initialized() and dist.get_rank() == 0:
        config = DistributedLoRABenchmark.get_optimal_sharding_config(
            L=8, E=32, R=1024, C=1024, r=32, world_size=8
        )
        print("\nOptimal Configuration Recommendation:")
        for key, value in config.items():
            print(f"  {key}: {value}")