#!/usr/bin/env python3
"""
OVMono3D Device Benchmark Tool
Supports: CUDA, CPU, Intel XPU (Arc GPU via upstream PyTorch)

Measures:
- Inference latency (mean, std, P50, P95, P99)
- Throughput (FPS)
- Model parameters and memory usage
- Optional FLOPS counting

Requirements for Intel XPU:
    PyTorch 2.4+ with XPU support (upstream)
    Intel GPU drivers installed
    See: https://pytorch.org/docs/stable/notes/get_start_xpu.html

Usage:
    # CUDA benchmark
    python benchmark_devices.py --device cuda --config configs/OVMono3D_dinov2_SFP.yaml \
        --weights checkpoints/ovmono3d_lift.pth

    # CPU benchmark
    python benchmark_devices.py --device cpu --config configs/OVMono3D_dinov2_SFP.yaml \
        --weights checkpoints/ovmono3d_lift.pth

    # Intel XPU benchmark (requires PyTorch 2.4+ with XPU support)
    python benchmark_devices.py --device xpu --config configs/OVMono3D_dinov2_SFP.yaml \
        --weights checkpoints/ovmono3d_lift.pth
"""

import os
import sys
import argparse
import logging
import time
import json
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import torch
from tqdm import tqdm

# Setup paths
sys.dont_write_bytecode = True
sys.path.append(os.getcwd())
sys.path.append('/opt/home/devuser/ovmono3d')

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import transforms as T

from cubercnn.config import get_cfg_defaults
from cubercnn.modeling.meta_arch import build_model
from cubercnn import util

# Try to import fvcore for FLOPS counting
try:
    from fvcore.nn import FlopCountAnalysis
    FVCORE_AVAILABLE = True
except ImportError:
    FVCORE_AVAILABLE = False
    logging.warning("fvcore not available. FLOPS counting disabled. Install: pip install fvcore")

# Check for Intel XPU support (upstream PyTorch 2.4+)
XPU_AVAILABLE = hasattr(torch, 'xpu') and torch.xpu.is_available()
if XPU_AVAILABLE:
    logging.info(f"Intel XPU detected: {torch.xpu.device_count()} device(s) available")
    logging.info(f"XPU Device 0: {torch.xpu.get_device_name(0)}")
else:
    logging.info("Intel XPU not available. For XPU support, install PyTorch 2.4+ with XPU: https://pytorch.org/docs/stable/notes/get_start_xpu.html")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class OVMono3DBenchmark:
    """Benchmark OVMono3D model across different devices"""
    
    def __init__(self, config_path: str, weights_path: str, device: str = 'cuda'):
        """
        Initialize benchmark
        
        Args:
            config_path: Path to model config file
            weights_path: Path to model weights
            device: Device to use ('cuda', 'cpu', 'xpu', 'cuda:0', etc.)
        """
        self.device = self._setup_device(device)
        self.device_type = device.split(':')[0]  # Extract base device type
        
        logger.info(f"Initializing OVMono3D on device: {self.device}")
        
        # Setup configuration
        self.cfg = self._setup_config(config_path, weights_path)
        
        # Build and load model
        self.model = self._load_model()
        
        # Setup augmentations
        self.augmentations = self._setup_transforms()
        
        # Count parameters
        self.model_params = self._count_parameters()
        
        logger.info(f"Model loaded successfully on {self.device}")
        logger.info(f"Parameters: {self.model_params['total']:.2f}M ({self.model_params['trainable']:.2f}M trainable)")
        logger.info(f"Model size: {self.model_params['total_mb']:.1f} MB")
    
    def _setup_device(self, device: str) -> str:
        """Setup and validate device"""
        device_lower = device.lower()
        
        if device_lower.startswith('cuda'):
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA requested but not available")
            # Set specific GPU if specified
            if ':' in device_lower:
                gpu_id = int(device_lower.split(':')[1])
                torch.cuda.set_device(gpu_id)
            logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
            return device_lower
            
        elif device_lower == 'cpu':
            logger.info("Using CPU device")
            return 'cpu'
            
        elif device_lower == 'xpu':
            # Check XPU availability (upstream PyTorch 2.4+)
            if not XPU_AVAILABLE:
                raise RuntimeError(
                    "XPU requested but not available. "
                    "Ensure you have:\n"
                    "  1. PyTorch 2.4+ with XPU support installed\n"
                    "  2. Intel GPU drivers installed\n"
                    "  See: https://pytorch.org/docs/stable/notes/get_start_xpu.html"
                )
            
            # Set specific XPU device if specified
            if ':' in device:
                xpu_id = int(device.split(':')[1])
                logger.info(f"Using XPU device {xpu_id}: {torch.xpu.get_device_name(xpu_id)}")
                return f'xpu:{xpu_id}'
            else:
                logger.info(f"Using XPU device 0: {torch.xpu.get_device_name(0)}")
                return 'xpu'
        else:
            raise ValueError(f"Unknown device: {device}. Use 'cuda', 'cpu', or 'xpu'")
    
    def _setup_config(self, config_path: str, weights_path: str):
        """Setup model configuration"""
        cfg = get_cfg()
        get_cfg_defaults(cfg)
        
        # Handle remote config files
        if config_path.startswith(util.CubeRCNNHandler.PREFIX):
            config_path = util.CubeRCNNHandler._get_local_path(util.CubeRCNNHandler, config_path)
        
        cfg.merge_from_file(config_path)
        cfg.MODEL.WEIGHTS = weights_path
        cfg.MODEL.DEVICE = self.device
        cfg.freeze()
        
        return cfg
    
    def _load_model(self):
        """Load and prepare model"""
        model = build_model(self.cfg)
        
        # Load weights
        DetectionCheckpointer(model).resume_or_load(self.cfg.MODEL.WEIGHTS, resume=True)
        
        # Move to device
        model.to(self.device)
        
        # Note: Upstream PyTorch handles XPU optimizations automatically
        # No need for explicit IPEX optimization calls
        
        model.eval()
        return model
    
    def _setup_transforms(self):
        """Setup image augmentations"""
        min_size = self.cfg.INPUT.MIN_SIZE_TEST
        max_size = self.cfg.INPUT.MAX_SIZE_TEST
        return T.AugmentationList([
            T.ResizeShortestEdge(min_size, max_size, "choice")
        ])
    
    def _count_parameters(self) -> Dict:
        """Count model parameters and size"""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        # Calculate model size
        param_size = sum(p.nelement() * p.element_size() for p in self.model.parameters())
        buffer_size = sum(b.nelement() * b.element_size() for b in self.model.buffers())
        size_mb = (param_size + buffer_size) / 1024**2
        
        return {
            'total': total_params / 1e6,
            'trainable': trainable_params / 1e6,
            'total_mb': size_mb,
            'total_gb': size_mb / 1024
        }
    
    def generate_random_images(self, batch_size: int, image_size: Tuple[int, int]) -> List[np.ndarray]:
        """Generate random synthetic images"""
        width, height = image_size
        images = []
        for _ in range(batch_size):
            img = np.random.randint(0, 256, size=(height, width, 3), dtype=np.uint8)
            images.append(img)
        return images
    
    def preprocess_images(self, images: List[np.ndarray], categories: List[str]) -> List[Dict]:
        """Preprocess images for inference"""
        batch_inputs = []
        
        for img in images:
            # Apply augmentations
            aug_input = T.AugInput(img)
            _ = self.augmentations(aug_input)
            processed_image = aug_input.image
            
            # Create intrinsic matrix (dummy values for benchmark)
            h, w = img.shape[:2]
            focal_length = 4.0 * h / 2  # NDC focal length
            K = np.array([
                [focal_length, 0.0, w/2],
                [0.0, focal_length, h/2],
                [0.0, 0.0, 1.0]
            ])
            
            # Prepare input dict
            input_dict = {
                'image': torch.as_tensor(
                    np.ascontiguousarray(processed_image.transpose(2, 0, 1))
                ).to(self.device),
                'height': img.shape[0],
                'width': img.shape[1],
                'K': K,
                'category_list': categories
            }
            batch_inputs.append(input_dict)
        
        return batch_inputs
    
    def synchronize(self):
        """Synchronize device for accurate timing"""
        if self.device_type == 'cuda':
            torch.cuda.synchronize()
        elif self.device_type == 'xpu' and hasattr(torch, 'xpu'):
            torch.xpu.synchronize()
        # CPU doesn't need synchronization
    
    def benchmark(self, batch_sizes: List[int] = [1, 2, 4, 8],
                  num_iterations: int = 100, warmup_iterations: int = 10,
                  image_size: Tuple[int, int] = (1242, 375),
                  categories: List[str] = None,
                  compute_flops: bool = True) -> Dict:
        """
        Run comprehensive benchmark
        
        Args:
            batch_sizes: List of batch sizes to test
            num_iterations: Number of iterations per batch size
            warmup_iterations: Number of warmup iterations
            image_size: (width, height) of generated images
            categories: List of categories to detect
            compute_flops: Whether to compute FLOPS
            
        Returns:
            Dictionary with benchmark results
        """
        if categories is None:
            categories = ['car', 'truck', 'bus', 'person', 'bicycle']
        
        logger.info("="*60)
        logger.info(f"BENCHMARK CONFIGURATION")
        logger.info("="*60)
        logger.info(f"Device: {self.device}")
        logger.info(f"Image size: {image_size}")
        logger.info(f"Batch sizes: {batch_sizes}")
        logger.info(f"Iterations: {num_iterations} (warmup: {warmup_iterations})")
        logger.info(f"Categories: {categories}")
        logger.info("="*60)
        
        results = {
            'device': self.device,
            'device_type': self.device_type,
            'image_size': image_size,
            'num_iterations': num_iterations,
            'warmup_iterations': warmup_iterations,
            'model_params': self.model_params,
            'batch_results': {}
        }
        
        # Compute FLOPS if requested
        if compute_flops and FVCORE_AVAILABLE:
            try:
                logger.info("\nComputing FLOPS...")
                flops_result = self._compute_flops(image_size, categories)
                results['flops'] = flops_result
                logger.info(f"FLOPS: {flops_result['gflops']:.2f} GFLOPs ({flops_result['gmacs']:.2f} GMACs)")
            except Exception as e:
                logger.warning(f"FLOPS computation failed: {e}")
                results['flops'] = None
        else:
            results['flops'] = None
        
        # Benchmark each batch size
        for batch_size in batch_sizes:
            logger.info(f"\n{'='*60}")
            logger.info(f"BENCHMARKING BATCH SIZE: {batch_size}")
            logger.info(f"{'='*60}")
            
            batch_result = self._benchmark_batch_size(
                batch_size, num_iterations, warmup_iterations,
                image_size, categories
            )
            
            results['batch_results'][batch_size] = batch_result
            
            # Print summary
            self._print_batch_summary(batch_size, batch_result)
        
        return results
    
    def _compute_flops(self, image_size: Tuple[int, int], categories: List[str]) -> Dict:
        """Compute FLOPS for the model"""
        width, height = image_size
        
        # Generate sample image
        img = np.random.randint(0, 256, size=(height, width, 3), dtype=np.uint8)
        
        # Preprocess
        aug_input = T.AugInput(img)
        _ = self.augmentations(aug_input)
        processed_image = aug_input.image
        
        # Create input
        focal_length = 4.0 * height / 2
        K = np.array([
            [focal_length, 0.0, width/2],
            [0.0, focal_length, height/2],
            [0.0, 0.0, 1.0]
        ])
        
        sample_input = [{
            'image': torch.as_tensor(
                np.ascontiguousarray(processed_image.transpose(2, 0, 1))
            ).to(self.device),
            'height': height,
            'width': width,
            'K': K,
            'category_list': categories
        }]
        
        # Compute FLOPS
        flop_counter = FlopCountAnalysis(self.model, (sample_input,))
        flops_total = flop_counter.total()
        
        return {
            'total_flops': int(flops_total),
            'gflops': flops_total / 1e9,
            'gmacs': flops_total / 2 / 1e9
        }
    
    def _benchmark_batch_size(self, batch_size: int, num_iterations: int,
                              warmup_iterations: int, image_size: Tuple[int, int],
                              categories: List[str]) -> Dict:
        """Benchmark a specific batch size"""
        
        # Generate random images
        logger.info(f"Generating {batch_size} random images...")
        images = self.generate_random_images(batch_size, image_size)
        
        # Preprocess
        logger.info("Preprocessing images...")
        batch_inputs = self.preprocess_images(images, categories)
        
        # Warmup
        logger.info(f"Running {warmup_iterations} warmup iterations...")
        for _ in range(warmup_iterations):
            with torch.no_grad():
                _ = self.model(batch_inputs)
            self.synchronize()
        
        # Benchmark
        logger.info(f"Running {num_iterations} benchmark iterations...")
        latencies = []
        
        for _ in tqdm(range(num_iterations-1), desc=f"Batch {batch_size}"):
            self.synchronize()
            start = time.time()
            
            with torch.no_grad():
                outputs = self.model(batch_inputs)
            
            self.synchronize()
            end = time.time()
            
            latencies.append(end - start)
        
        # Compute statistics
        latencies_ms = np.array(latencies) * 1000
        
        result = {
            'batch_size': batch_size,
            'latency_mean_ms': float(np.mean(latencies_ms)),
            'latency_std_ms': float(np.std(latencies_ms)),
            'latency_min_ms': float(np.min(latencies_ms)),
            'latency_max_ms': float(np.max(latencies_ms)),
            'latency_p50_ms': float(np.percentile(latencies_ms, 50)),
            'latency_p95_ms': float(np.percentile(latencies_ms, 95)),
            'latency_p99_ms': float(np.percentile(latencies_ms, 99)),
            'throughput_fps': batch_size / np.mean(latencies),
            'per_image_latency_ms': float(np.mean(latencies_ms) / batch_size),
            'per_image_fps': batch_size / np.mean(latencies_ms) * 1000
        }
        
        # Memory usage
        if self.device_type == 'cuda':
            result['memory_allocated_mb'] = torch.cuda.memory_allocated() / 1024**2
            result['memory_reserved_mb'] = torch.cuda.memory_reserved() / 1024**2
        elif self.device_type == 'xpu' and hasattr(torch, 'xpu'):
            try:
                result['memory_allocated_mb'] = torch.xpu.memory_allocated() / 1024**2
                result['memory_reserved_mb'] = torch.xpu.memory_reserved() / 1024**2
            except:
                pass
        
        return result
    
    def _print_batch_summary(self, batch_size: int, result: Dict):
        """Print summary for a batch size"""
        logger.info(f"\nBatch {batch_size} Results:")
        logger.info(f"  Latency (mean ± std): {result['latency_mean_ms']:.2f} ± {result['latency_std_ms']:.2f} ms")
        logger.info(f"  Latency (min/max): {result['latency_min_ms']:.2f} / {result['latency_max_ms']:.2f} ms")
        logger.info(f"  Latency (P50/P95/P99): {result['latency_p50_ms']:.2f} / {result['latency_p95_ms']:.2f} / {result['latency_p99_ms']:.2f} ms")
        logger.info(f"  Throughput: {result['throughput_fps']:.2f} FPS")
        logger.info(f"  Per-image: {result['per_image_latency_ms']:.2f} ms ({result['per_image_fps']:.2f} FPS)")
        
        if 'memory_allocated_mb' in result:
            logger.info(f"  Memory: {result['memory_allocated_mb']:.1f} MB allocated, {result['memory_reserved_mb']:.1f} MB reserved")
    
    def print_final_summary(self, results: Dict):
        """Print comprehensive benchmark summary"""
        print("\n" + "="*70)
        print(f"FINAL BENCHMARK SUMMARY - {self.device_type.upper()}")
        print("="*70)
        print(f"\nDevice: {results['device']}")
        print(f"Image Size: {results['image_size'][0]} x {results['image_size'][1]}")
        print(f"Iterations: {results['num_iterations']} (warmup: {results['warmup_iterations']})")
        print(f"\nModel Configuration:")
        print(f"  Parameters: {results['model_params']['total']:.2f}M total ({results['model_params']['trainable']:.2f}M trainable)")
        print(f"  Model Size: {results['model_params']['total_mb']:.1f} MB ({results['model_params']['total_gb']:.3f} GB)")
        
        if results['flops']:
            print(f"\nComputational Complexity:")
            print(f"  Total FLOPs: {results['flops']['total_flops']:,}")
            print(f"  GFLOPs: {results['flops']['gflops']:.2f}")
            print(f"  GMACs: {results['flops']['gmacs']:.2f}")
        
        print("\n" + "-"*70)
        print(f"{'Batch':<8} {'Latency (ms)':<20} {'Throughput':<15} {'Per-Image':<20} {'Memory (MB)':<15}")
        print(f"{'Size':<8} {'Mean ± Std':<20} {'(FPS)':<15} {'Latency (ms)':<20} {'Alloc / Rsrv':<15}")
        print("-"*70)
        
        for batch_size, result in sorted(results['batch_results'].items()):
            latency_str = f"{result['latency_mean_ms']:.2f} ± {result['latency_std_ms']:.2f}"
            throughput_str = f"{result['throughput_fps']:.2f}"
            per_image_str = f"{result['per_image_latency_ms']:.2f}"
            
            memory_str = "N/A"
            if 'memory_allocated_mb' in result:
                memory_str = f"{result['memory_allocated_mb']:.1f} / {result['memory_reserved_mb']:.1f}"
            
            print(f"{batch_size:<8} {latency_str:<20} {throughput_str:<15} {per_image_str:<20} {memory_str:<15}")
        
        print("-"*70)
        
        # Print detailed percentile statistics
        print("\nDetailed Latency Statistics (ms):")
        print("-"*70)
        print(f"{'Batch':<8} {'Min':<10} {'P50':<10} {'P95':<10} {'P99':<10} {'Max':<10}")
        print("-"*70)
        
        for batch_size, result in sorted(results['batch_results'].items()):
            print(f"{batch_size:<8} "
                  f"{result['latency_min_ms']:<10.2f} "
                  f"{result['latency_p50_ms']:<10.2f} "
                  f"{result['latency_p95_ms']:<10.2f} "
                  f"{result['latency_p99_ms']:<10.2f} "
                  f"{result['latency_max_ms']:<10.2f}")
        
        print("-"*70)
        
        # Print key takeaways
        print("\nKey Takeaways:")
        best_batch = max(results['batch_results'].items(), key=lambda x: x[1]['throughput_fps'])
        fastest_per_image = min(results['batch_results'].items(), key=lambda x: x[1]['per_image_latency_ms'])
        
        print(f"  • Best throughput: {best_batch[1]['throughput_fps']:.2f} FPS (batch size {best_batch[0]})")
        print(f"  • Fastest per-image: {fastest_per_image[1]['per_image_latency_ms']:.2f} ms (batch size {fastest_per_image[0]})")
        
        if results['flops']:
            # Calculate GFLOPS/s for batch size 1
            if 1 in results['batch_results']:
                gflops_per_sec = results['flops']['gflops'] / (results['batch_results'][1]['per_image_latency_ms'] / 1000)
                print(f"  • Computational throughput: {gflops_per_sec:.2f} GFLOPS/s (batch size 1)")
        
        print("="*70)


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark OVMono3D on different devices (CUDA, CPU, XPU)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # CUDA benchmark
  python benchmark_devices.py --device cuda --config configs/OVMono3D_dinov2_SFP.yaml \\
      --weights checkpoints/ovmono3d_lift.pth

  # CPU benchmark with reduced iterations
  python benchmark_devices.py --device cpu --config configs/OVMono3D_dinov2_SFP.yaml \\
      --weights checkpoints/ovmono3d_lift.pth --batch-sizes 1 2 \\
      --num-iterations 50

  # Intel XPU (Arc GPU) benchmark
  python benchmark_devices.py --device xpu --config configs/OVMono3D_dinov2_SFP.yaml \\
      --weights checkpoints/ovmono3d_lift.pth
        """
    )
    
    parser.add_argument('--config', required=True, help='Path to model config file')
    parser.add_argument('--weights', required=True, help='Path to model weights')
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu', 'xpu', 'cuda:0', 'cuda:1'],
                       help='Device to benchmark (default: cuda)')
    
    # Benchmark parameters
    parser.add_argument('--batch-sizes', nargs='+', type=int, default=[1],
                       help='Batch sizes to test (default: 1 2 4 8)')
    parser.add_argument('--num-iterations', type=int, default=100,
                       help='Number of iterations per batch size (default: 100)')
    parser.add_argument('--warmup-iterations', type=int, default=10,
                       help='Number of warmup iterations (default: 10)')
    parser.add_argument('--image-width', type=int, default=1242,
                       help='Width of generated images (default: 1242)')
    parser.add_argument('--image-height', type=int, default=375,
                       help='Height of generated images (default: 375)')
    parser.add_argument('--categories', nargs='+', default=['car', 'truck', 'bus', 'person', 'bicycle'],
                       help='Categories to detect (default: car truck bus person bicycle)')
    parser.add_argument('--no-flops', action='store_true',
                       help='Disable FLOPS counting')
    
    args = parser.parse_args()
    
    # Initialize benchmark
    benchmark = OVMono3DBenchmark(
        config_path=args.config,
        weights_path=args.weights,
        device=args.device
    )
    
    # Run benchmark
    results = benchmark.benchmark(
        batch_sizes=args.batch_sizes,
        num_iterations=args.num_iterations,
        warmup_iterations=args.warmup_iterations,
        image_size=(args.image_width, args.image_height),
        categories=args.categories,
        compute_flops=not args.no_flops
    )
    
    # Print final summary
    benchmark.print_final_summary(results)
    
    logger.info("\n" + "="*60)
    logger.info("BENCHMARK COMPLETED SUCCESSFULLY")
    logger.info("="*60)


if __name__ == '__main__':
    main()
