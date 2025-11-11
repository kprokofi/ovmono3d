#!/bin/bash

# Quick Start Guide for OVMono3D Benchmarking
# This script provides a guided setup and benchmark process

set -e

echo "======================================================================="
echo "OVMono3D Multi-Device Benchmark - Quick Start"
echo "======================================================================="
echo ""

# Step 1: Check devices
echo "Step 1: Checking available devices..."
echo "-----------------------------------------------------------------------"
python check_devices.py

echo ""
read -p "Press Enter to continue with benchmarking..."

# Step 2: Ask user which device to benchmark
echo ""
echo "Step 2: Select device to benchmark"
echo "-----------------------------------------------------------------------"
echo "Available options:"
echo "  1) CUDA (NVIDIA GPU)"
echo "  2) CPU"
echo "  3) XPU (Intel Arc GPU)"
echo "  4) All devices (recommended)"
echo ""
read -p "Enter your choice (1-4): " choice

# Configuration
CONFIG="/home/kprokofi/3d_object_detection/ovmono3d/configs/OVMono3D_dinov2_SFP.yaml"
WEIGHTS="/home/kprokofi/3d_object_detection/ovmono3d/checkpoints/ovmono3d_lift.pth"

# Check if config and weights exist
if [ ! -f "$CONFIG" ]; then
    echo "Error: Config file not found: $CONFIG"
    echo "Please update the CONFIG variable in this script"
    exit 1
fi

if [ ! -f "$WEIGHTS" ]; then
    echo "Error: Weights file not found: $WEIGHTS"
    echo "Please update the WEIGHTS variable in this script"
    exit 1
fi

case $choice in
    1)
        echo ""
        echo "Benchmarking CUDA device..."
        python benchmark_devices.py \
            --device cuda \
            --config "$CONFIG" \
            --weights "$WEIGHTS" \
            --no-flops 
        echo ""
        echo "Results saved to: benchmark_cuda/"
        ;;
    2)
        echo ""
        echo "Benchmarking CPU (reduced iterations for speed)..."
        python benchmark_devices.py \
            --device cpu \
            --config "$CONFIG" \
            --weights "$WEIGHTS" \
            --num-iterations 50 \
            --no-flops
        echo ""
        echo "Results saved to: benchmark_cpu/"
        ;;
    3)
        echo ""
        echo "Benchmarking Intel XPU..."
        python benchmark_devices.py \
            --device xpu \
            --config "$CONFIG" \
            --weights "$WEIGHTS" \
            --no-flops
        echo ""
        echo "Results saved to: benchmark_xpu/"
        ;;
    4)
        echo ""
        echo "Benchmarking all available devices..."
        ./run_all_device_benchmarks.sh
        echo ""
        echo "Results saved to: benchmark_results_all_devices/"
        ;;
    *)
        echo "Invalid choice. Exiting."
        exit 1
        ;;
esac

echo ""
echo "======================================================================="
echo "Benchmark Complete!"
echo "======================================================================="
echo ""
echo "Next steps:"
echo "  1. Review the JSON results for detailed metrics"
echo "  2. Check the summary .txt file for human-readable results"
echo "  3. If you ran all devices, see comparison_summary.txt"
echo "  4. Integrate results into your technical report"
echo ""
echo "For more options, see: BENCHMARK_README.md"
echo "======================================================================="
