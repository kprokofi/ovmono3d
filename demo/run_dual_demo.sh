#!/bin/bash

# Triple Model Demo Runner Script

# Default paths - modify these according to your setup
OVMONO3D_CONFIG="/home/kprokofi/3d_object_detection/ovmono3d/configs/OVMono3D_dinov2_SFP.yaml"
OVMONO3D_WEIGHTS="/home/kprokofi/3d_object_detection/ovmono3d/checkpoints/ovmono3d_lift.pth"
CUBERCNN_CONFIG="/home/kprokofi/.torch/iopath_cache/cubercnn/omni3d/cubercnn_DLA34_FPN.yaml"
CUBERCNN_WEIGHTS="/home/kprokofi/.torch/iopath_cache/cubercnn/omni3d/cubercnn_DLA34_FPN.pth"
DETANY3D_CONFIG="/home/kprokofi/3d_object_detection/DetAny3D/detect_anything/configs/demo.yaml"
DETANY3D_WEIGHTS="/home/kprokofi/3d_object_detection/DetAny3D/checkpoints/detany3d_ckpts/other_exp_ckpt.pth"
DINO_CONFIG="/home/kprokofi/3d_object_detection/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
DINO_WEIGHTS="/home/kprokofi/3d_object_detection/GroundingDINO/weights/groundingdino_swint_ogc.pth"

# Set CUDA device
export CUDA_VISIBLE_DEVICES=0

echo "=== Triple Model 3D Detection Demo ==="
echo "Comparing OVMono3D, CubeRCNN, and DetAny3D on the same images"
echo ""

# Check if JSON input file is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <json_input_file> [output_directory]"
    echo ""
    echo "Example:"
    echo "  $0 example_input.json ./results"
    echo ""
    echo "JSON format example:"
    echo '  [
    {
      "image_path": "/path/to/image.jpg",
      "text_prompt": "person . car . chair", 
      "K_matrix": [[525.0, 0.0, 320.0], [0.0, 525.0, 240.0], [0.0, 0.0, 1.0]],
      "gt_boxes_2d": [[100, 100, 200, 300]]  // optional
    }
  ]'
    echo ""
    echo "To create JSON from existing data:"
    echo "  python create_json_input.py --mode folder --image-folder /path/to/images --labels-file /path/to/labels.json --output-json input.json"
    exit 1
fi

JSON_FILE=$1
OUTPUT_DIR=${2:-"./demo_results_$(date +%Y%m%d_%H%M%S)"}
DEVICE=${3:-"cuda:0"}

# Check if JSON file exists
if [ ! -f "$JSON_FILE" ]; then
    echo "Error: JSON file not found: $JSON_FILE"
    exit 1
fi

# Check if model files exist
echo "Checking model files..."
missing_files=""

if [ ! -f "$OVMONO3D_CONFIG" ]; then
    missing_files="$missing_files\n  - OVMono3D config: $OVMONO3D_CONFIG"
fi

if [ ! -f "$OVMONO3D_WEIGHTS" ]; then
    missing_files="$missing_files\n  - OVMono3D weights: $OVMONO3D_WEIGHTS"
fi

if [ ! -f "$CUBERCNN_CONFIG" ]; then
    missing_files="$missing_files\n  - CubeRCNN config: $CUBERCNN_CONFIG"
fi

if [ ! -f "$CUBERCNN_WEIGHTS" ]; then
    missing_files="$missing_files\n  - CubeRCNN weights: $CUBERCNN_WEIGHTS"
fi

if [ ! -f "$DETANY3D_CONFIG" ]; then
    missing_files="$missing_files\n  - DetAny3D config: $DETANY3D_CONFIG"
fi

if [ ! -f "$DETANY3D_WEIGHTS" ]; then
    missing_files="$missing_files\n  - DetAny3D weights: $DETANY3D_WEIGHTS"
fi

if [ ! -f "$DINO_CONFIG" ]; then
    missing_files="$missing_files\n  - GroundingDINO config: $DINO_CONFIG"
fi

if [ ! -f "$DINO_WEIGHTS" ]; then
    missing_files="$missing_files\n  - GroundingDINO weights: $DINO_WEIGHTS"
fi

if [ -n "$missing_files" ]; then
    echo "Error: Missing model files:"
    echo -e "$missing_files"
    echo ""
    echo "Please download the required model files or update the paths in this script."
    exit 1
fi

echo "✓ All model files found"

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "Starting triple model demo..."
echo "JSON input: $JSON_FILE"
echo "Output directory: $OUTPUT_DIR"
echo ""

# Run the demo
python dual_model_demo.py \
    --json-file "$JSON_FILE" \
    --output-dir "$OUTPUT_DIR" \
    --ovmono3d-config "$OVMONO3D_CONFIG" \
    --ovmono3d-weights "$OVMONO3D_WEIGHTS" \
    --cubercnn-config "$CUBERCNN_CONFIG" \
    --cubercnn-weights "$CUBERCNN_WEIGHTS" \
    --detany3d-config "$DETANY3D_CONFIG" \
    --detany3d-weights "$DETANY3D_WEIGHTS" \
    --dino-config "$DINO_CONFIG" \
    --dino-weights "$DINO_WEIGHTS" \
    --device "$DEVICE"

if [ $? -eq 0 ]; then
    echo ""
    echo "=== Demo completed successfully! ==="
    echo "Results saved to: $OUTPUT_DIR"
    echo ""
    echo "Output files:"
    echo "  - *_ovmono3d.jpg    : OVMono3D visualizations"
    echo "  - *_cubercnn.jpg    : CubeRCNN visualizations"
    echo "  - *_detany3d.jpg    : DetAny3D visualizations" 
    echo "  - *_comparison.jpg  : Three-way comparisons"
    echo "  - results_summary.json : Detailed results and timing"
    echo ""
    echo "To view results:"
    echo "  ls -la $OUTPUT_DIR"
    echo "  firefox $OUTPUT_DIR/*.jpg  # View images"
    echo "  cat $OUTPUT_DIR/results_summary.json | jq  # View JSON results"
else
    echo ""
    echo "=== Demo failed! ==="
    echo "Check the error messages above and ensure all dependencies are installed."
    exit 1
fi
