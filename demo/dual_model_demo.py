#!/usr/bin/env python3
"""
Dual Model Demo: Compare OVMono3D and DetAny3D on the same images
Includes benchmarking functionality with FPS, latency, and GFLOPS measurements
"""

import os
import sys
import json
import argparse
import numpy as np
import cv2
import torch
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from tqdm import tqdm

# Try to import fvcore for FLOPS counting
from fvcore.nn import FlopCountAnalysis, flop_count_table
FVCORE_AVAILABLE = True

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add paths for both models
sys.path.append('/home/kprokofi/3d_object_detection/ovmono3d')
sys.path.append('/home/kprokofi/3d_object_detection/DetAny3D')

# OVMono3D imports
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import transforms as T
from cubercnn.config import get_cfg_defaults
from cubercnn.modeling.meta_arch import build_model as build_ovmono3d_model
from cubercnn import util, vis as ovmono3d_vis

# DetAny3D imports
from train_utils import *
from wrap_model import WrapModel
from detect_anything.datasets.utils import compute_3d_bbox_vertices, project_to_image, draw_bbox_2d
import yaml
from box import Box
from PIL import Image
import torch.nn.functional as F
DETANY3D_AVAILABLE = True

# GroundingDINO imports
try:
    from groundingdino.util.inference import load_model as load_dino_model
    from groundingdino.util.inference import predict as dino_predict
    import groundingdino.datasets.transforms as dino_T
    from torchvision.ops import box_convert
    GROUNDINGDINO_AVAILABLE = True
except ImportError as e:
    logger.warning(f"GroundingDINO imports failed: {e}")
    GROUNDINGDINO_AVAILABLE = False

class OVMono3DModel:
    """Wrapper for OVMono3D model"""
    
    def __init__(self, config_path: str, weights_path: str, device: str = 'cuda'):
        self.device = device
        self.cfg = self._setup_config(config_path, weights_path)
        self.model = self._load_model()
        self.augmentations = self._setup_transforms()
        self._count_parameters()
        
    def _setup_config(self, config_path: str, weights_path: str):
        cfg = get_cfg()
        get_cfg_defaults(cfg)
        cfg.merge_from_file(config_path)
        cfg.MODEL.WEIGHTS = weights_path
        cfg.MODEL.DEVICE = self.device
        cfg.freeze()
        return cfg
        
    def _load_model(self):
        model = build_ovmono3d_model(self.cfg)
        DetectionCheckpointer(model).resume_or_load(self.cfg.MODEL.WEIGHTS, resume=True)
        model.to(self.device)
        model.eval()
        return model
        
    def _setup_transforms(self):
        min_size = self.cfg.INPUT.MIN_SIZE_TEST
        max_size = self.cfg.INPUT.MAX_SIZE_TEST
        return T.AugmentationList([T.ResizeShortestEdge(min_size, max_size, "choice")])
    
    def _count_parameters(self):
        """Count model parameters"""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        # Calculate model size in MB
        param_size = 0
        for param in self.model.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in self.model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        size_mb = (param_size + buffer_size) / 1024**2
        
        self.model_params = {
            'total': total_params / 1e6,
            'trainable': trainable_params / 1e6,
            'total_mb': size_mb,
            'total_gb': size_mb / 1024
        }
        
        logger.info(f"OVMono3D parameters: {total_params:,} ({trainable_params:,} trainable)")
        logger.info(f"OVMono3D size: {size_mb:.1f} MB")
    
    def predict(self, image: np.ndarray, categories: List[str], K: np.ndarray) -> Dict:
        """Predict 3D objects using OVMono3D"""
        start_time = time.time()
        
        # Preprocess image
        aug_input = T.AugInput(image)
        _ = self.augmentations(aug_input)  
        processed_image = aug_input.image
        
        # Prepare input
        batched = [{
            'image': torch.as_tensor(
                np.ascontiguousarray(processed_image.transpose(2, 0, 1))
            ).to(self.device), 
            'height': image.shape[0], 
            'width': image.shape[1], 
            'K': K, 
            'category_list': categories
        }]
        
        # Run inference
        with torch.no_grad():
            predictions = self.model(batched)[0]['instances']
            
        inference_time = time.time() - start_time
        
        return {
            'predictions': predictions,
            'inference_time': inference_time,
            'num_detections': len(predictions)
        }
    
    def visualize_results(self, image: np.ndarray, predictions, K: np.ndarray, 
                         categories: List[str], threshold: float = 0.25) -> np.ndarray:
        """Visualize OVMono3D results"""
        meshes = []
        meshes_text = []
        
        if len(predictions) > 0:
            for idx, (corners3D, center_cam, center_2D, dimensions, pose, score, cat_idx) in enumerate(zip(
                    predictions.pred_bbox3D, predictions.pred_center_cam, predictions.pred_center_2D, 
                    predictions.pred_dimensions, predictions.pred_pose, predictions.scores, predictions.pred_classes
                )):
                
                if score < threshold:
                    continue
                    
                cat_name = categories[cat_idx] if cat_idx < len(categories) else f"class_{cat_idx}"
                bbox3D = center_cam.tolist() + dimensions.tolist()
                meshes_text.append(f'{cat_name} {score:.2f}')
                color = [c/255.0 for c in util.get_color(idx)]
                box_mesh = util.mesh_cuboid(bbox3D, pose.tolist(), color=color)
                meshes.append(box_mesh)
        
        if len(meshes) > 0:
            im_drawn_rgb, im_topdown, _ = ovmono3d_vis.draw_scene_view(
                image, K, meshes, text=meshes_text, 
                scale=image.shape[0], blend_weight=0.5, blend_weight_overlay=0.85
            )
            return im_drawn_rgb
        else:
            return image

class DetAny3DModel:
    """Wrapper for DetAny3D model"""
    
    def __init__(self, config_path: str, weights_path: str, dino_config: str, dino_weights: str, device: str = 'cuda:0'):
        if not DETANY3D_AVAILABLE:
            raise ImportError("DetAny3D dependencies not available")
        
        self.device = device
        self.cfg = self._load_config(config_path)
        self.model = self._load_model(weights_path)
        self.dino_model = self._load_dino_model(dino_config, dino_weights) if GROUNDINGDINO_AVAILABLE else None
        
        # Import ResizeLongestSide from SAM
        try:
            from segment_anything.utils.transforms import ResizeLongestSide
            self.sam_trans = ResizeLongestSide(self.cfg.model.pad)
        except ImportError:
            logger.warning("SAM ResizeLongestSide not available")
            self.sam_trans = None
        
        self._count_parameters()
        
    def _load_config(self, config_path: str):
        with open(config_path, 'r', encoding='utf-8') as f:
            cfg_dict = yaml.load(f.read(), Loader=yaml.FullLoader)
        return Box(cfg_dict)
        
    def _load_model(self, weights_path: str):
        # Disable distributed training
        torch.distributed.is_available = lambda: False
        torch.distributed.is_initialized = lambda: False
        torch.distributed.get_world_size = lambda group=None: 1
        torch.distributed.get_rank = lambda group=None: 0
        
        model = WrapModel(self.cfg)
        checkpoint = torch.load(weights_path, map_location=self.device)
        new_model_dict = model.state_dict()
        
        for k, v in new_model_dict.items():
            if k in checkpoint['state_dict'].keys() and checkpoint['state_dict'][k].size() == new_model_dict[k].size():
                new_model_dict[k] = checkpoint['state_dict'][k].detach()
        
        model.load_state_dict(new_model_dict)
        model.to(self.device)
        model.setup()
        model.eval()
        return model
        
    def _load_dino_model(self, config_path: str, weights_path: str):
        if not GROUNDINGDINO_AVAILABLE:
            return None
        return load_dino_model(config_path, weights_path)
    
    def _count_parameters(self):
        """Count model parameters"""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        # Calculate model size in MB
        param_size = 0
        for param in self.model.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in self.model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        size_mb = (param_size + buffer_size) / 1024**2
        
        self.model_params = {
            'total': total_params / 1e6,
            'trainable': trainable_params / 1e6,
            'total_mb': size_mb,
            'total_gb': size_mb / 1024
        }
        
        logger.info(f"DetAny3D parameters: {total_params:,} ({trainable_params:,} trainable)")
        logger.info(f"DetAny3D size: {size_mb:.1f} MB")
    
    def _convert_image_for_dino(self, img: np.ndarray):
        """Convert image for GroundingDINO"""
        transform = dino_T.Compose([
            dino_T.RandomResize([800], max_size=1333),
            dino_T.ToTensor(),
            dino_T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        image_source = Image.fromarray(img, 'RGB')
        image = np.asarray(image_source)
        image_transformed, _ = transform(image_source, None)
        return image, image_transformed
    
    def _preprocess_for_sam(self, x, cfg):
        """Preprocess for SAM"""
        sam_pixel_mean = torch.Tensor(cfg.dataset.pixel_mean).view(-1, 1, 1)
        sam_pixel_std = torch.Tensor(cfg.dataset.pixel_std).view(-1, 1, 1)
        x = (x - sam_pixel_mean) / sam_pixel_std
        
        h, w = x.shape[-2:]
        padh = cfg.model.pad - h
        padw = cfg.model.pad - w
        x = F.pad(x, (0, padw, 0, padh))
        return x
    
    def predict(self, image: np.ndarray, text_prompt: str, bbox_2d_list: Optional[List] = None) -> Dict:
        """Predict 3D objects using DetAny3D"""
        start_time = time.time()
        
        # Handle text prompts with GroundingDINO
        if text_prompt and self.dino_model is not None:
            image_source_dino, image_dino = self._convert_image_for_dino(image)
            boxes, logits, phrases = dino_predict(
                model=self.dino_model,
                image=image_dino,
                caption=text_prompt,
                box_threshold=0.37,
                text_threshold=0.25,
                remove_combined=False,
            )
            
            h, w, _ = image_source_dino.shape
            boxes = boxes * torch.Tensor([w, h, w, h])
            xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy")
            
            bbox_2d_list = []
            label_list = []
            for i, box in enumerate(xyxy):
                bbox_2d_list.append(box.to(torch.int).cpu().numpy().tolist())
                label_list.append(phrases[i])
        elif bbox_2d_list is not None:
            label_list = ["object"] * len(bbox_2d_list)
        else:
            return {
                'predictions': [],
                'inference_time': 0,
                'num_detections': 0,
                'error': 'No valid prompts provided'
            }
        
        if not bbox_2d_list:
            return {
                'predictions': [],
                'inference_time': time.time() - start_time,
                'num_detections': 0,
                'error': 'No objects detected'
            }
        
        # Prepare image for SAM
        original_size = tuple(image.shape[:-1])
        img_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float().unsqueeze(0)
        
        # Apply transformations
        img_tensor = self.sam_trans.apply_image_torch(img_tensor)
        img_tensor = self._crop_hw(img_tensor)
        before_pad_size = tuple(img_tensor.shape[2:])
        
        img_for_sam = self._preprocess_for_sam(img_tensor, self.cfg).to(self.device)
        
        # Prepare bounding boxes
        bbox_2d_tensor = torch.tensor(bbox_2d_list)
        bbox_2d_tensor = self.sam_trans.apply_boxes_torch(bbox_2d_tensor, original_size).to(torch.int).to(self.device)
        
        # Prepare input dictionary
        if self.cfg.model.vit_pad_mask:
            vit_pad_size = (before_pad_size[0] // self.cfg.model.image_encoder.patch_size, 
                           before_pad_size[1] // self.cfg.model.image_encoder.patch_size)
        else:
            vit_pad_size = (self.cfg.model.pad // self.cfg.model.image_encoder.patch_size, 
                           self.cfg.model.pad // self.cfg.model.image_encoder.patch_size)
        
        input_dict = {
            "images": img_for_sam,
            'vit_pad_size': torch.tensor(vit_pad_size).to(self.device).unsqueeze(0),
            "images_shape": torch.Tensor(before_pad_size).to(self.device).unsqueeze(0),
            "boxes_coords": bbox_2d_tensor,
        }
        
        # Run inference
        with torch.no_grad():
            ret_dict = self.model(input_dict)
        
        inference_time = time.time() - start_time
        
        return {
            'predictions': ret_dict,
            'labels': label_list,
            'bbox_2d': bbox_2d_list,
            'inference_time': inference_time,
            'num_detections': len(bbox_2d_list)
        }
    
    def _crop_hw(self, img):
        """Center crop image to be divisible by 14"""
        if img.dim() == 4:
            img = img.squeeze(0)
        h, w = img.shape[1:3]
        
        new_h = (h // 14) * 14
        new_w = (w // 14) * 14
        
        center_h, center_w = h // 2, w // 2
        start_h = center_h - new_h // 2
        start_w = center_w - new_w // 2
        
        img_cropped = img[:, start_h:start_h + new_h, start_w:start_w + new_w]
        return img_cropped.unsqueeze(0)
    
    def visualize_results(self, image: np.ndarray, result_dict: Dict) -> np.ndarray:
        """Visualize DetAny3D results"""
        if 'error' in result_dict:
            return image
            
        ret_dict = result_dict['predictions']
        labels = result_dict['labels']
        
        K = ret_dict['pred_K']
        decoded_bboxes_pred_2d, decoded_bboxes_pred_3d = decode_bboxes(ret_dict, self.cfg, K)
        rot_mat = rotation_6d_to_matrix(ret_dict['pred_pose_6d'])
        
        # Create visualization image
        img_for_sam = ret_dict.get('img_for_sam', None)
        if img_for_sam is not None:
            image_h, image_w = img_for_sam.shape[2], img_for_sam.shape[3]
            origin_img = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1) * img_for_sam[0, :, :image_h, :image_w].squeeze(0).detach().cpu() + torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
            todo = cv2.cvtColor(origin_img.permute(1, 2, 0).numpy(), cv2.COLOR_RGB2BGR)
        else:
            todo = image.copy()
        
        K_np = K.detach().cpu().numpy()
        
        # Draw 3D bounding boxes
        for i in range(len(decoded_bboxes_pred_2d)):
            x, y, z, w, h, l, yaw = decoded_bboxes_pred_3d[i].detach().cpu().numpy()
            rot_mat_i = rot_mat[i].detach().cpu().numpy()
            vertices_3d, fore_plane_center_3d = compute_3d_bbox_vertices(x, y, z, w, h, l, yaw, rot_mat_i)
            vertices_2d = project_to_image(vertices_3d, K_np.squeeze(0))
            
            color = (0, 255, 0)  # Green color
            draw_bbox_2d(todo, vertices_2d, color=color, thickness=3)
            
            if i < len(labels):
                label_text = f"{labels[i]} [{w:.1f},{h:.1f},{l:.1f}]"
                cv2.putText(todo, label_text, (int(vertices_2d[0][0]), int(vertices_2d[0][1]-10)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        return cv2.cvtColor(todo, cv2.COLOR_BGR2RGB)

class DualModelDemo:
    """Main demo class that handles both models"""
    
    def __init__(self, ovmono3d_config: str, ovmono3d_weights: str,
                 detany3d_config: str, detany3d_weights: str,
                 dino_config: str, dino_weights: str, device: str = 'cuda'):
        
        self.device = device
        
        logger.info(f"Initializing OVMono3D model on {device}...")
        self.ovmono3d = OVMono3DModel(ovmono3d_config, ovmono3d_weights, device=device)
        
        logger.info(f"Initializing DetAny3D model on {device}...")
        if DETANY3D_AVAILABLE:
            self.detany3d = DetAny3DModel(detany3d_config, detany3d_weights, dino_config, dino_weights, device=device)
        else:
            self.detany3d = None
            logger.warning("DetAny3D not available")
    
    def process_json_file(self, json_file_path: str, output_dir: str):
        """Process images specified in JSON file"""
        
        with open(json_file_path, 'r') as f:
            data = json.load(f)
        
        os.makedirs(output_dir, exist_ok=True)
        results = []
        
        for item in tqdm(data, desc="Processing images"):
            result = self.process_single_item(item, output_dir)
            results.append(result)
        
        # Save results summary
        with open(os.path.join(output_dir, 'results_summary.json'), 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to {output_dir}")
        return results
    
    def process_single_item(self, item: Dict, output_dir: str) -> Dict:
        """Process a single item from the JSON file"""
        
        image_path = item['image_path']
        text_prompt = item.get('text_prompt', '')
        K_matrix = np.array(item['K_matrix'])
        gt_boxes_2d = item.get('gt_boxes_2d', None)
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            return {'error': f'Could not load image: {image_path}'}
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_name = Path(image_path).stem
        
        result = {
            'image_path': image_path,
            'image_name': image_name,
            'text_prompt': text_prompt,
            'models': {}
        }
        
        # Process with OVMono3D
        logger.info(f"Processing {image_name} with OVMono3D...")
        categories = [cat.strip() for cat in text_prompt.split('.') if cat.strip()] if text_prompt else []
        
        if categories:
            ovmono3d_result = self.ovmono3d.predict(image_rgb, categories, K_matrix)
            ovmono3d_vis = self.ovmono3d.visualize_results(image_rgb, ovmono3d_result['predictions'], K_matrix, categories)
            
            # Save visualization
            ovmono3d_path = os.path.join(output_dir, f'{image_name}_ovmono3d.jpg')
            cv2.imwrite(ovmono3d_path, cv2.cvtColor(ovmono3d_vis, cv2.COLOR_RGB2BGR))
            
            result['models']['ovmono3d'] = {
                'inference_time': ovmono3d_result['inference_time'],
                'num_detections': ovmono3d_result['num_detections'],
                'visualization_path': ovmono3d_path
            }
        
        # Process with DetAny3D
        if self.detany3d is not None:
            logger.info(f"Processing {image_name} with DetAny3D...")
            detany3d_result = self.detany3d.predict(image_rgb, text_prompt, gt_boxes_2d)
            
            if 'error' not in detany3d_result:
                detany3d_vis = self.detany3d.visualize_results(image_rgb, detany3d_result)
                
                # Save visualization
                detany3d_path = os.path.join(output_dir, f'{image_name}_detany3d.jpg')
                cv2.imwrite(detany3d_path, cv2.cvtColor(detany3d_vis, cv2.COLOR_RGB2BGR))
                
                result['models']['detany3d'] = {
                    'inference_time': detany3d_result['inference_time'],
                    'num_detections': detany3d_result['num_detections'],
                    'visualization_path': detany3d_path
                }
            else:
                result['models']['detany3d'] = {'error': detany3d_result['error']}
        
        # Create side-by-side comparison
        self._create_comparison_image(result, output_dir)
        
        return result
    
    def _create_comparison_image(self, result: Dict, output_dir: str):
        """Create side-by-side comparison of both models"""
        
        ovmono3d_path = result['models'].get('ovmono3d', {}).get('visualization_path')
        detany3d_path = result['models'].get('detany3d', {}).get('visualization_path')
        
        if ovmono3d_path and detany3d_path:
            img1 = cv2.imread(ovmono3d_path)
            img2 = cv2.imread(detany3d_path)
            
            if img1 is not None and img2 is not None:
                # Resize images to same height
                h = min(img1.shape[0], img2.shape[0])
                img1 = cv2.resize(img1, (int(img1.shape[1] * h / img1.shape[0]), h))
                img2 = cv2.resize(img2, (int(img2.shape[1] * h / img2.shape[0]), h))
                
                # Add labels
                cv2.putText(img1, 'OVMono3D', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(img2, 'DetAny3D', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Concatenate horizontally
                comparison = np.hstack([img1, img2])
                
                # Save comparison
                comparison_path = os.path.join(output_dir, f'{result["image_name"]}_comparison.jpg')
                cv2.imwrite(comparison_path, comparison)
                
                result['comparison_path'] = comparison_path

    def benchmark_model(self, model_name: str, batch_sizes: List[int] = [1, 2, 4, 8],
                       num_iterations: int = 100, warmup_iterations: int = 10,
                       image_size: Tuple[int, int] = (1242, 375), compute_flops: bool = True) -> Dict:
        """
        Benchmark a model with synthetic random images
        
        Args:
            model_name: 'ovmono3d' or 'detany3d'
            batch_sizes: List of batch sizes to test
            num_iterations: Number of iterations per batch size
            warmup_iterations: Number of warmup iterations (not timed)
            image_size: (width, height) of generated images
            compute_flops: Whether to compute FLOPS (requires fvcore)
            
        Returns:
            Dictionary with benchmark results
        """
        if model_name == 'ovmono3d':
            model = self.ovmono3d
        elif model_name == 'detany3d':
            if self.detany3d is None:
                raise ValueError("DetAny3D not available")
            model = self.detany3d
        else:
            raise ValueError(f"Unknown model: {model_name}. Choose 'ovmono3d' or 'detany3d'")
        
        logger.info(f"Starting benchmark for {model_name}...")
        logger.info(f"Device: {self.device}")
        logger.info(f"Image size: {image_size}")
        logger.info(f"Batch sizes: {batch_sizes}")
        logger.info(f"Iterations: {num_iterations} (warmup: {warmup_iterations})")
        
        width, height = image_size
        results = {
            'model_name': model_name,
            'device': self.device,
            'image_size': image_size,
            'num_iterations': num_iterations,
            'warmup_iterations': warmup_iterations,
            'batch_results': {},
            'model_params': model.model_params,
        }
        
        # Try to compute FLOPS
        if compute_flops and FVCORE_AVAILABLE:
            try:
                logger.info("Computing FLOPS...")
                flops_input = self._prepare_flops_input(model_name, width, height)
                if flops_input is not None:
                    flop_counter = FlopCountAnalysis(model.model, flops_input)
                    flops_total = flop_counter.total()
                    results['flops'] = {
                        'total_flops': int(flops_total),
                        'gflops': flops_total / 1e9,
                        'gmacs': flops_total / 2 / 1e9,
                    }
                    logger.info(f"FLOPS: {results['flops']['gflops']:.2f} GFLOPs ({results['flops']['gmacs']:.2f} GMACs)")
            except Exception as e:
                logger.warning(f"Could not compute FLOPS: {e}")
                results['flops'] = None
        else:
            results['flops'] = None
        
        # Benchmark each batch size
        for batch_size in batch_sizes:
            logger.info(f"\n=== Benchmarking batch_size={batch_size} ===")
            
            # Generate random images
            logger.info(f"Generating {batch_size} random images...")
            random_images = []
            for _ in range(batch_size):
                # Generate random RGB image
                img = np.random.randint(0, 256, size=(height, width, 3), dtype=np.uint8)
                random_images.append(img)
            
            # Preprocess images
            logger.info("Preprocessing images...")
            preprocessed = self._preprocess_batch(model_name, random_images)
            
            # Warmup iterations
            logger.info(f"Running {warmup_iterations} warmup iterations...")
            for _ in range(warmup_iterations):
                with torch.no_grad():
                    _ = self._forward_batch(model_name, preprocessed)
                if self.device.startswith('cuda'):
                    torch.cuda.synchronize()
            
            # Benchmark iterations
            logger.info(f"Running {num_iterations} benchmark iterations...")
            latencies = []
            for _ in tqdm(range(num_iterations), desc=f"Batch {batch_size}"):
                if self.device.startswith('cuda'):
                    torch.cuda.synchronize()
                start = time.time()
                
                with torch.no_grad():
                    _ = self._forward_batch(model_name, preprocessed)
                
                if self.device.startswith('cuda'):
                    torch.cuda.synchronize()
                end = time.time()
                
                latencies.append(end - start)
            
            # Compute statistics
            latencies_ms = np.array(latencies) * 1000
            batch_result = {
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
                'per_image_fps': batch_size / np.mean(latencies_ms) * 1000,
            }
            
            # Add FLOPS metrics
            if results['flops'] is not None:
                gflops_per_image = results['flops']['gflops'] / batch_size
                gflops_per_second = results['flops']['gflops'] / np.mean(latencies)
                batch_result['gflops_per_image'] = gflops_per_image
                batch_result['gflops_per_second'] = gflops_per_second
                batch_result['gmacs_per_image'] = results['flops']['gmacs'] / batch_size
            
            # Memory usage (if CUDA)
            if self.device.startswith('cuda'):
                batch_result['memory_allocated_mb'] = torch.cuda.memory_allocated() / 1024**2
                batch_result['memory_reserved_mb'] = torch.cuda.memory_reserved() / 1024**2
            
            results['batch_results'][batch_size] = batch_result
            
            # Print summary
            logger.info(f"Batch {batch_size} results:")
            logger.info(f"  Latency: {batch_result['latency_mean_ms']:.2f} ± {batch_result['latency_std_ms']:.2f} ms")
            logger.info(f"  Throughput: {batch_result['throughput_fps']:.2f} FPS")
            logger.info(f"  Per-image: {batch_result['per_image_latency_ms']:.2f} ms ({batch_result['per_image_fps']:.2f} FPS)")
            if 'gflops_per_second' in batch_result:
                logger.info(f"  Compute: {batch_result['gflops_per_second']:.2f} GFLOPS/s")

        return results
    
    def _prepare_flops_input(self, model_name: str, width: int, height: int):
        """Prepare sample input for FLOPS counting"""
        try:
            if model_name == 'ovmono3d':
                # Generate a sample image tensor
                img = np.random.randint(0, 256, size=(height, width, 3), dtype=np.uint8)
                
                # Preprocess
                aug_input = T.AugInput(img)
                _ = self.ovmono3d.augmentations(aug_input)
                image_tensor = torch.as_tensor(aug_input.image.transpose(2, 0, 1).astype("float32")).to(self.device)
                
                # Create sample input dict
                sample_input = [{
                    'image': image_tensor,
                    'height': height,
                    'width': width,
                    'K': torch.eye(3),
                    'category_list': ['car', 'truck', 'bus']
                }]
                return (sample_input,)
                
            elif model_name == 'detany3d':
                # Create a sample batched input for DetAny3D
                img_tensor = torch.randn(1, 3, self.detany3d.cfg.model.pad, self.detany3d.cfg.model.pad).to(self.device)
                vit_pad_size = (self.detany3d.cfg.model.pad // self.detany3d.cfg.model.image_encoder.patch_size,
                               self.detany3d.cfg.model.pad // self.detany3d.cfg.model.image_encoder.patch_size)
                
                input_dict = {
                    "images": img_tensor,
                    'vit_pad_size': torch.tensor(vit_pad_size).to(self.device).unsqueeze(0),
                    "images_shape": torch.tensor([height, width]).to(self.device).unsqueeze(0),
                    "boxes_coords": torch.tensor([[100, 100, 200, 200]], dtype=torch.int).to(self.device),
                }
                return (input_dict,)
            
        except Exception as e:
            logger.warning(f"Failed to prepare FLOPS input: {e}")
            return None
    
    def _preprocess_batch(self, model_name: str, images: List[np.ndarray]):
        """Preprocess a batch of images for the specified model"""
        if model_name == 'ovmono3d':
            # Preprocess images for OVMono3D
            batch_inputs = []
            for img in images:
                aug_input = T.AugInput(img)
                _ = self.ovmono3d.augmentations(aug_input)
                image_tensor = torch.as_tensor(aug_input.image.transpose(2, 0, 1).astype("float32")).to(self.device)
                
                input_dict = {
                    'image': image_tensor,
                    'height': img.shape[0],
                    'width': img.shape[1],
                    'K': np.eye(3),
                    'category_list': ['car', 'truck', 'bus', 'person', 'bicycle']
                }
                batch_inputs.append(input_dict)
            return batch_inputs
            
        elif model_name == 'detany3d':
            # Preprocess for DetAny3D (simplified - just use bounding box prompts)
            batch_imgs = []
            batch_bboxes = []
            
            for img in images:
                # Convert to tensor and apply SAM transform
                img_tensor = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0)
                img_tensor = self.detany3d.sam_trans.apply_image_torch(img_tensor)
                img_tensor = self.detany3d._crop_hw(img_tensor)
                before_pad_size = tuple(img_tensor.shape[2:])
                
                img_for_sam = self.detany3d._preprocess_for_sam(img_tensor, self.detany3d.cfg)
                batch_imgs.append(img_for_sam)
                
                # Use a dummy bounding box
                bbox_2d = torch.tensor([[100, 100, 300, 300]], dtype=torch.int)
                bbox_2d = self.detany3d.sam_trans.apply_boxes_torch(bbox_2d, (img.shape[0], img.shape[1]))
                batch_bboxes.append(bbox_2d)
            
            # Stack into batch
            imgs_batch = torch.cat(batch_imgs, dim=0).to(self.device)
            bboxes_batch = torch.cat(batch_bboxes, dim=0).to(self.device)
            
            # Prepare vit_pad_size and images_shape
            if self.detany3d.cfg.model.vit_pad_mask:
                vit_pad_size = (before_pad_size[0] // self.detany3d.cfg.model.image_encoder.patch_size,
                               before_pad_size[1] // self.detany3d.cfg.model.image_encoder.patch_size)
            else:
                vit_pad_size = (self.detany3d.cfg.model.pad // self.detany3d.cfg.model.image_encoder.patch_size,
                               self.detany3d.cfg.model.pad // self.detany3d.cfg.model.image_encoder.patch_size)
            
            return {
                "images": imgs_batch,
                "image_for_dino": imgs_batch,
                'vit_pad_size': torch.tensor(vit_pad_size).to(self.device).unsqueeze(0).repeat(len(images), 1),
                "images_shape": torch.tensor(before_pad_size).to(self.device).unsqueeze(0).repeat(len(images), 1),
                "boxes_coords": bboxes_batch,
            }
    
    def _forward_batch(self, model_name: str, preprocessed):
        """Forward pass for a batch"""
        if model_name == 'ovmono3d':
            return self.ovmono3d.model(preprocessed)
        elif model_name == 'detany3d':
            return self.detany3d.model(preprocessed)

def main():
    parser = argparse.ArgumentParser(description="Dual Model 3D Detection Demo")
    
    # Mode selection
    parser.add_argument('--benchmark', action='store_true', help='Run benchmark mode')
    parser.add_argument('--json-file', help='JSON file with image data (for demo mode)')
    parser.add_argument('--output-dir', help='Output directory for results')
    
    # Model selection
    parser.add_argument('--model', choices=['ovmono3d', 'detany3d', 'both'], default='both',
                       help='Which model to benchmark (default: both)')
    parser.add_argument('--device', default='cuda', help='Device to use (cuda, cpu, cuda:0, etc.)')
    
    # Benchmark parameters
    parser.add_argument('--batch-sizes', nargs='+', type=int, default=[1, 2, 4, 8],
                       help='Batch sizes to test (default: 1 2 4 8)')
    parser.add_argument('--num-iterations', type=int, default=100,
                       help='Number of iterations per batch size (default: 100)')
    parser.add_argument('--warmup-iterations', type=int, default=10,
                       help='Number of warmup iterations (default: 10)')
    parser.add_argument('--image-width', type=int, default=1242,
                       help='Width of generated images (default: 1242)')
    parser.add_argument('--image-height', type=int, default=375,
                       help='Height of generated images (default: 375)')
    parser.add_argument('--no-flops', action='store_true',
                       help='Disable FLOPS counting')
    
    # OVMono3D arguments
    parser.add_argument('--ovmono3d-config', default='/home/kprokofi/3d_object_detection/ovmono3d/configs/OVMono3D_dinov2_SFP.yaml')
    parser.add_argument('--ovmono3d-weights', default='/home/kprokofi/3d_object_detection/ovmono3d/checkpoints/ovmono3d_lift.pth')
    
    # DetAny3D arguments  
    parser.add_argument('--detany3d-config', default='/home/kprokofi/3d_object_detection/DetAny3D/detect_anything/configs/demo.yaml')
    parser.add_argument('--detany3d-weights', default='/home/kprokofi/3d_object_detection/DetAny3D/checkpoints/detany3d_ckpts/detany3d.pth')
    
    # GroundingDINO arguments
    parser.add_argument('--dino-config', default='/home/kprokofi/3d_object_detection/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py')
    parser.add_argument('--dino-weights', default='/home/kprokofi/3d_object_detection/GroundingDINO/weights/groundingdino_swint_ogc.pth')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.benchmark and not args.json_file:
        parser.error("Either --benchmark or --json-file is required")
    
    # Initialize demo
    demo = DualModelDemo(
        ovmono3d_config=args.ovmono3d_config,
        ovmono3d_weights=args.ovmono3d_weights,
        detany3d_config=args.detany3d_config,
        detany3d_weights=args.detany3d_weights,
        dino_config=args.dino_config,
        dino_weights=args.dino_weights,
        device=args.device
    )
    
    if args.benchmark:
        # Run benchmark mode
        image_size = (args.image_width, args.image_height)
        compute_flops = not args.no_flops
        
        benchmark_results = {}
        
        # Benchmark OVMono3D
        if args.model in ['ovmono3d', 'both']:
            logger.info("\n" + "="*60)
            logger.info("BENCHMARKING OVMono3D")
            logger.info("="*60)
            ovmono3d_results = demo.benchmark_model(
                'ovmono3d',
                batch_sizes=args.batch_sizes,
                num_iterations=args.num_iterations,
                warmup_iterations=args.warmup_iterations,
                image_size=image_size,
                compute_flops=compute_flops
            )
            benchmark_results['ovmono3d'] = ovmono3d_results
            
            # Save results
            output_dir = args.output_dir or 'benchmark_results'
            os.makedirs(output_dir, exist_ok=True)
            ovmono3d_json = os.path.join(output_dir, 'ovmono3d_benchmark.json')
            with open(ovmono3d_json, 'w') as f:
                json.dump(ovmono3d_results, f, indent=2)
            logger.info(f"\nOVMono3D results saved to {ovmono3d_json}")
        
        # Benchmark DetAny3D
        if args.model in ['detany3d', 'both'] and demo.detany3d is not None:
            logger.info("\n" + "="*60)
            logger.info("BENCHMARKING DetAny3D")
            logger.info("="*60)
            detany3d_results = demo.benchmark_model(
                'detany3d',
                batch_sizes=args.batch_sizes,
                num_iterations=args.num_iterations,
                warmup_iterations=args.warmup_iterations,
                image_size=image_size,
                compute_flops=compute_flops
            )
            benchmark_results['detany3d'] = detany3d_results
            print(detany3d_results)
            # Save results
            # output_dir = args.output_dir or 'benchmark_results'
            # detany3d_json = os.path.join(output_dir, 'detany3d_benchmark.json')
            # with open(detany3d_json, 'w') as f:
            #     json.dump(detany3d_results, f, indent=2)
            # logger.info(f"\nDetAny3D results saved to {detany3d_json}")
        
        # Print comparison if both models were benchmarked
        if 'ovmono3d' in benchmark_results and 'detany3d' in benchmark_results:
            logger.info("\n" + "="*60)
            logger.info("COMPARISON SUMMARY")
            logger.info("="*60)
            
            # Compare batch_size=1 results
            ov_b1 = benchmark_results['ovmono3d']['batch_results'].get(1, {})
            da_b1 = benchmark_results['detany3d']['batch_results'].get(1, {})
            
            logger.info("\nBatch Size 1 Comparison:")
            logger.info(f"{'Metric':<25} {'OVMono3D':>15} {'DetAny3D':>15} {'Winner':>10}")
            logger.info("-" * 70)
            
            if 'latency_mean_ms' in ov_b1 and 'latency_mean_ms' in da_b1:
                ov_lat = ov_b1['latency_mean_ms']
                da_lat = da_b1['latency_mean_ms']
                winner = 'OVMono3D' if ov_lat < da_lat else 'DetAny3D'
                logger.info(f"{'Latency (ms)':<25} {ov_lat:>15.2f} {da_lat:>15.2f} {winner:>10}")
            
            if 'per_image_fps' in ov_b1 and 'per_image_fps' in da_b1:
                ov_fps = ov_b1['per_image_fps']
                da_fps = da_b1['per_image_fps']
                winner = 'OVMono3D' if ov_fps > da_fps else 'DetAny3D'
                logger.info(f"{'FPS':<25} {ov_fps:>15.2f} {da_fps:>15.2f} {winner:>10}")
            
            if 'gflops_per_image' in ov_b1 and 'gflops_per_image' in da_b1:
                ov_gf = ov_b1['gflops_per_image']
                da_gf = da_b1['gflops_per_image']
                winner = 'OVMono3D' if ov_gf < da_gf else 'DetAny3D'
                logger.info(f"{'GFLOPs/image':<25} {ov_gf:>15.2f} {da_gf:>15.2f} {winner:>10}")
            
            # Model parameters
            ov_params = benchmark_results['ovmono3d']['model_params']['total']
            da_params = benchmark_results['detany3d']['model_params']['total']
            winner = 'OVMono3D' if ov_params < da_params else 'DetAny3D'
            logger.info(f"{'Parameters (M)':<25} {ov_params:>15.2f} {da_params:>15.2f} {winner:>10}")
            
            # Save comparison
            output_dir = args.output_dir or 'benchmark_results'
            comparison_path = os.path.join(output_dir, 'comparison_summary.txt')
            with open(comparison_path, 'w') as f:
                f.write("="*70 + "\n")
                f.write("BENCHMARK COMPARISON: OVMono3D vs DetAny3D\n")
                f.write("="*70 + "\n\n")
                f.write(f"Device: {args.device}\n")
                f.write(f"Image Size: {image_size}\n")
                f.write(f"Iterations: {args.num_iterations}\n\n")
                f.write(f"{'Metric':<25} {'OVMono3D':>15} {'DetAny3D':>15} {'Winner':>10}\n")
                f.write("-" * 70 + "\n")
                
                if 'latency_mean_ms' in ov_b1 and 'latency_mean_ms' in da_b1:
                    winner = 'OVMono3D' if ov_b1['latency_mean_ms'] < da_b1['latency_mean_ms'] else 'DetAny3D'
                    f.write(f"{'Latency (ms)':<25} {ov_b1['latency_mean_ms']:>15.2f} {da_b1['latency_mean_ms']:>15.2f} {winner:>10}\n")
                
                if 'per_image_fps' in ov_b1 and 'per_image_fps' in da_b1:
                    winner = 'OVMono3D' if ov_b1['per_image_fps'] > da_b1['per_image_fps'] else 'DetAny3D'
                    f.write(f"{'FPS':<25} {ov_b1['per_image_fps']:>15.2f} {da_b1['per_image_fps']:>15.2f} {winner:>10}\n")
                
                if 'gflops_per_image' in ov_b1 and 'gflops_per_image' in da_b1:
                    winner = 'OVMono3D' if ov_b1['gflops_per_image'] < da_b1['gflops_per_image'] else 'DetAny3D'
                    f.write(f"{'GFLOPs/image':<25} {ov_b1['gflops_per_image']:>15.2f} {da_b1['gflops_per_image']:>15.2f} {winner:>10}\n")
                
                winner = 'OVMono3D' if ov_params < da_params else 'DetAny3D'
                f.write(f"{'Parameters (M)':<25} {ov_params:>15.2f} {da_params:>15.2f} {winner:>10}\n")
            
            logger.info(f"\nComparison saved to {comparison_path}")
    
    else:
        # Run demo mode
        if not args.output_dir:
            parser.error("--output-dir is required for demo mode")
        
        # Process images
        results = demo.process_json_file(args.json_file, args.output_dir)
        
        # Print summary
        ovmono3d_times = [r['models'].get('ovmono3d', {}).get('inference_time', 0) for r in results]
        detany3d_times = [r['models'].get('detany3d', {}).get('inference_time', 0) for r in results if 'detany3d' in r['models']]
        
        logger.info("=== SUMMARY ===")
        logger.info(f"Processed {len(results)} images")
        if ovmono3d_times:
            logger.info(f"OVMono3D avg time: {np.mean(ovmono3d_times):.3f}s")
        if detany3d_times:
            logger.info(f"DetAny3D avg time: {np.mean(detany3d_times):.3f}s")

if __name__ == "__main__":
    main()
