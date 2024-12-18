import time
import torch
import numpy as np
from PIL import Image
from pathlib import Path
import psutil
import cpuinfo
from typing import Dict, List, Tuple, Optional
import logging
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import json
import torchvision.transforms as transforms
import random
import warnings

warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class COCODatasetHandler:
    def __init__(self, coco_dir: str):
        self.coco_dir = Path(coco_dir)
        self.ann_file = self.coco_dir / 'annotations' / 'instances_val2017.json'
        self.img_dir = self.coco_dir / 'val2017'
        
        if not self.ann_file.exists() or not self.img_dir.exists():
            raise FileNotFoundError(f"COCO dataset not found in {coco_dir}")
            
        self.coco = COCO(str(self.ann_file))
        self.load_categories()
        
    def load_categories(self):
        """Load COCO categories"""
        self.categories = self.coco.loadCats(self.coco.getCatIds())
        self.category_names = [cat['name'] for cat in self.categories]
        logger.info(f"Loaded {len(self.category_names)} categories")
        
    def get_random_images(self, num_images: int = 100) -> List[str]:
        """Get random subset of images for evaluation"""
        img_ids = self.coco.getImgIds()
        selected_ids = random.sample(img_ids, min(num_images, len(img_ids)))
        return [str(self.img_dir / self.coco.loadImgs(img_id)[0]['file_name']) 
                for img_id in selected_ids]

class ModelEvaluator:
    def __init__(self, coco_handler: COCODatasetHandler):
        self.device = torch.device('cpu')
        self.models: Dict[str, torch.nn.Module] = {}
        self.weights_dir = Path('weights')
        self.coco_handler = coco_handler
        self.transform = transforms.Compose([
            transforms.Resize((640, 640)),
            transforms.ToTensor(),
        ])
        
    def load_yolo_model(self, model_name: str, weight_path: Path) -> Optional[torch.nn.Module]:
        """Load different versions of YOLO models"""
        try:
            if 'yolov5' in model_name:
                model = torch.hub.load('ultralytics/yolov5', 'custom', path=str(weight_path))
                return model.to(self.device)
            else:
                try:
                    from ultralytics import YOLO
                except ImportError:
                    logger.info("Installing ultralytics package...")
                    import subprocess
                    subprocess.check_call(["pip", "install", "ultralytics"])
                    from ultralytics import YOLO
                
                model = YOLO(str(weight_path))
                return model
                
        except Exception as e:
            logger.error(f"Error loading {model_name}: {str(e)}")
            return None

    def load_all_models(self):
        """Load all available models from weights directory"""
        weight_files = list(self.weights_dir.glob('*.pt'))
        
        logger.info(f"\nFound {len(weight_files)} model files:")
        for f in weight_files:
            logger.info(f"- {f.name}")
        
        logger.info("\nLoading models...")
        for weight_path in weight_files:
            model_name = weight_path.stem
            logger.info(f"\nAttempting to load {model_name}...")
            
            try:
                model = self.load_yolo_model(model_name, weight_path)
                    
                if model is not None:
                    self.models[model_name] = model
                    logger.info(f"✓ Successfully loaded {model_name}")
                else:
                    logger.error(f"✗ Failed to load {model_name}: model is None")
            except Exception as e:
                logger.error(f"✗ Error loading {model_name}: {str(e)}")
                logger.debug("Traceback:", exc_info=True)

    def preprocess_image(self, image_path: str) -> torch.Tensor:
        """Preprocess image for inference"""
        image = Image.open(image_path).convert('RGB')
        return self.transform(image).unsqueeze(0)

    def measure_inference_time(self, model: torch.nn.Module, image: torch.Tensor, num_runs: int = 100) -> float:
        """Measure average inference time for YOLOv5"""
        times = []
        
        # Warmup
        for _ in range(10):
            with torch.no_grad():
                _ = model(image)
        
        # Actual measurement
        for _ in range(num_runs):
            start_time = time.time()
            with torch.no_grad():
                _ = model(image)
            times.append(time.time() - start_time)
            
        return np.mean(times)

    def measure_ultralytics_inference_time(self, model, image, num_runs: int = 100) -> float:
        """Measure inference time for Ultralytics models (YOLO8,9,10,11)"""
        times = []
        
        # Warmup
        for _ in range(10):
            with torch.no_grad():
                _ = model(image, verbose=False)
        
        # Actual measurement
        for _ in range(num_runs):
            start_time = time.time()
            with torch.no_grad():
                _ = model(image, verbose=False)
            times.append(time.time() - start_time)
            
        return np.mean(times)

    def get_system_info(self) -> Dict[str, str]:
        """Get CPU specifications"""
        cpu_info = cpuinfo.get_cpu_info()
        system_info = {
            'cpu_model': cpu_info['brand_raw'],
            'cpu_cores': str(psutil.cpu_count(logical=False)),
            'cpu_threads': str(psutil.cpu_count(logical=True)),
            'memory': f"{psutil.virtual_memory().total / (1024**3):.1f} GB"
        }
        return system_info

    def evaluate_models(self, num_images: int = 100, num_runs: int = 100):
        """Evaluate all loaded models on COCO dataset"""
        if not self.models:
            logger.error("No models loaded. Please load models first.")
            return
        
        # Store results for all models
        results = {}
        
        # Get random subset of COCO images
        test_images = self.coco_handler.get_random_images(num_images)
        
        # First collect all results
        for model_name, model in self.models.items():
            logger.info(f"Evaluating {model_name}...")
            try:
                total_time = 0
                num_processed = 0
                
                for img_path in test_images:
                    try:
                        if 'yolov5' in model_name:
                            image = self.preprocess_image(img_path)
                            avg_time = self.measure_inference_time(model, image, num_runs=num_runs)
                        else:  # For YOLO 8,9,10,11
                            img = Image.open(img_path)
                            avg_time = self.measure_ultralytics_inference_time(model, img, num_runs=num_runs)
                        
                        total_time += avg_time
                        num_processed += 1
                        
                    except Exception as img_error:
                        logger.error(f"Error processing image {img_path}: {str(img_error)}")
                        continue
                
                if num_processed > 0:
                    avg_inference_time = total_time / num_processed
                    fps = 1 / avg_inference_time
                    
                    results[model_name] = {
                        'inference_time': avg_inference_time * 1000,  # Convert to ms
                        'fps': fps,
                        'images_processed': num_processed
                    }
                
            except Exception as e:
                logger.error(f"Error evaluating {model_name}: {str(e)}")
                logger.error("Stack trace:", exc_info=True)
        
        # Print system information
        logger.info("\n" + "="*50)
        logger.info("EVALUATION RESULTS")
        logger.info("="*50)
        
        system_info = self.get_system_info()
        logger.info("\nSystem Information:")
        for key, value in system_info.items():
            logger.info(f"{key}: {value}")
        
        # Print results table
        logger.info("\nModel Performance:")
        logger.info("-" * 80)
        logger.info(f"{'Model Name':<20} {'Inference Time (ms)':<20} {'FPS':<15} {'Images':<10}")
        logger.info("-" * 80)
        
        # Sort models by FPS for easy comparison
        sorted_models = sorted(results.items(), key=lambda x: x[1]['fps'], reverse=True)
        
        for model_name, result in sorted_models:
            logger.info(
                f"{model_name:<20} "
                f"{result['inference_time']:<20.2f} "
                f"{result['fps']:<15.2f} "
                f"{result['images_processed']:<10}"
            )
        
        logger.info("-" * 80)

def parse_args():
    """Parse command line arguments"""
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate object detection models on CPU')
    
    parser.add_argument('--coco-path', 
                        type=str, 
                        required=True,
                        help='Path to COCO dataset directory')
    
    parser.add_argument('--num-images', 
                        type=int, 
                        default=100,
                        help='Number of images to evaluate (default: 100)')
    
    parser.add_argument('--weights-dir', 
                        type=str, 
                        default='weights',
                        help='Directory containing model weights (default: weights)')
    
    parser.add_argument('--batch-size', 
                        type=int, 
                        default=1,
                        help='Batch size for inference (default: 1)')
    
    parser.add_argument('--num-runs', 
                        type=int, 
                        default=100,
                        help='Number of runs for FPS calculation (default: 100)')
    
    return parser.parse_args()

def main():
    # Parse arguments
    args = parse_args()
    
    # Initialize COCO dataset handler
    try:
        coco_handler = COCODatasetHandler(args.coco_path)
    except FileNotFoundError as e:
        logger.error(f"Error: {e}")
        logger.error(f"Please make sure the COCO dataset exists at: {args.coco_path}")
        return
    
    # Create evaluator
    evaluator = ModelEvaluator(coco_handler)
    evaluator.weights_dir = Path(args.weights_dir)
    
    # Check if weights directory exists
    if not evaluator.weights_dir.exists():
        logger.error(f"Weights directory not found: {args.weights_dir}")
        return
    
    # Load all models
    evaluator.load_all_models()
    
    # Print summary of loaded models
    logger.info("\nSuccessfully loaded models:")
    logger.info("-" * 40)
    for idx, model_name in enumerate(evaluator.models.keys(), 1):
        logger.info(f"{idx}. {model_name}")
    logger.info("-" * 40)
    
    if not evaluator.models:
        logger.error("No models were successfully loaded. Exiting...")
        return
    
    # Add delay before starting evaluation
    logger.info("\nStarting evaluation in 5 seconds...")
    time.sleep(5)
    
    # Run evaluation on COCO dataset
    evaluator.evaluate_models(num_images=args.num_images, num_runs=args.num_runs)

if __name__ == "__main__":
    main()


# python evaluate_models.py \
#     --coco-path /path/to/coco \
#     --num-images 200 \
#     --weights-dir /path/to/weights \
#     --batch-size 1 \
#     --num-runs 50