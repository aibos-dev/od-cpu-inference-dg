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
import matplotlib.pyplot as plt
import seaborn as sns

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
        return selected_ids, [str(self.img_dir / self.coco.loadImgs(img_id)[0]['file_name']) 
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
        self.results_dir = Path('results')
        self.results_dir.mkdir(exist_ok=True)
        
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
    
    def get_model_predictions(self, model, image_path: str) -> List[Dict]:
        """Get model predictions in COCO format"""
        try:
            if isinstance(model, torch.nn.Module):  # YOLOv5
                image = self.preprocess_image(image_path)
                results = model(image)
                
                # Handle different YOLOv5 result formats
                if hasattr(results, 'xyxy'):
                    predictions = results.xyxy[0].cpu().numpy()
                elif isinstance(results, torch.Tensor):
                    # Handle tensor output (older YOLOv5 versions)
                    if len(results.shape) == 3:  # [batch, num_boxes, box_attrs]
                        predictions = results[0].cpu().numpy()
                    else:  # [num_boxes, box_attrs]
                        predictions = results.cpu().numpy()
                elif hasattr(results, 'pred'):
                    # Handle newer YOLOv5 versions
                    predictions = results.pred[0].cpu().numpy()
                else:
                    # Try to get boxes from Results object
                    predictions = []
                    if len(results) > 0 and hasattr(results[0], 'boxes'):
                        boxes = results[0].boxes
                        if boxes is not None and len(boxes) > 0:
                            predictions = boxes.data.cpu().numpy()
                    
                logger.debug(f"YOLOv5 predictions shape: {predictions.shape if len(predictions) > 0 else '(0,)'}")
                
            else:  # YOLO 8,9,10,11
                results = model(image_path, conf=0.25, verbose=False)
                predictions = []
                if len(results) > 0 and hasattr(results[0], 'boxes'):
                    boxes = results[0].boxes
                    if boxes is not None and len(boxes) > 0:
                        predictions = boxes.data.cpu().numpy()
                        logger.debug(f"YOLO predictions shape: {predictions.shape}")

            # Convert to COCO format
            coco_predictions = []
            for pred in predictions:
                if len(pred) >= 6:  # Make sure we have all required values
                    x1, y1, x2, y2, conf, cls = pred[:6]
                    # Convert to COCO format and ensure all values are valid
                    bbox = [
                        max(0, float(x1)),
                        max(0, float(y1)),
                        max(0, float(x2 - x1)),
                        max(0, float(y2 - y1))
                    ]
                    
                    # Only add valid predictions
                    if all(b > 0 for b in bbox) and conf > 0:
                        coco_predictions.append({
                            'bbox': bbox,
                            'score': float(conf),
                            'category_id': int(cls) + 1  # COCO uses 1-indexed categories
                        })
            
            logger.debug(f"Number of valid predictions: {len(coco_predictions)}")
            return coco_predictions

        except Exception as e:
            logger.error(f"Error in prediction for {image_path}: {str(e)}")
            logger.debug("Stack trace:", exc_info=True)
            return []
            
    def evaluate_accuracy(self, model, image_ids: List[int]) -> Dict[str, float]:
        """Evaluate model accuracy using COCO metrics"""
        predictions = []
        total_correct = 0
        total_predictions = 0
        
        for img_id in image_ids:
            img_info = self.coco_handler.coco.loadImgs(img_id)[0]
            img_path = str(self.coco_handler.img_dir / img_info['file_name'])
            
            # Get predictions for this image
            detections = self.get_model_predictions(model, img_path)
            
            # Get ground truth annotations
            ann_ids = self.coco_handler.coco.getAnnIds(imgIds=img_id)
            annotations = self.coco_handler.coco.loadAnns(ann_ids)
            
            # Count correct predictions (IoU > 0.5)
            for det in detections:
                det['image_id'] = img_id
                predictions.append(det)
                
                # Convert detection bbox to [x1, y1, x2, y2] format
                det_bbox = det['bbox']
                det_bbox = [
                    det_bbox[0],
                    det_bbox[1],
                    det_bbox[0] + det_bbox[2],
                    det_bbox[1] + det_bbox[3]
                ]
                
                # Check against ground truth
                for ann in annotations:
                    if ann['category_id'] == det['category_id']:
                        gt_bbox = ann['bbox']
                        gt_bbox = [
                            gt_bbox[0],
                            gt_bbox[1],
                            gt_bbox[0] + gt_bbox[2],
                            gt_bbox[1] + gt_bbox[3]
                        ]
                        
                        
                        # Calculate IoU
                        iou = self.calculate_iou(det_bbox, gt_bbox)
                        if iou > 0.5:
                            total_correct += 1
                            break
                
                total_predictions += 1
        
        # Calculate accuracy
        accuracy = total_correct / max(total_predictions, 1)
        
        # If no predictions were made, return zeros
        if not predictions:
            return {
                'mAP': 0.0,
                'accuracy': 0.0
            }
        
        # Create COCO results object
        coco_dt = self.coco_handler.coco.loadRes(predictions)
        
        # Run COCO evaluation
        cocoEval = COCOeval(self.coco_handler.coco, coco_dt, 'bbox')
        cocoEval.params.imgIds = image_ids
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()
        
        return {
            'mAP': cocoEval.stats[0],
            'accuracy': accuracy
        }
        
    def calculate_iou(self, bbox1: List[float], bbox2: List[float]) -> float:
        """Calculate Intersection over Union between two bounding boxes"""
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        
        union = area1 + area2 - intersection
        
        return intersection / max(union, 1e-6)

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

    def plot_metrics(self, results: Dict[str, Dict]):
        """Plot comparison graphs for the models"""
        # Speed vs Accuracy plot
        plt.figure(figsize=(12, 6))
        models = list(results.keys())
        fps = [results[m]['fps'] for m in models]
        map_values = [results[m]['accuracy'] * 100 for m in models]  # Convert to percentage
        
        plt.scatter(fps, map_values, s=100)
        for i, model in enumerate(models):
            plt.annotate(model, (fps[i], map_values[i]))
            
        plt.xlabel('FPS')
        plt.ylabel('Accuracy (%)')
        plt.title('Speed vs Accuracy Trade-off')
        plt.grid(True)
        plt.savefig(self.results_dir / 'speed_vs_accuracy.png')
        plt.close()
        
        # Detailed metrics comparison
        metrics = ['mAP',  'accuracy']
        values = {model: [results[model][m] * 100 for m in metrics] for model in models}
        
        plt.figure(figsize=(15, 8))
        x = np.arange(len(metrics))
        width = 0.8 / len(models)
        
        for i, (model, vals) in enumerate(values.items()):
            plt.bar(x + i * width, vals, width, label=model)
            
        plt.xlabel('Metrics')
        plt.ylabel('Performance (%)')
        plt.title('Detailed Performance Metrics Comparison')
        plt.xticks(x + width * (len(models) - 1) / 2, metrics, rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.grid(True)
        plt.savefig(self.results_dir / 'detailed_metrics.png')
        plt.close()

    def evaluate_models(self, num_images: int = 100, num_runs: int = 100):
        """Evaluate all loaded models on COCO dataset"""
        if not self.models:
            logger.error("No models loaded. Please load models first.")
            return
        
        # Store results for all models
        results = {}
        
        # Get random subset of COCO images
        image_ids, test_images = self.coco_handler.get_random_images(num_images)
        
        # First collect all results
        for model_name, model in self.models.items():
            logger.info(f"Evaluating {model_name}...")
            try:
                # Measure speed
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
                    
                    # Measure accuracy
                    accuracy_metrics = self.evaluate_accuracy(model, image_ids)
                    
                    results[model_name] = {
                        'inference_time': avg_inference_time * 1000,  # Convert to ms
                        'fps': fps,
                        'images_processed': num_processed,
                        **accuracy_metrics  # Include all accuracy metrics
                    }
                
            except Exception as e:
                logger.error(f"Error evaluating {model_name}: {str(e)}")
                logger.error("Stack trace:", exc_info=True)
        
        # Plot comparison graphs
        self.plot_metrics(results)
        
        # Print results
        self.print_evaluation_results(results)
        
    def print_evaluation_results(self, results: Dict[str, Dict]):
        """Print formatted evaluation results"""
        logger.info("\n" + "="*50)
        logger.info("EVALUATION RESULTS")
        logger.info("="*50)
        
        system_info = self.get_system_info()
        logger.info("\nSystem Information:")
        for key, value in system_info.items():
            logger.info(f"{key}: {value}")
        
        # Print results table
        logger.info("\nModel Performance:")
        logger.info("-" * 100)
        logger.info(f"{'Model Name':<15} {'Time (ms)':<10} {'FPS':<8} {'mAP':<8} {'Accuracy':<8} {'Images':<8}")
        logger.info("-" * 100)
        
        # Sort models by accuracy
        sorted_models = sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True)
        
        for model_name, result in sorted_models:
            logger.info(
                f"{model_name:<15} "
                f"{result['inference_time']:<10.2f} "
                f"{result['fps']:<8.2f} "
                f"{result['mAP']*100:<8.2f} "
                f"{result['accuracy']*100:<8.2f} "
                f"{result['images_processed']:<8}"
            )
        
        logger.info("-" * 100)
        logger.info("\nGraphs have been saved to the 'results' directory")



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

#Run file 
#python evaluate_models-coco.py --coco-path ../coco --num-images 100 --weights-dir ../sh/weights --batch-size 16 --num-runs 20