import time
import torch
import numpy as np
from PIL import Image
from pathlib import Path
import psutil
import cpuinfo
from typing import Dict, List, Tuple, Optional
import logging
import cv2
import warnings
import matplotlib.pyplot as plt
from typing import Generator
import torchvision.transforms as transforms
import os

warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VideoHandler:
    def __init__(self, video_dir: str):
        self.video_dir = Path(video_dir)
        if not self.video_dir.exists():
            raise FileNotFoundError(f"Video directory not found: {video_dir}")
        
        self.video_files = list(self.video_dir.glob('*.mp4')) + \
                          list(self.video_dir.glob('*.avi')) + \
                          list(self.video_dir.glob('*.mov'))
        
        if not self.video_files:
            raise FileNotFoundError(f"No video files found in {video_dir}")
            
        logger.info(f"Found {len(self.video_files)} video files")
        
        # Create frames directory
        self.frames_dir = self.video_dir / 'frames'
        self.frames_dir.mkdir(exist_ok=True)
        
    def extract_frames(self, target_fps: float = 2.0) -> List[Path]:
        """Extract frames from videos at specified FPS"""
        extracted_frames = []
        
        for video_file in self.video_files:
            logger.info(f"\nProcessing {video_file.name}")
            
            # Create directory for this video's frames
            video_frames_dir = self.frames_dir / video_file.stem
            video_frames_dir.mkdir(exist_ok=True)
            
            cap = cv2.VideoCapture(str(video_file))
            
            # Get video properties
            original_fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / original_fps
            
            logger.info(f"Video properties:")
            logger.info(f"- Duration: {duration:.2f} seconds")
            logger.info(f"- Original FPS: {original_fps}")
            logger.info(f"- Total frames: {frame_count}")
            
            # Calculate frame sampling interval
            frame_interval = int(original_fps / target_fps)
            expected_frames = int(duration * target_fps)
            
            logger.info(f"Extracting frames at {target_fps} FPS")
            logger.info(f"Expected number of frames: {expected_frames}")
            
            frame_count = 0
            saved_count = 0
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_count % frame_interval == 0:
                    # Generate frame filename with timestamp
                    timestamp = frame_count / original_fps
                    frame_file = video_frames_dir / f"frame_{timestamp:.3f}s.jpg"
                    
                    # Save frame
                    cv2.imwrite(str(frame_file), frame)
                    extracted_frames.append(frame_file)
                    saved_count += 1
                    
                    # Show progress
                    if saved_count % 10 == 0:
                        logger.info(f"Saved {saved_count} frames...")
                
                frame_count += 1
            
            cap.release()
            logger.info(f"Completed {video_file.name}: saved {saved_count} frames")
        
        logger.info(f"\nTotal frames extracted: {len(extracted_frames)}")
        return extracted_frames

    def get_frame_paths(self) -> List[Path]:
        """Get paths of all extracted frames"""
        frames = []
        for video_dir in self.frames_dir.iterdir():
            if video_dir.is_dir():
                frames.extend(list(video_dir.glob('*.jpg')))
        return sorted(frames)

class ModelEvaluator:
    def __init__(self, video_handler: VideoHandler):
        self.device = torch.device('cpu')
        self.models: Dict[str, torch.nn.Module] = {}
        self.weights_dir = Path('weights')
        self.video_handler = video_handler
        self.transform = self.get_transform()
        self.results_dir = Path('results')
        self.results_dir.mkdir(exist_ok=True)
        
    def get_transform(self):
        return transforms.Compose([
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
        weight_files = list(self.weights_dir.glob('*.pt'))+list(self.weights_dir.glob('*.pth'))
        
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

    def preprocess_frame(self, frame: np.ndarray) -> torch.Tensor:
        """Preprocess video frame for inference"""
        image = Image.fromarray(frame)
        return self.transform(image).unsqueeze(0)
    
    def get_model_predictions(self, model, frame: np.ndarray) -> List[Dict]:
        """Get model predictions for a video frame"""
        try:
            if isinstance(model, torch.nn.Module):  # YOLOv5
                image = self.preprocess_frame(frame)
                results = model(image)
                
                if hasattr(results, 'pred'):
                    predictions = results.pred[0].cpu().numpy()
                else:
                    predictions = []
                    if len(results) > 0 and hasattr(results[0], 'boxes'):
                        boxes = results[0].boxes
                        if boxes is not None and len(boxes) > 0:
                            predictions = boxes.data.cpu().numpy()
                    
            else:  # YOLO 8,9,10,11
                results = model(frame, conf=0.25, verbose=False)
                predictions = []
                if len(results) > 0 and hasattr(results[0], 'boxes'):
                    boxes = results[0].boxes
                    if boxes is not None and len(boxes) > 0:
                        predictions = boxes.data.cpu().numpy()

            # Convert to standard format
            standard_predictions = []
            for pred in predictions:
                if len(pred) >= 6:  # Make sure we have all required values
                    x1, y1, x2, y2, conf, cls = pred[:6]
                    standard_predictions.append({
                        'bbox': [float(x1), float(y1), float(x2), float(y2)],
                        'score': float(conf),
                        'class_id': int(cls)
                    })
            
            return standard_predictions

        except Exception as e:
            logger.error(f"Error in prediction: {str(e)}")
            logger.debug("Stack trace:", exc_info=True)
            return []

    def measure_inference_time(self, model, frame: np.ndarray, num_runs: int = 10) -> float:
        """Measure average inference time"""
        times = []
        
        # # Warmup
        # for _ in range(num_runs):
        #     with torch.no_grad():
        #         _ = self.get_model_predictions(model, frame)
        # Single warmup run
        with torch.no_grad():
            _ = self.get_model_predictions(model, frame)
        
        # Actual measurement
        for _ in range(num_runs):
            start_time = time.time()
            with torch.no_grad():
                _ = self.get_model_predictions(model, frame)
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
        """Plot performance metrics"""
        plt.figure(figsize=(12, 6))
        models = list(results.keys())
        fps_values = [results[m]['fps'] for m in models]
        avg_detections = [results[m]['avg_detections'] for m in models]
        
        plt.scatter(fps_values, avg_detections, s=100)
        for i, model in enumerate(models):
            plt.annotate(model, (fps_values[i], avg_detections[i]))
            
        plt.xlabel('FPS')
        plt.ylabel('Average Detections per Frame')
        plt.title('Speed vs Detection Rate')
        plt.grid(True)
        plt.savefig(self.results_dir / 'performance_metrics.png')
        plt.close()

    def evaluate_models(self, sample_rate: int = 30, num_runs: int = 10):
        """Evaluate all loaded models on video frames"""
        if not self.models:
            logger.error("No models loaded. Please load models first.")
            return
        
        results = {}
        frame_paths = self.video_handler.get_frame_paths()
        
        # Process frames in batches to manage memory
        batch_size = 10
        
        for model_name, model in self.models.items():
            logger.info(f"\nEvaluating {model_name}...")
            
            total_time = 0
            total_frames = 0
            total_detections = 0
            
            try:
                # First, measure average inference time on a subset of frames
                sample_frames = frame_paths[::len(frame_paths)//min(10, len(frame_paths))]
                avg_inference_time = 0
                
                for frame_path in sample_frames[:5]:  # Use only 5 frames for timing
                    frame = cv2.imread(str(frame_path))
                    if frame is None:
                        continue
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    avg_inference_time += self.measure_inference_time(model, frame, num_runs)
                
                avg_inference_time /= len(sample_frames[:5])
                
                # Now process all frames for detection count
                for i in range(0, len(frame_paths), batch_size):
                    batch_paths = frame_paths[i:i + batch_size]
                    
                    for frame_path in batch_paths:
                        frame = cv2.imread(str(frame_path))
                        if frame is None:
                            continue
                        
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        predictions = self.get_model_predictions(model, frame)
                        total_detections += len(predictions)
                        total_frames += 1
                    
                    if total_frames % 50 == 0:
                        logger.info(f"Processed {total_frames} frames...")
                    
                    # Clear memory
                    if hasattr(torch, 'cuda'):
                        torch.cuda.empty_cache()
                
                if total_frames > 0:
                    fps = 1 / avg_inference_time
                    avg_detections = total_detections / total_frames
                    
                    results[model_name] = {
                        'inference_time': avg_inference_time * 1000,  # Convert to ms
                        'fps': fps,
                        'frames_processed': total_frames,
                        'total_detections': total_detections,
                        'avg_detections': avg_detections
                    }
                
            except Exception as e:
                logger.error(f"Error evaluating {model_name}: {str(e)}")
                logger.debug("Stack trace:", exc_info=True)
        
        # Plot comparison graphs
        self.plot_metrics(results)
        # Print results
        self.print_evaluation_results(results)
        
    def print_evaluation_results(self, results: Dict[str, Dict]):
        """Print formatted evaluation results"""
        logger.info("\n" + "="*60)
        logger.info("VIDEO EVALUATION RESULTS")
        logger.info("="*60)
        
        system_info = self.get_system_info()
        logger.info("\nSystem Information:")
        for key, value in system_info.items():
            logger.info(f"{key}: {value}")
        
        logger.info("\nModel Performance:")
        logger.info("-" * 120)
        logger.info(f"{'Model Name':<15} {'Time (ms)':<10} {'FPS':<8} {'Frames':<8} {'Total Det.':<10} {'Avg Det.':<8}")
        logger.info("-" * 120)
        
        # Sort models by FPS
        sorted_models = sorted(results.items(), key=lambda x: x[1]['fps'], reverse=True)
        
        for model_name, result in sorted_models:
            logger.info(
                f"{model_name:<15} "
                f"{result['inference_time']:<10.2f} "
                f"{result['fps']:<8.2f} "
                f"{result['frames_processed']:<8} "
                f"{result['total_detections']:<10} "
                f"{result['avg_detections']:<8.2f}"
            )
        
        logger.info("-" * 120)
        logger.info("\nPerformance graph has been saved to the 'results' directory")


def parse_args():
    """Parse command line arguments"""
    import argparse
    parser = argparse.ArgumentParser(description='Extract frames and evaluate object detection models')
    
    parser.add_argument('--video-dir', 
                        type=str, 
                        required=True,
                        default='../sample_video',
                        help='Path to directory containing video files')
    
    parser.add_argument('--weights-dir', 
                        type=str, 
                        default='../weights',
                        help='Directory containing model weights (default: weights)')
    
    parser.add_argument('--target-fps', 
                        type=float, 
                        default=2.0,
                        help='Target FPS for frame extraction (default: 2.0)')
    
    parser.add_argument('--num-runs', 
                        type=int, 
                        default=10,
                        help='Number of runs for FPS calculation (default: 100)')
    
    parser.add_argument('--skip-extraction',
                        action='store_true',
                        help='Skip frame extraction if frames already exist')
    
    return parser.parse_args()

def main():
    # Parse arguments
    args = parse_args()
    
    # Initialize video handler
    try:
        video_handler = VideoHandler(args.video_dir)
    except FileNotFoundError as e:
        logger.error(f"Error: {e}")
        return
    
    # Extract frames if needed
    if not args.skip_extraction or not list(video_handler.frames_dir.glob('*/*.jpg')):
        logger.info(f"Extracting frames at {args.target_fps} FPS...")
        extracted_frames = video_handler.extract_frames(target_fps=args.target_fps)
    else:
        logger.info("Using existing extracted frames...")
        extracted_frames = video_handler.get_frame_paths()
    
    if not extracted_frames:
        logger.error("No frames available for evaluation")
        return
    
    # Create evaluator
    evaluator = ModelEvaluator(video_handler)
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
    
    # Run evaluation on extracted frames
    evaluator.evaluate_models(num_runs=args.num_runs)

if __name__ == "__main__":
    main()
    
    
# Run file   
# python evaluate-models-video.py  --video-dir ../sample_video --weights-dir ../sh/weights --target-fps 2 --num-runs 2