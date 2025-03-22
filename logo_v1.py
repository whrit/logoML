import os
import glob
import json
import xml.etree.ElementTree as ET
import torch
from PIL import Image
import clip
import cv2
from tqdm import tqdm
import yaml
import random
import shutil
from ultralytics import YOLO
from datetime import datetime
import time
import concurrent.futures
from functools import partial
import gc
import psutil
import platform
import queue
import threading
import numpy as np

# =============================================================================
# PART 0: GPU & System Configuration
# =============================================================================

def setup_gpu_environment():
    """
    Set up GPU environment based on available hardware (CUDA or MPS)
    Returns device for PyTorch operations
    """
    print("Setting up GPU environment...")
    
    # Check system info
    system_info = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else "N/A",
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "mps_available": hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
    }
    
    print("System information:")
    for key, value in system_info.items():
        print(f"  {key}: {value}")
    
    # Set device priority: CUDA > MPS > CPU
    if torch.cuda.is_available():
        # NVIDIA GPU
        device = torch.device("cuda")
        
        # Set CUDA device properties for maximum performance
        torch.backends.cudnn.benchmark = CONFIG["cudnn_benchmark"]
        torch.backends.cudnn.deterministic = False
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        # Print available GPU(s)
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_mem = torch.cuda.get_device_properties(i).total_memory / 1024**3  # GB
            print(f"  CUDA GPU {i}: {gpu_name} with {gpu_mem:.2f} GB memory")
            
            # A100-specific optimizations
            if 'a100' in gpu_name.lower():
                print("  Detected NVIDIA A100 GPU - applying specialized optimizations")
                # Optimize CUDA memory allocator settings
                try:
                    # Use memory allocator settings instead of non-existent stream cache limit
                    torch.cuda.memory._set_allocator_settings("expandable_segments:True")
                    # Set max split size for better memory utilization on A100
                    torch.cuda.memory._set_allocator_settings(f"max_split_size_mb:{CONFIG['max_split_size_mb']}")
                    print("  Applied A100-specific memory allocator settings")
                except Exception as e:
                    print(f"  Warning: Could not apply CUDA memory optimizations: {e}")
                    
                # Try to reserve more L2 cache for tensors if using CUDA 11.4+
                try:
                    if hasattr(torch.cuda.memory, 'set_per_device_cache_limit'):
                        torch.cuda.memory.set_per_device_cache_limit(1024 * 1024 * 1024)  # 1GB L2 cache
                        print("  Set 1GB L2 cache limit for A100")
                    else:
                        print("  L2 cache configuration not available in this PyTorch version")
                except Exception as e:
                    print(f"  Warning: Could not configure L2 cache: {e}")
                    
                # For A100, BF16 is typically better than FP16
                CONFIG["use_bf16"] = True
        
        # Set optimal CUDA configurations
        print("  Configuring CUDA memory settings...")
        try:
            # Use current GPU memory allocation strategy (best for newer GPUs)
            torch.cuda.memory._set_allocator_settings("expandable_segments:True")
            print("  Enabled expandable segments for memory allocation")
        except Exception as e:
            print(f"  Warning: Could not configure expandable segments: {e}")
        
        # Enable CUDA graphs if available
        if 'cuda_graphs' in CONFIG and CONFIG['cuda_graphs']:
            try:
                # Different ways to enable CUDA graphs depending on PyTorch version
                if hasattr(torch.cuda, '_C'):
                    torch.cuda._C._set_graph_executor_enabled(True)
                    print("  Enabled CUDA graph execution via torch.cuda._C")
                elif hasattr(torch, '_C') and hasattr(torch._C, '_cuda_setGraphExecutorEnabled'):
                    torch._C._cuda_setGraphExecutorEnabled(True)
                    print("  Enabled CUDA graph execution via torch._C")
                else:
                    print("  CUDA graph execution not supported in this PyTorch version")
                    CONFIG['cuda_graphs'] = False
            except Exception as e:
                print(f"  Warning: Could not enable CUDA graphs: {e}")
                # If CUDA graphs failed, disable it in config
                CONFIG['cuda_graphs'] = False
            
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        # Apple Silicon GPU
        device = torch.device("mps")
        print("  Using Apple MPS (Metal Performance Shaders) for acceleration")
    else:
        # CPU fallback
        device = torch.device("cpu")
        print("  No GPU detected, using CPU. Performance will be limited.")
        
        # Set number of threads based on available CPU cores
        cpu_count = os.cpu_count()
        torch.set_num_threads(cpu_count)
        print(f"  CPU threads configured: {cpu_count}")
    
    # Memory information
    mem_info = psutil.virtual_memory()
    print(f"  System memory: {mem_info.total / (1024**3):.2f} GB total, {mem_info.available / (1024**3):.2f} GB available")
    
    print(f"Device selected: {device}")
    return device

# Configuration
CONFIG = {
    # Paths
    "dataset_root": "data/LogoDet-3K",            # Path to original LogoDet-3K dataset
    "output_root": "data/logodet3k_prepared",     # Path to save prepared dataset
    "results_dir": "results",                         # Directory to save results
    
    # Training settings
    "binary": True,                                   # Train for binary logo detection
    "epochs": 100,                                    # Number of training epochs
    "batch_size": 32,                                 # Batch size - increased for A100
    "img_size": 640,                                  # Image size for YOLOv11 - increased for A100
    "learning_rate": 0.001,                           # Initial learning rate
    
    # Model settings
    "yolo_model": "yolo11m.pt",                       # YOLOv11 model to use
    "clip_model": "ViT-B/32",                         # CLIP model to use
    
    # Detector settings
    "yolo_conf": 0.25,                                # YOLOv11 confidence threshold
    "clip_conf": 0.5,                                 # CLIP confidence threshold
    
    # Test settings
    "num_test_images": 10,                            # Number of test images to visualize
    "visualize_results": True,                        # Whether to visualize results
    
    # Performance settings
    "num_workers": 16,                                # Number of workers for data loading
    "prefetch_factor": 4,                             # Prefetch factor for data loading
    "pin_memory": True,                               # Pin memory for faster GPU transfer
    "mixed_precision": True,                          # Use mixed precision (FP16/BF16)
    "amp_scaler": True,                               # Use automatic mixed precision (AMP) scaler
    "threads_per_process": 8,                         # Number of threads per process
    "max_split_size_mb": 1024,                        # Max size in MB for splitting operations
    "compile_model": True,                            # Use torch.compile (requires PyTorch 2.0+)
    "use_bf16": True,                                 # Use BF16 instead of FP16 (better for A100)
    "cuda_graphs": True,                              # Enable CUDA Graphs for faster training
    "memory_format": "channels_last",                 # Use channels_last memory format for better performance
    "cudnn_benchmark": True,                          # Enable cudnn benchmark
    "multi_gpu": False,                               # Enable multi-GPU training if needed
}

def update_config_for_system():
    """
    Dynamically update configuration based on detected system resources
    Handles scaling for everything from A100 GPUs to MacBook Pro MPS
    """
    print("Detecting system resources for optimal configuration...")
    
    # Detect CPU resources
    cpu_count = os.cpu_count()
    cpu_threads = psutil.cpu_count(logical=True)
    
    # Detect memory resources
    mem_info = psutil.virtual_memory()
    total_ram_gb = mem_info.total / (1024**3)
    available_ram_gb = mem_info.available / (1024**3)
    
    # Detect GPU type and resources
    gpu_type = "unknown"
    gpu_name = "none"
    gpu_mem_gb = 0
    has_tensor_cores = False
    
    if torch.cuda.is_available():
        # NVIDIA GPU
        gpu_type = "cuda"
        gpu_count = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        
        # Check for Tensor Cores (Volta, Turing, Ampere, Ada, or Hopper architecture)
        has_tensor_cores = any(arch in gpu_name.lower() for arch in 
                              ['tesla v', 'tesla t', 'tesla a', 'a100', 'a10', 'a40', 'a30',
                               'a800', 'h100', 'rtx', 'titan v', 'titan rtx', 'quadro rtx',
                               'quadro gv'])
        
        # Special case for A100 and H100
        is_a100 = 'a100' in gpu_name.lower()
        is_h100 = 'h100' in gpu_name.lower()
        
        print(f"Detected NVIDIA GPU: {gpu_name} with {gpu_mem_gb:.1f}GB memory")
        
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        # Apple Silicon
        gpu_type = "mps"
        
        # Try to detect Apple Silicon model
        try:
            import subprocess
            result = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'], 
                                   capture_output=True, text=True)
            if 'Apple M1' in result.stdout:
                gpu_name = 'Apple M1'
                # Approximate GPU memory for M1 chips (shared memory)
                gpu_mem_gb = min(8.0, total_ram_gb / 4)
            elif 'Apple M2' in result.stdout:
                gpu_name = 'Apple M2'
                gpu_mem_gb = min(10.0, total_ram_gb / 4)
            elif 'Apple M3' in result.stdout:
                gpu_name = 'Apple M3'
                gpu_mem_gb = min(12.0, total_ram_gb / 4)
            else:
                gpu_name = 'Apple Silicon'
                gpu_mem_gb = min(8.0, total_ram_gb / 4)
        except:
            gpu_name = 'Apple Silicon'
            gpu_mem_gb = min(8.0, total_ram_gb / 4)
            
        print(f"Detected {gpu_name} with approximately {gpu_mem_gb:.1f}GB shared GPU memory")
        
    else:
        # CPU only
        gpu_type = "cpu"
        print("No GPU detected, using CPU only")
    
    # Display resource summary
    print(f"System resources summary:")
    print(f"  CPU cores: {cpu_count} physical, {cpu_threads} logical")
    print(f"  RAM: {total_ram_gb:.1f}GB total, {available_ram_gb:.1f}GB available")
    print(f"  GPU: {gpu_name} ({gpu_type}) with {gpu_mem_gb:.1f}GB memory")
    
    # -----------------------------------------------------------------
    # Dynamically scale configuration parameters based on resources
    # -----------------------------------------------------------------
    
    # 1. Batch size scaling - primarily dependent on GPU memory
    original_batch_size = CONFIG["batch_size"]
    
    if gpu_type == "cuda":
        # NVIDIA GPU scaling
        if gpu_mem_gb >= 80:  # H100, A100 80GB, etc.
            CONFIG["batch_size"] = original_batch_size * 8
        elif gpu_mem_gb >= 40:  # A100 40GB, A40, etc.
            CONFIG["batch_size"] = original_batch_size * 6
        elif gpu_mem_gb >= 24:  # RTX 3090, RTX 4090, etc.
            CONFIG["batch_size"] = original_batch_size * 4
        elif gpu_mem_gb >= 16:  # RTX 3080, etc.
            CONFIG["batch_size"] = original_batch_size * 3
        elif gpu_mem_gb >= 12:  # RTX 3060, etc.
            CONFIG["batch_size"] = original_batch_size * 2
        elif gpu_mem_gb >= 8:   # Older GPUs
            CONFIG["batch_size"] = original_batch_size
        else:  # Low memory GPUs
            CONFIG["batch_size"] = max(1, original_batch_size // 2)
            
    elif gpu_type == "mps":
        # Apple Silicon - be more conservative due to shared memory
        if gpu_mem_gb >= 10:  # Higher-end Apple Silicon
            CONFIG["batch_size"] = original_batch_size
        else:  # Base models
            CONFIG["batch_size"] = max(1, original_batch_size // 2)
            
    else:
        # CPU only - reduce batch size significantly
        CONFIG["batch_size"] = max(1, original_batch_size // 4)
    
    # 2. Number of workers scaling - primarily dependent on CPU cores
    if cpu_count:
        if gpu_type == "cuda" and gpu_mem_gb >= 16:
            # High-end systems can use more workers
            CONFIG["num_workers"] = min(16, max(4, cpu_count - 2))
        else:
            # More conservative for other systems
            CONFIG["num_workers"] = min(8, max(2, cpu_count - 1))
    
    # 3. Image size scaling - affects both GPU memory and processing time
    if gpu_type == "cuda" and gpu_mem_gb >= 24:
        # High-end GPUs can handle larger image sizes
        CONFIG["img_size"] = 640  # Increased from 640
    else:
        # Keep default for other systems
        pass
    
    # 4. Mixed precision settings
    if gpu_type == "cuda":
        if has_tensor_cores:
            # Full mixed precision support
            CONFIG["mixed_precision"] = True
            CONFIG["amp_scaler"] = True
        else:
            # Older NVIDIA GPUs
            CONFIG["mixed_precision"] = True
            CONFIG["amp_scaler"] = True
    elif gpu_type == "mps":
        # MPS support for mixed precision is limited
        CONFIG["mixed_precision"] = False
        CONFIG["amp_scaler"] = False
    else:
        # CPU doesn't support mixed precision
        CONFIG["mixed_precision"] = False
        CONFIG["amp_scaler"] = False
    
    # 5. Model compilation (PyTorch 2.0+)
    pytorch_version = tuple(map(int, torch.__version__.split('.')[:2]))
    if pytorch_version >= (2, 0):
        if gpu_type == "cuda" and has_tensor_cores:
            # Tensor Core GPUs benefit the most from compilation
            CONFIG["compile_model"] = True
        else:
            # Less benefit on other hardware
            CONFIG["compile_model"] = False
    else:
        # Not available on older PyTorch
        CONFIG["compile_model"] = False
    
    # 6. Memory optimizations
    if total_ram_gb >= 64:
        # High RAM systems
        CONFIG["prefetch_factor"] = 4
        CONFIG["max_split_size_mb"] = 1024
    elif total_ram_gb >= 32:
        # Medium RAM systems
        CONFIG["prefetch_factor"] = 3
        CONFIG["max_split_size_mb"] = 768
    elif total_ram_gb >= 16:
        # Lower RAM systems
        CONFIG["prefetch_factor"] = 2
        CONFIG["max_split_size_mb"] = 512
    else:
        # Very low RAM systems
        CONFIG["prefetch_factor"] = 1
        CONFIG["max_split_size_mb"] = 256
    
    # 7. Pin memory (beneficial for CUDA, not for MPS)
    CONFIG["pin_memory"] = (gpu_type == "cuda")
    
    # Print summary of adjusted configuration
    print("\nDynamically adjusted configuration:")
    print(f"  Batch size: {original_batch_size} → {CONFIG['batch_size']}")
    print(f"  Num workers: {CONFIG['num_workers']}")
    print(f"  Image size: {CONFIG['img_size']}")
    print(f"  Mixed precision: {CONFIG['mixed_precision']}")
    print(f"  Model compilation: {CONFIG['compile_model']}")
    print(f"  Prefetch factor: {CONFIG['prefetch_factor']}")
    print(f"  Pin memory: {CONFIG['pin_memory']}")
    
    return CONFIG

# =============================================================================
# PART 1: DATA PREPARATION WITH THREADING
# =============================================================================

def parallel_xml_to_yolo(args):
    """
    Parallelized version of XML to YOLO conversion
    
    Args:
        args: Tuple of (xml_path, output_path, brand_to_class, brand, binary)
    
    Returns:
        1 if successful, 0 if failed
    """
    xml_path, output_path, brand_to_class, brand, binary = args
    
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # Get image dimensions
        img_width = int(root.find('./size/width').text)
        img_height = int(root.find('./size/height').text)
        
        yolo_annotations = []
        
        for obj in root.findall('./object'):
            # For binary classification, use class 0 for all logos
            if binary:
                class_id = 0
            else:
                # Use the brand-to-class map
                class_id = brand_to_class.get(brand, 0)
            
            # Get bounding box coordinates
            bbox = obj.find('bndbox')
            if bbox is None:
                continue
                
            xmin = float(bbox.find('xmin').text)
            ymin = float(bbox.find('ymin').text)
            xmax = float(bbox.find('xmax').text)
            ymax = float(bbox.find('ymax').text)
            
            # Ensure coordinates are within image boundaries
            xmin = max(0, min(xmin, img_width))
            xmax = max(0, min(xmax, img_width))
            ymin = max(0, min(ymin, img_height))
            ymax = max(0, min(ymax, img_height))
            
            # Convert to YOLO format (x_center, y_center, width, height)
            x_center = (xmin + xmax) / (2 * img_width)
            y_center = (ymin + ymax) / (2 * img_height)
            width = (xmax - xmin) / img_width
            height = (ymax - ymin) / img_height
            
            # Add to annotations list
            yolo_annotations.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
        
        # Write annotations to file
        if yolo_annotations:
            with open(output_path, 'w') as f:
                for line in yolo_annotations:
                    f.write(line + '\n')
        else:
            # Create an empty file if there are no annotations
            open(output_path, 'w').close()
        
        return 1
            
    except Exception as e:
        # Create an empty file in case of errors
        open(output_path, 'w').close()
        return 0

def process_brand_directory(category, brand, brand_path, brand_to_class):
    """
    Process a single brand directory to extract image paths and metadata
    
    Args:
        category: Category name
        brand: Brand name
        brand_path: Path to the brand directory
        brand_to_class: Mapping of brand names to class IDs
    
    Returns:
        List of tuples (img_path, xml_path, category, base_brand)
    """
    # Extract normalized brand name
    base_brand = brand.split('-')[0] if '-' in brand else brand
    base_brand = base_brand.lower()
    
    # Find all JPG files in the brand folder
    image_files = glob.glob(os.path.join(brand_path, "*.jpg"))
    
    result = []
    for img_path in image_files:
        xml_path = img_path.replace('.jpg', '.xml')
        if os.path.exists(xml_path):
            # Store the image path, XML path, category, and brand
            result.append((img_path, xml_path, category, base_brand))
    
    return result

def prepare_logodet3k(dataset_root, output_root, binary=True, num_workers=4):
    """
    Prepare LogoDet-3K dataset with category/brand hierarchy for YOLO training
    Uses threading and multiprocessing for faster preparation
    
    Args:
        dataset_root: Path to the original LogoDet-3K folder
        output_root: Path where to save the prepared dataset
        binary: If True, prepare for binary classification (logo/non-logo),
                otherwise use multi-class (each logo as separate class)
        num_workers: Number of worker processes/threads to use
    
    Returns:
        yaml_content: YAML configuration
        all_brands: List of all brands
    """
    start_time = time.time()
    
    # Verify dataset_root exists
    if not os.path.exists(dataset_root):
        print(f"ERROR: Dataset root directory '{dataset_root}' does not exist!")
        print(f"Please check that the path is correct. Current working directory: {os.getcwd()}")
        print(f"Available directories: {[d for d in os.listdir('.') if os.path.isdir(d)]}")
        
        # Create a dummy YAML config and return empty brands list for graceful degradation
        # so the script can continue even with dataset issues
        dummy_yaml = {
            'path': os.path.abspath(output_root),
            'train': os.path.join('train', 'images'),
            'val': os.path.join('val', 'images'),
            'test': os.path.join('test', 'images'),
            'names': {0: 'logo'}
        }
        
        os.makedirs(output_root, exist_ok=True)
        with open(os.path.join(output_root, 'logodet3k.yaml'), 'w') as f:
            yaml.dump(dummy_yaml, f, default_flow_style=False)
            
        return dummy_yaml, []
    
    # Auto-detect optimal workers based on CPU cores
    cpu_count = os.cpu_count()
    if num_workers <= 0:
        # Use all available cores minus 1 to keep system responsive
        num_workers = max(1, cpu_count - 1)
    
    print(f"Using {num_workers} workers for dataset preparation (out of {cpu_count} CPU cores)")
    
    # Create output directories
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(output_root, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_root, split, 'labels'), exist_ok=True)
    
    # Load category mapping
    category_map = {}
    classes_path = os.path.join(dataset_root, 'classes.txt')
    if os.path.exists(classes_path):
        print(f"Found classes.txt at {classes_path}")
        with open(classes_path, 'r') as f:
            for line in f:
                parts = line.strip().split(' ', 1)
                if len(parts) == 2:
                    category_map[parts[1]] = int(parts[0])
        print(f"Loaded {len(category_map)} categories from classes.txt")
    else:
        print(f"WARNING: classes.txt not found at {classes_path}")
        print(f"Directory contents of {dataset_root}: {os.listdir(dataset_root)}")
    
    # Find all category folders
    all_items = os.listdir(dataset_root)
    category_folders = [f for f in all_items 
                       if os.path.isdir(os.path.join(dataset_root, f)) and 
                       f in category_map]
    
    print(f"Found {len(category_folders)} category folders in dataset: {category_folders}")
    
    if len(category_folders) == 0:
        print("WARNING: No valid category folders found that match entries in classes.txt")
        print("This likely means either:")
        print("  1. The dataset path is incorrect")
        print("  2. The classes.txt format doesn't match the expected format")
        print("  3. The dataset structure is different than expected")
        print("\nTrying to proceed anyway, but training may fail...")
    
    # Create brand-to-class mapping for multi-class version
    brand_to_class = {}
    class_id = 0
    
    # Create a list to store all brands for CLIP classification
    all_brands = []
    
    print("Scanning for brand folders...")
    # Find all brand folders and create class mapping
    for category in tqdm(category_folders, desc="Categories"):
        category_path = os.path.join(dataset_root, category)
        brand_folders = [f for f in os.listdir(category_path) 
                         if os.path.isdir(os.path.join(category_path, f))]
        
        for brand in brand_folders:
            # Clean up brand name (remove variant indicators)
            base_brand = brand.split('-')[0] if '-' in brand else brand
            base_brand = base_brand.lower()  # Normalize to lowercase
            
            # If this is a new brand (not a variant), add it to the class map
            if base_brand not in brand_to_class:
                brand_to_class[base_brand] = class_id
                all_brands.append(base_brand)
                class_id += 1
    
    # Save the brand mapping
    with open(os.path.join(output_root, 'brands.txt'), 'w') as f:
        for i, brand in enumerate(all_brands):
            f.write(f"{i} {brand}\n")
    
    print(f"Found {len(all_brands)} unique logo brands")
    
    # Process all images and annotations in parallel
    print("Scanning for image and annotation files...")
    
    all_image_paths = []
    
    # Using ThreadPoolExecutor with more threads for I/O-bound operations
    max_io_threads = min(num_workers * 2, 32)  # Scale I/O threads higher, but cap at 32
    
    # Chunk size for better load balancing
    chunk_size = max(10, len(category_folders) // max_io_threads)
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_io_threads) as executor:
        # Create a list to hold futures
        futures = []
        
        # Submit tasks for each brand directory
        for category in category_folders:
            category_path = os.path.join(dataset_root, category)
            brand_folders = [f for f in os.listdir(category_path) 
                             if os.path.isdir(os.path.join(category_path, f))]
            
            for brand in brand_folders:
                brand_path = os.path.join(category_path, brand)
                future = executor.submit(
                    process_brand_directory, 
                    category, 
                    brand, 
                    brand_path, 
                    brand_to_class
                )
                futures.append(future)
        
        # Collect results as they complete
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing brands"):
            results = future.result()
            all_image_paths.extend(results)
    
    # Shuffle all image paths with a fixed seed for reproducibility
    random.seed(42)
    random.shuffle(all_image_paths)
    
    # Split into train/val/test (70/15/15)
    n_files = len(all_image_paths)
    n_train = int(0.7 * n_files)
    n_val = int(0.15 * n_files)
    
    train_files = all_image_paths[:n_train]
    val_files = all_image_paths[n_train:n_train+n_val]
    test_files = all_image_paths[n_train+n_val:]
    
    print(f"Total images with annotations: {n_files}")
    print(f"Train set: {len(train_files)} images")
    print(f"Validation set: {len(val_files)} images")
    print(f"Test set: {len(test_files)} images")
    
    # Also create a brands.json file for CLIP classification
    with open(os.path.join(output_root, 'brands.json'), 'w') as f:
        json.dump(all_brands, f, indent=2)
    
    # Prepare for faster file operations with memory mapping if available
    try:
        cache_memory = psutil.virtual_memory().available > 16 * 1024**3  # If >16GB RAM available
    except:
        cache_memory = False
    
    # Process each split with optimized parallel file operations
    for split, files in [('train', train_files), ('val', val_files), ('test', test_files)]:
        print(f"Processing {split} set...")
        split_time = time.time()
        
        # Use more threads for I/O operations
        io_threads = min(num_workers * 4, 64)  # Even more threads for pure I/O
        
        # Copy images in parallel using ThreadPoolExecutor (good for I/O operations)
        with concurrent.futures.ThreadPoolExecutor(max_workers=io_threads) as executor:
            # Prepare image copy tasks
            image_tasks = []
            for img_path, _, _, _ in files:
                img_name = os.path.basename(img_path)
                out_img_path = os.path.join(output_root, split, 'images', img_name)
                image_tasks.append((img_path, out_img_path))
            
            # Execute image copy tasks in chunks for better performance
            chunk_size = max(100, len(image_tasks) // (io_threads * 2))
            list(tqdm(executor.map(lambda p: shutil.copy(*p), image_tasks, chunksize=chunk_size), 
                      total=len(image_tasks), 
                      desc=f"Copying {split} images"))
        
        # Optimize chunk size for CPU-intensive tasks based on number of workers
        cpu_chunk_size = max(10, len(files) // (num_workers * 2))
        
        # Convert XML to YOLO format in parallel using ProcessPoolExecutor (CPU-intensive)
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            # Prepare XML conversion tasks
            xml_tasks = []
            for _, xml_path, _, brand in files:
                label_name = os.path.basename(xml_path).replace('.xml', '.txt')
                out_label_path = os.path.join(output_root, split, 'labels', label_name)
                xml_tasks.append((xml_path, out_label_path, brand_to_class, brand, binary))
            
            # Execute XML conversion tasks with optimized chunk size
            results = list(tqdm(executor.map(parallel_xml_to_yolo, xml_tasks, chunksize=cpu_chunk_size), 
                               total=len(xml_tasks), 
                               desc=f"Converting {split} annotations"))
            
            success_count = sum(results)
            print(f"Successfully converted {success_count}/{len(xml_tasks)} annotations")
        
        print(f"Processed {split} set in {time.time() - split_time:.2f} seconds")
    
    # Create YAML configuration file for YOLO
    yaml_content = {
        'path': os.path.abspath(output_root),
        'train': os.path.join('train', 'images'),
        'val': os.path.join('val', 'images'),
        'test': os.path.join('test', 'images'),
    }
    
    if binary:
        yaml_content['names'] = {0: 'logo'}
    else:
        yaml_content['names'] = {i: brand for i, brand in enumerate(all_brands)}
    
    with open(os.path.join(output_root, 'logodet3k.yaml'), 'w') as f:
        yaml.dump(yaml_content, f, default_flow_style=False)
    
    print(f"Dataset prepared and saved to {output_root}")
    print(f"Configuration file created at {os.path.join(output_root, 'logodet3k.yaml')}")
    print(f"Brand list saved to {os.path.join(output_root, 'brands.json')} for CLIP classification")
    print(f"Total preparation time: {time.time() - start_time:.2f} seconds")
    
    return yaml_content, all_brands

# =============================================================================
# PART 2: CLIP-BASED ZERO-SHOT CLASSIFIER WITH GPU OPTIMIZATION
# =============================================================================

class ClipClassifier:
    """
    CLIP-based zero-shot classifier for logo classification
    Optimized for GPU acceleration
    """
    def __init__(self, 
                 logo_classes=None, 
                 model_name="ViT-B/32", 
                 device=None, 
                 batch_size=32,
                 mixed_precision=True):
        """
        Initialize the CLIP classifier
        
        Args:
            logo_classes: List of logo class names
            model_name: CLIP model name
            device: PyTorch device (cuda, mps, or cpu)
            batch_size: Batch size for classification
            mixed_precision: Whether to use mixed precision
        """
        # Set device
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Set batch size - increase for A100
        if self.device.type == 'cuda' and 'a100' in torch.cuda.get_device_name(0).lower():
            # A100 can handle much larger batches
            self.batch_size = batch_size * 4
            print(f"A100 GPU detected - increasing CLIP batch size to {self.batch_size}")
        else:
            self.batch_size = batch_size
        
        # Determine precision mode
        if self.device.type == 'cuda' and mixed_precision:
            if CONFIG.get('use_bf16', False) and torch.cuda.is_bf16_supported():
                self.mixed_precision = 'bf16'
                print("Using BF16 precision for CLIP (optimal for A100)")
            else:
                self.mixed_precision = 'fp16'
                print("Using FP16 precision for CLIP")
        else:
            self.mixed_precision = False
        
        print(f"Initializing CLIP classifier on {self.device}")
        
        # Load CLIP model
        self.model, self.preprocess = clip.load(model_name, device=self.device, jit=False)
        
        # Apply memory format optimization for CUDA devices
        if self.device.type == 'cuda' and CONFIG.get('memory_format', '') == 'channels_last':
            try:
                self.model = self.model.to(memory_format=torch.channels_last)
                print("Applied channels_last memory format to CLIP model")
            except:
                print("Failed to apply channels_last memory format")
        
        # Apply PyTorch 2.0+ compile if available and on CUDA
        if hasattr(torch, 'compile') and self.device.type == 'cuda' and CONFIG.get("compile_model", False):
            try:
                print("Compiling CLIP model with torch.compile()...")
                self.model = torch.compile(self.model)
                print("Successfully compiled CLIP model")
            except Exception as e:
                print(f"Failed to compile CLIP model: {e}")
        
        # Set model to evaluation mode
        self.model.eval()
        
        # If logo classes are not provided, extract them from the dataset
        self.logo_classes = logo_classes if logo_classes else []
        print(f"Loaded {len(self.logo_classes)} logo classes for zero-shot classification")
        
        # Create text embeddings for all logo classes
        self.create_text_embeddings()
    
    def create_text_embeddings(self):
        """
        Create text embeddings for all logo classes using CLIP
        Optimized for batch processing
        """
        if not self.logo_classes:
            print("No logo classes provided for CLIP classification")
            self.text_embeddings = None
            return
        
        # Create prompts for each logo class
        prompts = [
            f"a logo of {logo_class}" for logo_class in self.logo_classes
        ]
        
        # Process in batches to avoid OOM for large number of classes
        # A100 can handle larger batch sizes
        batch_size = 2048 if self.device.type == 'cuda' and 'a100' in torch.cuda.get_device_name(0).lower() else 1024
        text_embeddings = []
        
        with torch.no_grad():
            for i in range(0, len(prompts), batch_size):
                # Get batch
                batch_prompts = prompts[i:i+batch_size]
                
                # Tokenize batch
                batch_tokens = clip.tokenize(batch_prompts).to(self.device)
                
                # Use mixed precision if enabled
                if self.mixed_precision == 'bf16':
                    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                        batch_embeddings = self.model.encode_text(batch_tokens).float()
                elif self.mixed_precision == 'fp16':
                    with torch.cuda.amp.autocast():
                        batch_embeddings = self.model.encode_text(batch_tokens).float()
                else:
                    batch_embeddings = self.model.encode_text(batch_tokens).float()
                
                # Normalize batch embeddings
                batch_embeddings = batch_embeddings / batch_embeddings.norm(dim=-1, keepdim=True)
                
                # Append to list
                text_embeddings.append(batch_embeddings)
        
        # Concatenate all batches
        self.text_embeddings = torch.cat(text_embeddings, dim=0)
    
    def classify_image(self, image):
        """
        Classify a single image using CLIP
        
        Args:
            image: PIL Image object
        
        Returns:
            predicted_class: str, the predicted logo class
            confidence: float, confidence score
        """
        # Preprocess the image
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)
        
        # Generate image embedding
        with torch.no_grad():
            if self.mixed_precision == 'bf16':
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    image_embedding = self.model.encode_image(image_input).float()
            elif self.mixed_precision == 'fp16':
                with torch.cuda.amp.autocast():
                    image_embedding = self.model.encode_image(image_input).float()
            else:
                image_embedding = self.model.encode_image(image_input).float()
        
        # Normalize image embedding
        image_embedding = image_embedding / image_embedding.norm(dim=-1, keepdim=True)
        
        # Calculate similarity scores
        similarity = torch.matmul(image_embedding, self.text_embeddings.T)
        
        # Get the predicted class
        class_idx = similarity.argmax().item()
        confidence = similarity.max().item()
        
        return self.logo_classes[class_idx], confidence
    
    def classify_batch(self, images):
        """
        Classify a batch of images using CLIP
        Optimized for GPU parallel processing
        
        Args:
            images: List of PIL Image objects
        
        Returns:
            predicted_classes: List of predicted logo classes
            confidences: List of confidence scores
        """
        if not images:
            return [], []
        
        # Process in batches to avoid OOM
        predicted_classes = []
        confidences = []
        
        # Process in smaller batches
        for i in range(0, len(images), self.batch_size):
            batch_images = images[i:i+self.batch_size]
            
            # Preprocess the images
            batch_inputs = torch.stack([self.preprocess(img) for img in batch_images]).to(self.device)
            
            # Generate image embeddings
            with torch.no_grad():
                if self.mixed_precision == 'bf16':
                    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                        batch_embeddings = self.model.encode_image(batch_inputs).float()
                elif self.mixed_precision == 'fp16':
                    with torch.cuda.amp.autocast():
                        batch_embeddings = self.model.encode_image(batch_inputs).float()
                else:
                    batch_embeddings = self.model.encode_image(batch_inputs).float()
            
            # Normalize image embeddings
            batch_embeddings = batch_embeddings / batch_embeddings.norm(dim=-1, keepdim=True)
            
            # Calculate similarity scores
            batch_similarities = torch.matmul(batch_embeddings, self.text_embeddings.T)
            
            # Get the predicted classes
            batch_class_indices = batch_similarities.argmax(dim=1)
            batch_confidences = batch_similarities.max(dim=1).values
            
            # Convert to lists
            batch_predicted = [self.logo_classes[idx] for idx in batch_class_indices.cpu().numpy()]
            batch_conf = batch_confidences.cpu().numpy()
            
            # Append to results
            predicted_classes.extend(batch_predicted)
            confidences.extend(batch_conf)
        
        return predicted_classes, confidences
    
    def add_new_logo_class(self, class_name):
        """
        Add a new logo class to the classifier without retraining
        
        Args:
            class_name: str, the name of the new logo class
        """
        # Add the class to the list
        self.logo_classes.append(class_name)
        
        # Create prompt for the new class
        prompt = f"a logo of {class_name}"
        
        # Tokenize prompt
        text_token = clip.tokenize([prompt]).to(self.device)
        
        # Generate text embedding
        with torch.no_grad():
            if self.mixed_precision == 'bf16':
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    new_embedding = self.model.encode_text(text_token).float()
            elif self.mixed_precision == 'fp16':
                with torch.cuda.amp.autocast():
                    new_embedding = self.model.encode_text(text_token).float()
            else:
                new_embedding = self.model.encode_text(text_token).float()
        
        # Normalize embedding
        new_embedding = new_embedding / new_embedding.norm(dim=-1, keepdim=True)
        
        # Add to existing embeddings
        if self.text_embeddings is not None:
            self.text_embeddings = torch.cat([self.text_embeddings, new_embedding], dim=0)
        else:
            self.text_embeddings = new_embedding
        
        print(f"Added new logo class: {class_name}")

# =============================================================================
# PART 3: TWO-STAGE ZERO-SHOT LOGO DETECTOR WITH PARALLELIZATION
# =============================================================================

class ZeroShotLogoDetector:
    """
    Two-stage zero-shot logo detection:
    1. YOLOv11 for logo localization
    2. CLIP for zero-shot logo classification
    
    Optimized for GPU parallelization
    """
    def __init__(self, 
                 yolo_weights, 
                 clip_model="ViT-B/32", 
                 logo_classes=None, 
                 yolo_conf=0.25, 
                 clip_conf=0.5,
                 device=None,
                 batch_size=32,
                 mixed_precision=True):
        """
        Initialize the two-stage logo detector
        
        Args:
            yolo_weights: Path to YOLOv11 weights
            clip_model: CLIP model name
            logo_classes: List of logo classes
            yolo_conf: YOLOv11 confidence threshold
            clip_conf: CLIP confidence threshold
            device: PyTorch device (cuda, mps, or cpu)
            batch_size: Batch size for CLIP classification
            mixed_precision: Whether to use mixed precision
        """
        # Set device
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load trained YOLOv11 binary logo detector with GPU optimization
        self.yolo_detector = YOLO(yolo_weights)
        
        # Ensure YOLO uses the correct device
        if self.device.type == 'cuda':
            self.yolo_detector.to('cuda')
        elif self.device.type == 'mps':
            # YOLO may not support MPS directly, but we'll try
            try:
                self.yolo_detector.to('mps')
            except:
                print("YOLO doesn't support MPS directly, using CPU for YOLO")
        
        # Load CLIP-based classifier with GPU optimization
        self.clip_classifier = ClipClassifier(
            logo_classes=logo_classes, 
            model_name=clip_model,
            device=self.device,
            batch_size=batch_size,
            mixed_precision=mixed_precision
        )
        
        # Confidence thresholds
        self.yolo_conf_threshold = yolo_conf
        self.clip_conf_threshold = clip_conf
        
        # Performance settings
        self.batch_size = batch_size
        self.mixed_precision = mixed_precision
    
    def detect_and_classify(self, image_path, save_result=False, output_path=None):
        """
        Detect and classify logos in an image
        
        Args:
            image_path: str, path to the image file
            save_result: bool, whether to save visualization result
            output_path: str, path to save the result image (if save_result=True)
        
        Returns:
            results: list of dicts with keys 'box', 'class', 'confidence'
        """
        # Read the image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not read image {image_path}")
            return []
            
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        
        # Stage 1: Detect logos using YOLOv11 with GPU acceleration
        yolo_results = self.yolo_detector(image_path, conf=self.yolo_conf_threshold)[0]
        
        # Extract bounding boxes
        boxes = yolo_results.boxes.xyxy.cpu().numpy()
        confidences = yolo_results.boxes.conf.cpu().numpy()
        
        results = []
        logo_crops = []
        
        # Process each detected logo
        for i, (box, conf) in enumerate(zip(boxes, confidences)):
            x1, y1, x2, y2 = map(int, box)
            
            # Ensure box is within image bounds
            x1, x2 = max(0, x1), min(image.shape[1], x2)
            y1, y2 = max(0, y1), min(image.shape[0], y2)
            
            # Skip tiny boxes
            if x2 - x1 < 4 or y2 - y1 < 4:
                continue
            
            # Crop the logo region
            logo_crop = pil_image.crop((x1, y1, x2, y2))
            logo_crops.append(logo_crop)
        
        # If no logos detected, return empty results
        if not logo_crops:
            return results
        
        # Stage 2: Classify cropped logos using CLIP with batch processing
        predicted_classes, clip_confidences = self.clip_classifier.classify_batch(logo_crops)
        
        # Combine results
        for i, (box, yolo_conf, predicted_class, clip_conf) in enumerate(zip(
            boxes, confidences, predicted_classes, clip_confidences)):
            
            # Only include results with high CLIP confidence
            if clip_conf >= self.clip_conf_threshold:
                x1, y1, x2, y2 = map(int, box)
                results.append({
                    'box': [x1, y1, x2, y2],
                    'class': predicted_class,
                    'yolo_confidence': float(yolo_conf),
                    'clip_confidence': float(clip_conf)
                })
        
        # Visualize and save results if requested
        if save_result and results:
            output_img = image.copy()
            for result in results:
                x1, y1, x2, y2 = result['box']
                class_name = result['class']
                conf = result['clip_confidence']
                
                # Draw bounding box
                cv2.rectangle(output_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Draw class name and confidence
                label = f"{class_name}: {conf:.2f}"
                cv2.putText(output_img, label, (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Save the output image
            if output_path is None:
                output_path = image_path.replace('.', '_result.')
            cv2.imwrite(output_path, output_img)
            print(f"Result saved to {output_path}")
        
        return results
    
    def parallel_process_image(self, img_path, output_dir=None, visualize=True):
        """
        Process a single image for parallel evaluation
        
        Args:
            img_path: Path to image
            output_dir: Output directory for visualization
            visualize: Whether to save visualization
        
        Returns:
            results: Detection results
        """
        # Generate output path
        if output_dir and visualize:
            img_name = os.path.basename(img_path)
            output_path = os.path.join(output_dir, f"{os.path.splitext(img_name)[0]}_result.jpg")
        else:
            output_path = None
        
        # Detect and classify
        results = self.detect_and_classify(img_path, save_result=visualize, output_path=output_path)
        return results
    
    def evaluate(self, test_dir, num_images=10, visualize=True, output_dir=None, num_workers=4):
        """
        Evaluate the detector on a test directory with parallel processing
        
        Args:
            test_dir: str, path to the test directory
            num_images: int, number of images to evaluate (0 for all)
            visualize: bool, whether to visualize results
            output_dir: str, path to save the visualization results
            num_workers: int, number of worker processes
        """
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Find all images in the test directory
        image_files = glob.glob(os.path.join(test_dir, "*.jpg")) + \
                     glob.glob(os.path.join(test_dir, "*.png"))
        
        # Limit the number of images if specified
        if num_images > 0:
            image_files = image_files[:num_images]
        
        print(f"Evaluating on {len(image_files)} images{'...' if len(image_files) > 0 else ''}")
        
        all_results = []
        
        # For GPU models, process images sequentially to avoid CUDA issues
        if hasattr(self.device, 'type') and self.device.type in ['cuda', 'mps']:
            for img_path in tqdm(image_files, desc="Evaluating"):
                results = self.parallel_process_image(img_path, output_dir, visualize)
                all_results.append(results)
        else:
            # For CPU, use parallel processing
            print("Using CPU parallel processing for evaluation")
            with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
                process_func = partial(self.parallel_process_image, output_dir=output_dir, visualize=visualize)
                all_results = list(tqdm(executor.map(process_func, image_files), total=len(image_files), desc="Evaluating"))
        
        return all_results

    def add_new_brand(self, brand_name):
        """
        Add a new brand to the classifier without retraining
        
        Args:
            brand_name: str, the name of the new brand
        """
        self.clip_classifier.add_new_logo_class(brand_name)
        print(f"Added new brand '{brand_name}' to the zero-shot classifier")

# =============================================================================
# PART 4: TRAINING PIPELINE WITH GPU OPTIMIZATION
# =============================================================================

def train_yolov11_binary(data_yaml, yolo_model, epochs=100, batch_size=16, img_size=640, device=None):
    """
    Train YOLOv11 for binary logo detection with GPU optimization
    
    Args:
        data_yaml: Path to the YAML configuration file
        yolo_model: YOLOv11 model to use
        epochs: Number of training epochs
        batch_size: Batch size
        img_size: Image size
        device: Training device (cuda, mps, or cpu)
    
    Returns:
        Path to the trained model weights
    """
    print("Initializing YOLOv11 training...")
    
    # Set PYTORCH_CUDA_ALLOC_CONF to avoid memory fragmentation
    if device == 'cuda':
        import os
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        print("Set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid memory fragmentation")
    
    # Determine device if not provided
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    
    print(f"Training on device: {device}")
    
    # Initialize YOLOv11 model
    model = YOLO(yolo_model)
    
    # Additional optimization parameters
    num_workers = CONFIG['num_workers']
    # Use disk cache for deterministic results instead of RAM cache
    cache = 'disk'  # Changed from RAM-based condition to always use disk cache
    print("Using disk cache for deterministic training results")
    
    # Explicitly disable mixed precision since AMP checks are failing
    mixed_precision = False
    use_bf16 = False
    print("Explicitly disabling mixed precision (AMP) to prevent training instability")
    
    # Memory format processing - apply separately
    if device == 'cuda' and CONFIG.get('memory_format', '') == 'channels_last':
        print("Using channels_last memory format for improved performance")
        # We'll handle this separately since YOLO doesn't support it directly
    
    # Reduce batch size to prevent CUDA OOM
    if device == 'cuda':
        # Try to free up some CUDA memory first
        torch.cuda.empty_cache()
        
        # Check available GPU memory and adjust batch size accordingly
        total_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # Convert to GB
        free_mem = torch.cuda.memory_reserved(0) / (1024**3)  # Convert to GB
        print(f"GPU memory: {free_mem:.2f}GB free of {total_mem:.2f}GB total")
        
        # Significantly reduce batch size to prevent OOM - use a much smaller batch size
        original_batch = batch_size
        # Further reduce batch size to avoid OOM errors
        batch_size = min(batch_size, 32)  # Cap at 32
        print(f"Reducing batch size from {original_batch} to {batch_size} to prevent CUDA OOM errors")
        
        # Note: YOLO doesn't directly support gradient accumulation through 'accumulate' parameter
        print(f"Note: Using smaller batch size without gradient accumulation")
    
    print(f"Training configuration: batch_size={batch_size}, img_size={img_size}, epochs={epochs}")
    print(f"Optimization: num_workers={num_workers}, cache={cache}, mixed_precision={mixed_precision}")
    
    # Multi-GPU detection - only for info, we'll let YOLO handle this internally
    if torch.cuda.device_count() > 1:
        print(f"Multiple GPUs detected: {torch.cuda.device_count()} - YOLO will handle distribution")
    
    # Use a unique run name to avoid conflicts with previous runs
    run_name = f"yolov11_binary_{int(time.time())}"
    
    # Create a dictionary of valid YOLO training parameters
    train_args = {
        'data': data_yaml,
        'epochs': epochs,
        'imgsz': img_size,
        'batch': batch_size,
        'save': True,
        'device': device,
        'workers': num_workers,
        'cache': cache,
        'project': 'logo_detection',
        'name': run_name,  # Use timestamp-based unique name
        'patience': 20,  # Early stopping patience
        'optimizer': 'AdamW',  # Better optimizer for A100
        'lr0': 0.001,  # Initial learning rate
        'lrf': 0.01,   # Final learning rate = lr0 * lrf
        'momentum': 0.937,  # SGD momentum/Adam beta1
        'weight_decay': 0.0005,  # Optimizer weight decay
        'warmup_epochs': 3.0,  # Warmup epochs
        'warmup_momentum': 0.8,  # Warmup initial momentum
        'warmup_bias_lr': 0.1,  # Warmup initial bias lr
        'box': 7.5,  # Box loss gain
        'cls': 0.5,  # Cls loss gain
        'hsv_h': 0.015,  # Image HSV-Hue augmentation
        'hsv_s': 0.7,  # Image HSV-Saturation augmentation
        'hsv_v': 0.4,  # Image HSV-Value augmentation
        'degrees': 0.0,  # Image rotation (+/- deg)
        'translate': 0.1,  # Image translation (+/- fraction)
        'scale': 0.5,  # Image scale (+/- gain)
        'shear': 0.0,  # Image shear (+/- deg)
        'perspective': 0.0,  # Image perspective (+/- fraction)
        'flipud': 0.0,  # Image flip up-down (probability)
        'fliplr': 0.5,  # Image flip left-right (probability)
        'mosaic': 1.0,  # Image mosaic (probability)
        'mixup': 0.0,  # Image mixup (probability)
        'copy_paste': 0.0,  # Segment copy-paste (probability)
        # Hardware acceleration parameters
        'amp': False,  # Explicitly disable mixed precision to avoid AMP anomalies
        'cos_lr': True,  # Use cosine learning rate scheduler
        'close_mosaic': 10,  # Close mosaic after N epochs
    }
    
    # Train the model with validated parameters
    results = model.train(**train_args)
    
    # Get path to best weights using the actual run directory
    best_weights = os.path.join('logo_detection', run_name, 'weights', 'best.pt')
    
    # Verify that the weights file exists
    if not os.path.exists(best_weights):
        print(f"WARNING: Expected weights file not found at {best_weights}")
        # Try to find weights in results output
        if hasattr(results, 'best') and results.best:
            best_weights = results.best
            print(f"Using best weights path from training results: {best_weights}")
        else:
            # Fallback to searching for any best.pt file in logo_detection directory
            import glob
            possible_weights = glob.glob('logo_detection/*/weights/best.pt')
            if possible_weights:
                best_weights = possible_weights[-1]  # Take the most recent one
                print(f"Found alternative weights file: {best_weights}")
            else:
                print("ERROR: Could not find any weights file!")
    
    print(f"Training completed. Best weights saved to {best_weights}")
    return best_weights

# =============================================================================
# PART 5: MEMORY MANAGEMENT UTILITIES
# =============================================================================

def clean_memory():
    """
    Clean up memory by forcing garbage collection and emptying CUDA cache
    Optimized for high-memory GPUs like A100
    """
    print("Performing memory cleanup...")
    
    # Force Python garbage collection
    gc.collect()
    
    # Clean CUDA cache if available
    if torch.cuda.is_available():
        # Basic cache clearing
        torch.cuda.empty_cache()
        
        # Check if we're using an A100 or similar high-memory GPU
        is_high_mem_gpu = False
        try:
            gpu_name = torch.cuda.get_device_name(0).lower()
            is_high_mem_gpu = any(name in gpu_name for name in ['a100', 'a10g', 'a6000', 'a40', 'h100'])
            
            if is_high_mem_gpu:
                print(f"Performing A100/high-memory GPU-specific cleanup for {gpu_name}")
        except:
            pass
        
        # More aggressive memory cleanup for CUDA
        try:
            # Clear IPC caches
            torch.cuda.ipc_collect()
            
            # Reset peak memory stats
            torch.cuda.reset_peak_memory_stats()
            
            # Force cache to be reset by moving small tensor on/off device
            dummy = torch.zeros(1, device='cuda')
            dummy.cpu()
            del dummy
            
            # If high-memory GPU, try additional steps
            if is_high_mem_gpu:
                # Try to use CUDA built-in memory trimming
                try:
                    import ctypes
                    libcudart = ctypes.cdll.LoadLibrary('libcudart.so')
                    # Memory trim operation
                    libcudart.cudaDeviceSetLimit(ctypes.c_int(0x02), ctypes.c_size_t(128))
                except:
                    pass
                    
                # For PyTorch 2.0+, use memory stats to help with diagnosing issues
                if hasattr(torch.cuda, 'memory_summary'):
                    print("Memory summary after cleanup:")
                    print(torch.cuda.memory_summary(abbreviated=True))
        except Exception as e:
            print(f"Error during advanced memory cleanup: {e}")
            pass
    
    # Report current memory usage
    try:
        process = psutil.Process()
        print(f"Process memory usage: {process.memory_info().rss / (1024**3):.2f} GB")
        
        vm = psutil.virtual_memory()
        print(f"System memory: {vm.used / (1024**3):.2f} GB used of {vm.total / (1024**3):.2f} GB total")
        
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024**3)
            reserved = torch.cuda.memory_reserved() / (1024**3)
            print(f"CUDA memory: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")
    except:
        pass

# =============================================================================
# PART 6: MAIN EXECUTION
# =============================================================================

def main():
    """
    Main execution function for the complete two-stage zero-shot logo detection pipeline
    Optimized for maximum GPU utilization
    """
    # Set up GPU environment
    device = setup_gpu_environment()
    
    # Update configuration based on system capabilities
    update_config_for_system()
    
    # Create timestamp for runs
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Check if dataset exists, if not provide a warning
    if not os.path.exists(CONFIG["dataset_root"]):
        print(f"\nWARNING: Dataset not found at {CONFIG['dataset_root']}!")
        print("You need to download the LogoDet-3K dataset or set the correct path.")
        print("For testing purposes, we'll continue with a minimal configuration.")
        print("You can download the dataset from: https://github.com/Wangjing1551/LogoDet-3K\n")
        
        # Create a minimal dummy dataset folder structure
        create_dummy_dataset = input("Would you like to create a minimal test dataset? (y/n): ").lower() == 'y'
        
        if create_dummy_dataset:
            print("Creating minimal test dataset structure...")
            prepare_dummy_dataset(CONFIG["dataset_root"], CONFIG["output_root"])
    
    # 1. Prepare the dataset with parallelization
    print("="*80)
    print("STEP 1: Preparing LogoDet-3K dataset with parallel processing")
    print("="*80)
    yaml_config, all_brands = prepare_logodet3k(
        dataset_root=CONFIG["dataset_root"],
        output_root=CONFIG["output_root"],
        binary=CONFIG["binary"],
        num_workers=CONFIG["num_workers"]
    )
    
    # Clean memory after dataset preparation
    clean_memory()
    
    # Get path to YAML config
    yaml_path = os.path.join(CONFIG["output_root"], 'logodet3k.yaml')
    
    # Check if brands were found
    if len(all_brands) == 0:
        print("\nWARNING: No brands found in the dataset!")
        print("This likely means the dataset path is incorrect or the dataset structure is not as expected.")
        print("Skipping training and evaluation steps as they require a valid dataset.")
        print("\nPlease check the dataset path and structure, then run the script again.")
        
        # Return early with None
        return None
    
    # 2. Train YOLOv11 for binary logo detection with GPU optimization
    print("\n" + "="*80)
    print("STEP 2: Training YOLOv11 binary logo detector with GPU acceleration")
    print("="*80)
    best_weights = train_yolov11_binary(
        data_yaml=yaml_path,
        yolo_model=CONFIG["yolo_model"],
        epochs=CONFIG["epochs"],
        batch_size=CONFIG["batch_size"],
        img_size=CONFIG["img_size"],
        device=device.type
    )
    
    # Clean memory after training
    clean_memory()
    
    # 3. Create the two-stage detector with GPU optimization
    print("\n" + "="*80)
    print("STEP 3: Setting up GPU-optimized two-stage logo detector")
    print("="*80)
    detector = ZeroShotLogoDetector(
        yolo_weights=best_weights,
        clip_model=CONFIG["clip_model"],
        logo_classes=all_brands,
        yolo_conf=CONFIG["yolo_conf"],
        clip_conf=CONFIG["clip_conf"],
        device=device,
        batch_size=CONFIG["batch_size"],
        mixed_precision=CONFIG["mixed_precision"]
    )
    
    # 4. Evaluate on test set with GPU parallelization
    print("\n" + "="*80)
    print("STEP 4: Evaluating on test set with GPU acceleration")
    print("="*80)
    
    # Create results directory
    results_dir = os.path.join(CONFIG["results_dir"], f"run_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)
    
    # Run evaluation
    test_dir = os.path.join(CONFIG["output_root"], 'test', 'images')
    detector.evaluate(
        test_dir=test_dir,
        num_images=CONFIG["num_test_images"],
        visualize=CONFIG["visualize_results"],
        output_dir=results_dir,
        num_workers=CONFIG["num_workers"]
    )
    
    print(f"\nEvaluation complete. Results saved to {results_dir}")
    
    # 5. Demo: Add a new brand without retraining
    print("\n" + "="*80)
    print("STEP 5: Demonstration - Adding a new brand without retraining")
    print("="*80)
    new_brand = "new_example_brand"
    detector.add_new_brand(new_brand)
    
    print(f"\nAdded new brand '{new_brand}' to the zero-shot classifier")
    print("\nReady to detect this new brand without retraining!")
    
    print("\n" + "="*80)
    print("COMPLETE: GPU-optimized two-stage zero-shot logo detection pipeline is ready")
    print("="*80)
    
    return detector

def prepare_dummy_dataset(dataset_root, output_root):
    """Create a minimal dummy dataset structure for testing"""
    try:
        # Create main dataset directory
        os.makedirs(dataset_root, exist_ok=True)
        
        # Create classes.txt
        with open(os.path.join(dataset_root, 'classes.txt'), 'w') as f:
            f.write("0 Food\n1 Clothes\n2 Electronics\n")
        
        # Create category folders
        categories = ['Food', 'Clothes', 'Electronics']
        for cat in categories:
            cat_dir = os.path.join(dataset_root, cat)
            os.makedirs(cat_dir, exist_ok=True)
            
            # Create brand folders inside each category
            brands = [f"{cat}-Brand1", f"{cat}-Brand2"]
            for brand in brands:
                brand_dir = os.path.join(cat_dir, brand)
                os.makedirs(brand_dir, exist_ok=True)
                
                # Create dummy image and XML files (just 2 per brand)
                for i in range(2):
                    # Create a small black image with a white rectangle as a "logo"
                    img_path = os.path.join(brand_dir, f"{brand}_{i}.jpg")
                    img = np.zeros((300, 300, 3), dtype=np.uint8)
                    cv2.rectangle(img, (100, 100), (200, 200), (255, 255, 255), -1)
                    cv2.imwrite(img_path, img)
                    
                    # Create corresponding XML annotation with correct <name> tag
                    xml_path = os.path.join(brand_dir, f"{brand}_{i}.xml")
                    xml_content = f"""<annotation>
    <folder>{cat}</folder>
    <filename>{brand}_{i}.jpg</filename>
    <size>
        <width>300</width>
        <height>300</height>
        <depth>3</depth>
    </size>
    <object>
        <name>{brand}</name>
        <bndbox>
            <xmin>100</xmin>
            <ymin>100</ymin>
            <xmax>200</xmax>
            <ymax>200</ymax>
        </bndbox>
    </object>
</annotation>"""
                    with open(xml_path, 'w') as f:
                        f.write(xml_content)
        
        print(f"Created minimal test dataset at {dataset_root}")
        return True
    except Exception as e:
        print(f"Error creating dummy dataset: {e}")
        return False

# Execute if run as a script
if __name__ == "__main__":
    detector = main()