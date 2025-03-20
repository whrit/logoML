#!/usr/bin/env python3
"""
Script for training YOLOv11 models for logo detection
"""

import os
import argparse
import yaml
from ultralytics import YOLO

def parse_args():
    parser = argparse.ArgumentParser(description='Train YOLOv11 for logo detection')
    parser.add_argument('--dataset', type=str, default='flickr27', choices=['flickr27', 'logodet3k'],
                        help='Dataset to use: flickr27 or logodet3k')
    parser.add_argument('--img-size', type=int, default=640, help='Training image size')
    parser.add_argument('--batch', type=int, default=16, help='Batch size')
    parser.add_argument('--epochs', type=int, default=300, help='Number of epochs')
    parser.add_argument('--model', type=str, default='yolo11n.pt', 
                        choices=['yolo11n.pt', 'yolo11s.pt', 'yolo11m.pt', 'yolo11l.pt', 'yolo11x.pt'],
                        help='Model size')
    parser.add_argument('--workers', type=int, default=8, help='Number of workers for dataloader')
    parser.add_argument('--device', default='', help='CUDA device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--name', type=str, default='yolo11_logo_det', help='Name of the experiment')
    parser.add_argument('--resume', action='store_true', help='Resume training from last checkpoint')
    parser.add_argument('--config', type=str, default='src/cfg/training/yolo11.yaml', 
                        help='Path to YAML config file')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Set the data configuration based on dataset choice
    data_yaml = 'data/logo_data_flickr.yaml' if args.dataset == 'flickr27' else 'data/logo_data.yaml'
    
    # Load the config file
    if os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    else:
        print(f"Config file {args.config} not found. Using default parameters.")
        config = {}
    
    # Update config with command line arguments
    config['data'] = data_yaml
    config['imgsz'] = args.img_size
    config['batch'] = args.batch
    config['epochs'] = args.epochs
    config['model'] = args.model
    config['name'] = args.name
    config['device'] = args.device
    config['workers'] = args.workers
    
    # Initialize the model
    model = YOLO(args.model)
    
    # Start training
    print(f"Starting YOLOv11 training with {args.dataset} dataset")
    
    # Train the model using the config
    results = model.train(**config)
    
    # Evaluate the model
    print("Evaluating model...")
    model.val()
    
    print(f"Training complete. Model saved to {os.path.join('runs/detect', args.name)}")

if __name__ == "__main__":
    main() 