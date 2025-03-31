#!/usr/bin/env python3
"""
Script for running inference with YOLOv11 models for logo detection
"""

import os
import argparse
from ultralytics import YOLO

def parse_args():
    parser = argparse.ArgumentParser(description='Run inference with YOLOv11 for logo detection')
    parser.add_argument('--weights', type=str, required=True, help='Model weights path')
    parser.add_argument('--source', type=str, required=True, help='Source to run inference on (file/dir/URL/glob)')
    parser.add_argument('--img-size', type=int, default=640, help='Inference image size')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='Maximum number of detections per image')
    parser.add_argument('--device', default='', help='CUDA device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='Show results')
    parser.add_argument('--save-txt', action='store_true', help='Save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='Save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='Save cropped prediction boxes')
    parser.add_argument('--name', default='exp', help='Save results to project/name')
    parser.add_argument('--line-width', default=3, type=int, help='Bounding box thickness (pixels)')
    parser.add_argument('--show-labels', default=True, action='store_false', help='Show labels')
    parser.add_argument('--show-conf', default=True, action='store_false', help='Show confidences')
    parser.add_argument('--vid-stride', type=int, default=1, help='Video frame-rate stride')
    parser.add_argument('--project', default='runs/detect', help='Save results to project/name')
    return parser.parse_args()

def main():
    args = parse_args()
    
    print(f"Loading YOLOv11 model from {args.weights}")
    model = YOLO(args.weights)
    
    # Configure inference parameters
    kwargs = {
        "source": args.source,
        "conf": args.conf_thres,
        "iou": args.iou_thres,
        "imgsz": args.img_size,
        "device": args.device,
        "show": args.view_img,
        "save": True,  # Ensure images are saved
        "save_txt": args.save_txt,
        "save_conf": args.save_conf,
        "save_crop": args.save_crop,
        "name": args.name,
        "line_width": args.line_width,
        "show_labels": args.show_labels,
        "show_conf": args.show_conf,
        "vid_stride": args.vid_stride,
        "max_det": args.max_det,
        "project": args.project
    }
    
    # Run inference
    print(f"Running inference on {args.source}")
    print(f"Using project path: {args.project}")
    print(f"Using name: {args.name}")
    full_save_path = os.path.join(os.getcwd(), args.project, args.name) if not os.path.isabs(args.project) else os.path.join(args.project, args.name)
    print(f"Will save to: {full_save_path}")
    
    # Ensure the save directory exists
    if not os.path.exists(full_save_path):
        print(f"Creating directory: {full_save_path}")
        os.makedirs(full_save_path, exist_ok=True)
    
    results = model.predict(**kwargs)
    
    # Check if results were saved
    print(f"Checking if result directory exists: {full_save_path}")
    print(f"Directory exists: {os.path.exists(full_save_path)}")
    if os.path.exists(full_save_path):
        print(f"Contents: {os.listdir(full_save_path)}")
    
    # Print results summary
    for r in results:
        boxes = r.boxes  # Boxes object for bounding box outputs
        if hasattr(r, 'probs'):
            probs = r.probs  # Class probabilities for classification outputs
        
        if len(boxes) > 0:
            print(f"Detected {len(boxes)} logos")
        else:
            print("No logos detected")
    
    print(f"Results saved to {os.path.join(args.project, args.name)}")

if __name__ == "__main__":
    main() 