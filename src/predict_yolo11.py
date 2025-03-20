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
    parser.add_argument('--line-thickness', default=3, type=int, help='Bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='Hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='Hide confidences')
    parser.add_argument('--vid-stride', type=int, default=1, help='Video frame-rate stride')
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
        "save_txt": args.save_txt,
        "save_conf": args.save_conf,
        "save_crop": args.save_crop,
        "name": args.name,
        "line_thickness": args.line_thickness,
        "hide_labels": args.hide_labels,
        "hide_conf": args.hide_conf,
        "vid_stride": args.vid_stride,
        "max_det": args.max_det
    }
    
    # Run inference
    print(f"Running inference on {args.source}")
    results = model.predict(**kwargs)
    
    # Print results summary
    for r in results:
        boxes = r.boxes  # Boxes object for bounding box outputs
        if hasattr(r, 'probs'):
            probs = r.probs  # Class probabilities for classification outputs
        
        if len(boxes) > 0:
            print(f"Detected {len(boxes)} logos")
        else:
            print("No logos detected")
    
    print(f"Results saved to {os.path.join('runs/detect', args.name)}")

if __name__ == "__main__":
    main() 