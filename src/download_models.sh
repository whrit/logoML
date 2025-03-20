#!/bin/bash
# Script to download YOLOv11 pretrained models

# Create directory for models if it doesn't exist
mkdir -p models

# Download YOLOv11n model (nano)
echo "Downloading YOLOv11n model..."
python -c "from ultralytics import YOLO; YOLO('yolo11n.pt')" || echo "Failed to download YOLOv11n"

# Download YOLOv11s model (small)
echo "Downloading YOLOv11s model..."
python -c "from ultralytics import YOLO; YOLO('yolo11s.pt')" || echo "Failed to download YOLOv11s"

# Download YOLOv11m model (medium)
echo "Downloading YOLOv11m model..."
python -c "from ultralytics import YOLO; YOLO('yolo11m.pt')" || echo "Failed to download YOLOv11m"

# Download YOLOv11l model (large)
echo "Downloading YOLOv11l model..."
python -c "from ultralytics import YOLO; YOLO('yolo11l.pt')" || echo "Failed to download YOLOv11l"

echo "Download complete. Models are stored in ~/.cache/ultralytics/"
echo "You can also find model information in the Ultralytics GitHub repository: https://github.com/ultralytics/ultralytics" 