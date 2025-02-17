#!/bin/bash

# Create weights directory if it doesn't exist
WEIGHTS_DIR="weights"
mkdir -p $WEIGHTS_DIR

# Change to weights directory
cd $WEIGHTS_DIR

echo "Downloading YOLO11n..."
wget https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt

echo "Downloading YOLO10n..."
wget https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov10n.pt

echo "Downloading YOLO9t..."
wget https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov9t.pt

echo "Downloading YOLOv8n..."
wget https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt


echo "All downloads completed!"
echo "Models have been downloaded to the '$WEIGHTS_DIR' directory"

# List all downloaded files with their sizes
echo -e "\nDownloaded models:"
ls -lh

# Verify all files were downloaded successfully
echo -e "\nVerifying downloads..."
for file in *.pt *.pth; do
    if [ -f "$file" ]; then
        echo "✓ $file downloaded successfully"
    else
        echo "✗ Failed to download $file"
    fi
done