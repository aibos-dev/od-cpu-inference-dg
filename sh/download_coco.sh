#!/bin/bash

# Function to check if a command was successful
check_status() {
    if [ $? -eq 0 ]; then
        echo "✓ $1 successful"
    else
        echo "✗ Error: $1 failed"
        exit 1
    fi
}

# Create dataset directory if it doesn't exist
echo "Creating COCO directory..."
COCO_DIR="coco"
mkdir -p $COCO_DIR
check_status "Directory creation"

# Change to COCO directory
cd $COCO_DIR

# Download COCO validation dataset
echo "Downloading COCO validation dataset..."
wget -c http://images.cocodataset.org/zips/val2017.zip
check_status "Validation dataset download"

echo "Downloading COCO annotations..."
wget -c http://images.cocodataset.org/annotations/annotations_trainval2017.zip
check_status "Annotations download"

# Extract files
echo "Extracting validation dataset..."
unzip -q val2017.zip
check_status "Validation dataset extraction"

echo "Extracting annotations..."
unzip -q annotations_trainval2017.zip
check_status "Annotations extraction"

# Clean up zip files (optional)
echo "Cleaning up..."
rm val2017.zip annotations_trainval2017.zip
check_status "Cleanup"

# Verify the extraction
echo -e "\nVerifying extracted files..."
if [ -d "val2017" ] && [ -d "annotations" ]; then
    echo "✓ COCO dataset successfully set up"
    echo "  - Validation images: $(ls val2017 | wc -l) files"
    echo "  - Annotation files: $(ls annotations/*.json | wc -l) files"
    echo -e "\nCOCO dataset is ready for use!"
else
    echo "✗ Error: Dataset extraction incomplete"
    exit 1
fi