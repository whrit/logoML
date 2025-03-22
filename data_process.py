import os
import glob
import xml.etree.ElementTree as ET
import numpy as np
from tqdm import tqdm
import shutil
import yaml
import random

def prepare_logodet3k(dataset_root, output_root, binary=True):
    """
    Prepare LogoDet-3K dataset with category/brand hierarchy for YOLO training
    
    Args:
        dataset_root: Path to the original LogoDet-3K folder
        output_root: Path where to save the prepared dataset
        binary: If True, prepare for binary classification (logo/non-logo),
                otherwise use multi-class (each logo as separate class)
    """
    # Create output directories
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(output_root, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_root, split, 'labels'), exist_ok=True)
    
    # Load category mapping
    category_map = {}
    if os.path.exists(os.path.join(dataset_root, 'classes.txt')):
        with open(os.path.join(dataset_root, 'classes.txt'), 'r') as f:
            for line in f:
                parts = line.strip().split(' ', 1)
                if len(parts) == 2:
                    category_map[parts[1]] = int(parts[0])
    
    # Find all category folders
    category_folders = [f for f in os.listdir(dataset_root) 
                       if os.path.isdir(os.path.join(dataset_root, f)) and 
                       f in category_map]
    
    # Create brand-to-class mapping for multi-class version
    brand_to_class = {}
    class_id = 0
    
    # Create a list to store all brands for CLIP classification
    all_brands = []
    
    # Find all brand folders and create class mapping
    for category in category_folders:
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
    
    # Process all images and annotations
    all_image_paths = []
    
    # Find all image files and their corresponding XML files
    for category in tqdm(category_folders, desc="Scanning categories"):
        category_path = os.path.join(dataset_root, category)
        brand_folders = [f for f in os.listdir(category_path) 
                         if os.path.isdir(os.path.join(category_path, f))]
        
        for brand in brand_folders:
            brand_path = os.path.join(category_path, brand)
            
            # Extract normalized brand name
            base_brand = brand.split('-')[0] if '-' in brand else brand
            base_brand = base_brand.lower()
            
            # Find all JPG files in the brand folder
            image_files = glob.glob(os.path.join(brand_path, "*.jpg"))
            
            for img_path in image_files:
                xml_path = img_path.replace('.jpg', '.xml')
                if os.path.exists(xml_path):
                    # Store the image path, XML path, category, and brand
                    all_image_paths.append((img_path, xml_path, category, base_brand))
    
    # Shuffle all image paths
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
    import json
    with open(os.path.join(output_root, 'brands.json'), 'w') as f:
        json.dump(all_brands, f, indent=2)
    
    # Process each split
    for split, files in [('train', train_files), ('val', val_files), ('test', test_files)]:
        for img_path, xml_path, category, brand in tqdm(files, desc=f"Processing {split} set"):
            # Generate output file names
            img_name = os.path.basename(img_path)
            label_name = img_name.replace('.jpg', '.txt')
            
            # Output paths
            out_img_path = os.path.join(output_root, split, 'images', img_name)
            out_label_path = os.path.join(output_root, split, 'labels', label_name)
            
            # Copy image
            shutil.copy(img_path, out_img_path)
            
            # Convert XML to YOLO format
            convert_xml_to_yolo(xml_path, out_label_path, brand_to_class, brand, binary)
    
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

def convert_xml_to_yolo(xml_path, output_path, brand_to_class, brand, binary=True):
    """
    Convert XML annotation file to YOLO format
    
    Args:
        xml_path: Path to the XML annotation file
        output_path: Path to save the YOLO format annotation
        brand_to_class: Dictionary mapping brand names to class IDs
        brand: Brand name for this XML file
        binary: If True, use class 0 for all logos
    """
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
            
    except Exception as e:
        print(f"Error processing {xml_path}: {e}")
        # Create an empty file in case of errors
        open(output_path, 'w').close()


if __name__ == "__main__":
    # For binary classification (logo/non-logo)
    prepare_logodet3k(
        dataset_root="data/LogoDet-3k", 
        output_root="data/logodet3k_binary",
        binary=True
    )