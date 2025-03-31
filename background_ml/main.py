import os
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import glob
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from cairosvg import svg2png
from io import BytesIO

# Configuration
INPUT_SIZE = (256, 256)  # Input image size
BATCH_SIZE = 16
EPOCHS = 15
BASE_DIR = "logo_dataset"
SVG_LOGOS_DIR = os.path.join(BASE_DIR, "svg_logos")  # Main directory containing category subfolders
JPEG_INPUT_DIR = os.path.join(BASE_DIR, "jpeg_logos")
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "processed")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

# Create necessary directories
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(os.path.join(PROCESSED_DATA_DIR, "images"), exist_ok=True)
os.makedirs(os.path.join(PROCESSED_DATA_DIR, "masks"), exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(JPEG_INPUT_DIR, exist_ok=True)

# Function to find all SVG files in a directory and its subdirectories
def find_all_svg_files(base_dir):
    """
    Recursively find all SVG files in a directory and its subdirectories
    
    Args:
        base_dir: Base directory to search
        
    Returns:
        List of tuples (full_path, category) for each SVG file
    """
    svg_files = []
    
    # Walk through all subdirectories
    for root, dirs, files in os.walk(base_dir):
        category = os.path.basename(root)
        if category == os.path.basename(base_dir):
            # If it's the base directory, don't count it as a category
            category = "uncategorized"
            
        # Find all SVG files in this directory
        for file in files:
            if file.lower().endswith('.svg'):
                full_path = os.path.join(root, file)
                svg_files.append((full_path, category))
    
    return svg_files

# 1. Simple thresholding approach (no ML)

def threshold_based_removal(input_path, output_path, threshold=240):
    """
    Remove white background using simple thresholding
    
    Args:
        input_path: Path to input JPEG logo
        output_path: Path to save output PNG
        threshold: Pixel values above this will be made transparent (0-255)
    """
    # Read image
    img = cv2.imread(input_path)
    
    # Check if image was read successfully
    if img is None:
        print(f"Error: Could not read image at {input_path}")
        return False
    
    # Convert to RGB (from BGR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Convert to grayscale for thresholding
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Create binary mask - white (255) becomes black (0), everything else becomes white (255)
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
    
    # Optional: Clean up the mask
    # Remove small noise with morphological operations
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    # Create transparent image (RGBA)
    rgba = np.zeros((img.shape[0], img.shape[1], 4), dtype=np.uint8)
    rgba[:, :, 0:3] = img_rgb
    rgba[:, :, 3] = binary
    
    # Save using PIL (better alpha channel handling)
    Image.fromarray(rgba).save(output_path)
    return True

def batch_process_logos(input_dir, output_dir, threshold=240):
    """Process all JPG logos in a directory and save them as transparent PNGs"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all jpg files
    jpg_files = glob.glob(os.path.join(input_dir, "*.jpg")) + \
                glob.glob(os.path.join(input_dir, "*.jpeg"))
    
    if not jpg_files:
        print(f"No JPG files found in {input_dir}")
        return
    
    print(f"Processing {len(jpg_files)} logos...")
    
    success_count = 0
    for jpg_file in tqdm(jpg_files):
        basename = os.path.splitext(os.path.basename(jpg_file))[0]
        output_path = os.path.join(output_dir, f"{basename}.png")
        
        if threshold_based_removal(jpg_file, output_path, threshold):
            success_count += 1
    
    print(f"Successfully processed {success_count} out of {len(jpg_files)} logos.")

# 2. Create training data from SVG logos

def svg_to_png(svg_path, output_size=(256, 256)):
    """
    Convert SVG to PNG with transparency
    
    Args:
        svg_path: Path to SVG file
        output_size: Size of output PNG
        
    Returns:
        PIL Image object of the PNG with transparency
    """
    try:
        # Read SVG file
        with open(svg_path, 'rb') as f:
            svg_data = f.read()
        
        # Convert SVG to PNG
        png_data = svg2png(
            bytestring=svg_data,
            output_width=output_size[0],
            output_height=output_size[1]
        )
        
        # Convert to PIL Image
        img = Image.open(BytesIO(png_data))
        
        # Ensure it's RGBA (has transparency)
        if img.mode != 'RGBA':
            img = img.convert('RGBA')
            
        return img
    except Exception as e:
        print(f"Error converting SVG to PNG: {e}")
        return None

def create_training_data_from_svgs():
    """
    Convert SVGs from all subfolders to PNGs with transparency and create training data
    with original SVG over white background as input image
    """
    # Find all SVG files in subfolders
    svg_files_with_categories = find_all_svg_files(SVG_LOGOS_DIR)
    
    if not svg_files_with_categories:
        print(f"No SVG files found in {SVG_LOGOS_DIR} or its subfolders")
        return False
    
    print(f"Creating training data from {len(svg_files_with_categories)} SVG logos across categories...")
    
    # Create subdirectories for categories in the processed directory if needed
    category_dirs = set()
    for _, category in svg_files_with_categories:
        category_dirs.add(category)
    
    # Process each SVG file
    for svg_file, category in tqdm(svg_files_with_categories):
        # Convert SVG to PNG with transparency
        img = svg_to_png(svg_file, INPUT_SIZE)
        if img is None:
            continue
        
        # Extract alpha channel as mask
        _, _, _, alpha = img.split()
        
        # Create white background image
        white_bg = Image.new("RGB", INPUT_SIZE, (255, 255, 255))
        
        # Paste the logo onto white background
        white_bg.paste(img, (0, 0), alpha)
        
        # Generate a unique basename to avoid collisions
        file_basename = os.path.splitext(os.path.basename(svg_file))[0]
        basename = f"{category}_{file_basename}"
        
        # Save the image with white background (simulating JPEG input)
        white_bg.save(os.path.join(PROCESSED_DATA_DIR, "images", f"{basename}.jpg"), "JPEG")
        
        # Save the alpha channel as mask
        alpha.save(os.path.join(PROCESSED_DATA_DIR, "masks", f"{basename}.png"))
    
    print(f"Created training data from {len(svg_files_with_categories)} logos across {len(category_dirs)} categories.")
    return True

# 3. Data Generator and other functions remain the same
class DataGenerator(tf.keras.utils.Sequence):
    """Simple data generator for training"""
    def __init__(self, img_paths, mask_paths, batch_size=BATCH_SIZE, img_size=INPUT_SIZE):
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.batch_size = batch_size
        self.img_size = img_size
        self.indexes = np.arange(len(self.img_paths))
        self.on_epoch_end()
        
    def __len__(self):
        return max(1, len(self.img_paths) // self.batch_size)
    
    def on_epoch_end(self):
        np.random.shuffle(self.indexes)
        
    def __getitem__(self, idx):
        batch_indexes = self.indexes[idx * self.batch_size:(idx + 1) * self.batch_size]
        
        batch_imgs = []
        batch_masks = []
        
        for i in batch_indexes:
            # Load image and mask
            img = cv2.imread(self.img_paths[i])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            mask = cv2.imread(self.mask_paths[i], cv2.IMREAD_GRAYSCALE)
            
            # Normalize
            img = img / 255.0
            mask = mask / 255.0
            
            batch_imgs.append(img)
            batch_masks.append(np.expand_dims(mask, axis=-1))
            
        return np.array(batch_imgs), np.array(batch_masks)

def prepare_dataset():
    """Prepare and split the dataset"""
    # Get all processed images and masks
    image_files = sorted(glob.glob(os.path.join(PROCESSED_DATA_DIR, "images", "*.jpg")))
    mask_files = sorted(glob.glob(os.path.join(PROCESSED_DATA_DIR, "masks", "*.png")))
    
    if len(image_files) == 0 or len(mask_files) == 0:
        print("No training data found. Run create_training_data_from_svgs() first.")
        return None, None, None
    
    # Split into train and validation sets
    train_img, val_img, train_mask, val_mask = train_test_split(
        image_files, mask_files, test_size=0.2, random_state=42)
    
    # Create data generators
    train_gen = DataGenerator(train_img, train_mask)
    val_gen = DataGenerator(val_img, val_mask)
    
    print(f"Dataset split: {len(train_img)} training, {len(val_img)} validation images")
    
    return train_gen, val_gen, val_img

# 4. Simple CNN Model

def build_simple_model(input_shape=(256, 256, 3)):
    """Build a simple CNN model for binary segmentation"""
    inputs = tf.keras.layers.Input(input_shape)
    
    # Encoder
    x = tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same')(inputs)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    
    x = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    
    x = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    
    # Decoder
    x = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(x)
    x = tf.keras.layers.UpSampling2D(size=(2, 2))(x)
    
    x = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(x)
    x = tf.keras.layers.UpSampling2D(size=(2, 2))(x)
    
    x = tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same')(x)
    x = tf.keras.layers.UpSampling2D(size=(2, 2))(x)
    
    # Output
    outputs = tf.keras.layers.Conv2D(1, 1, activation='sigmoid')(x)
    
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# 5. Training

def train_model(train_gen, val_gen):
    """Train the model with callbacks for monitoring"""
    model = build_simple_model(input_shape=(*INPUT_SIZE, 3))
    
    # Define callbacks
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        os.path.join(BASE_DIR, 'model_weights.h5'),
        monitor='val_accuracy',
        save_best_only=True,
        mode='max'
    )
    
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=5,
        mode='max'
    )
    
    # Train the model
    history = model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=val_gen,
        callbacks=[checkpoint, early_stopping]
    )
    
    # Save the final model
    model.save(os.path.join(BASE_DIR, 'logo_bg_removal_model.h5'))
    
    return model, history

# 6. Visualization

def visualize_results(model, val_imgs, num_samples=3):
    """Visualize model predictions on validation images"""
    idxs = np.random.choice(len(val_imgs), min(num_samples, len(val_imgs)), replace=False)
    
    plt.figure(figsize=(15, 5 * num_samples))
    
    for i, idx in enumerate(idxs):
        # Load and preprocess image
        img = cv2.imread(val_imgs[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_normalized = img / 255.0
        
        # Predict mask
        pred_mask = model.predict(np.expand_dims(img_normalized, axis=0))[0]
        pred_mask = (pred_mask > 0.5).astype(np.uint8) * 255
        
        # Create transparent result
        transparent = np.zeros((img.shape[0], img.shape[1], 4), dtype=np.uint8)
        transparent[:,:,:3] = img
        transparent[:,:,3] = pred_mask.squeeze()
        
        # Plot results
        plt.subplot(num_samples, 3, i*3 + 1)
        plt.imshow(img)
        plt.title(f"Original Image {i+1}")
        plt.axis('off')
        
        plt.subplot(num_samples, 3, i*3 + 2)
        plt.imshow(pred_mask.squeeze(), cmap='gray')
        plt.title(f"Predicted Mask {i+1}")
        plt.axis('off')
        
        plt.subplot(num_samples, 3, i*3 + 3)
        plt.imshow(transparent)
        plt.title(f"Result {i+1}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, 'results.png'))
    plt.show()

# 7. Process new logos using trained model

def process_with_model(model_path, input_path, output_path):
    """Process a new logo using the trained model"""
    # Load the model
    model = tf.keras.models.load_model(model_path)
    
    # Read and preprocess the image
    img = cv2.imread(input_path)
    if img is None:
        print(f"Error: Could not read image at {input_path}")
        return False
        
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, INPUT_SIZE)
    img_normalized = img_resized / 255.0
    
    # Predict mask
    pred_mask = model.predict(np.expand_dims(img_normalized, axis=0))[0]
    pred_mask = (pred_mask > 0.5).astype(np.uint8) * 255
    
    # Create transparent image
    rgba = np.zeros((INPUT_SIZE[0], INPUT_SIZE[1], 4), dtype=np.uint8)
    rgba[:,:,:3] = img_resized
    rgba[:,:,3] = pred_mask.squeeze()
    
    # Save using PIL
    Image.fromarray(rgba).save(output_path)
    
    print(f"Processed image saved to {output_path}")
    return True

def batch_process_with_model(model_path, input_dir, output_dir):
    """Process all JPG logos in a directory using the trained model"""
    # Load the model
    model = tf.keras.models.load_model(model_path)
    
    # Get all jpg files
    jpg_files = glob.glob(os.path.join(input_dir, "*.jpg")) + \
                glob.glob(os.path.join(input_dir, "*.jpeg"))
    
    if not jpg_files:
        print(f"No JPG files found in {input_dir}")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"Processing {len(jpg_files)} logos with model...")
    
    for jpg_file in tqdm(jpg_files):
        # Read and preprocess the image
        img = cv2.imread(jpg_file)
        if img is None:
            continue
            
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, INPUT_SIZE)
        img_normalized = img_resized / 255.0
        
        # Predict mask
        pred_mask = model.predict(np.expand_dims(img_normalized, axis=0))[0]
        pred_mask = (pred_mask > 0.5).astype(np.uint8) * 255
        
        # Create transparent image
        rgba = np.zeros((INPUT_SIZE[0], INPUT_SIZE[1], 4), dtype=np.uint8)
        rgba[:,:,:3] = img_resized
        rgba[:,:,3] = pred_mask.squeeze()
        
        # Save using PIL
        basename = os.path.splitext(os.path.basename(jpg_file))[0]
        output_path = os.path.join(output_dir, f"{basename}.png")
        Image.fromarray(rgba).save(output_path)
    
    print(f"Processed {len(jpg_files)} logos. Results saved to {output_dir}")

# 8. Direct SVG to PNG conversion (no ML needed for SVGs!)

def convert_svg_to_png(svg_path, output_path, size=None):
    """
    Convert SVG directly to PNG with transparency
    
    Args:
        svg_path: Path to SVG file
        output_path: Path to save output PNG
        size: Optional size for output image (width, height)
    """
    try:
        # Read SVG file
        with open(svg_path, 'rb') as f:
            svg_data = f.read()
        
        # Convert SVG to PNG
        kwargs = {}
        if size:
            kwargs['output_width'] = size[0]
            kwargs['output_height'] = size[1]
            
        png_data = svg2png(bytestring=svg_data, **kwargs)
        
        # Save PNG
        with open(output_path, 'wb') as f:
            f.write(png_data)
            
        return True
    except Exception as e:
        print(f"Error converting SVG to PNG: {e}")
        return False

def batch_convert_svgs(svg_dir, output_dir, size=None, maintain_subfolders=True):
    """
    Convert all SVGs in a directory and its subdirectories to PNGs with transparency
    
    Args:
        svg_dir: Directory containing SVG files and possibly subfolders
        output_dir: Directory to save output PNGs
        size: Optional size for output images
        maintain_subfolders: Whether to maintain subfolder structure in output
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all SVG files in subdirectories
    svg_files_with_categories = find_all_svg_files(svg_dir)
    
    if not svg_files_with_categories:
        print(f"No SVG files found in {svg_dir} or its subfolders")
        return
    
    print(f"Converting {len(svg_files_with_categories)} SVGs to PNGs...")
    
    # Create category subdirectories in output if needed
    if maintain_subfolders:
        category_dirs = set([category for _, category in svg_files_with_categories if category != "uncategorized"])
        for category in category_dirs:
            os.makedirs(os.path.join(output_dir, category), exist_ok=True)
    
    success_count = 0
    for svg_file, category in tqdm(svg_files_with_categories):
        basename = os.path.splitext(os.path.basename(svg_file))[0]
        
        # Determine output path based on whether to maintain subfolders
        if maintain_subfolders and category != "uncategorized":
            category_dir = os.path.join(output_dir, category)
            os.makedirs(category_dir, exist_ok=True)
            output_path = os.path.join(category_dir, f"{basename}.png")
        else:
            # Add category prefix to filename to avoid collisions
            if category != "uncategorized":
                basename = f"{category}_{basename}"
            output_path = os.path.join(output_dir, f"{basename}.png")
        
        if convert_svg_to_png(svg_file, output_path, size):
            success_count += 1
    
    print(f"Successfully converted {success_count} out of {len(svg_files_with_categories)} SVGs.")

# 9. Main execution function with option to use either method

def main(use_ml=False, threshold=240, convert_svgs=False, maintain_subfolders=True):
    """
    Main execution function
    
    Args:
        use_ml: Whether to use machine learning (True) or simple thresholding (False)
        threshold: Threshold value for the simple approach (0-255)
        convert_svgs: Whether to directly convert SVGs to PNGs (no ML needed)
        maintain_subfolders: Whether to maintain subfolder structure in output (for SVG conversion)
    """
    if convert_svgs:
        if not os.path.exists(SVG_LOGOS_DIR):
            print(f"Error: SVG logos directory {SVG_LOGOS_DIR} does not exist.")
            print(f"Please create this directory and place your SVG logos there.")
            return
            
        print("Converting SVGs directly to PNGs (no ML needed)...")
        batch_convert_svgs(SVG_LOGOS_DIR, OUTPUT_DIR, maintain_subfolders=maintain_subfolders)
        return
    
    if use_ml:
        if not os.path.exists(SVG_LOGOS_DIR):
            print(f"Error: SVG logos directory {SVG_LOGOS_DIR} does not exist.")
            print(f"Please create this directory and place your SVG logos there.")
            return
            
        print("Using machine learning approach...")
        
        # Create training data from SVG logos
        if not create_training_data_from_svgs():
            return
        
        # Prepare dataset
        train_gen, val_gen, val_imgs = prepare_dataset()
        if train_gen is None:
            return
        
        # Train model
        model, history = train_model(train_gen, val_gen)
        
        # Visualize results
        visualize_results(model, val_imgs)
        
        # If there are JPEGs to process, process them with the trained model
        if os.path.exists(JPEG_INPUT_DIR):
            batch_process_with_model(
                os.path.join(BASE_DIR, 'logo_bg_removal_model.h5'),
                JPEG_INPUT_DIR,
                OUTPUT_DIR
            )
    else:
        if not os.path.exists(JPEG_INPUT_DIR):
            print(f"Error: JPEG input directory {JPEG_INPUT_DIR} does not exist.")
            print(f"Please create this directory and place your JPEG logos there.")
            return
            
        print("Using simple thresholding approach...")
        batch_process_logos(JPEG_INPUT_DIR, OUTPUT_DIR, threshold)
    
    print(f"Processed logos saved to {OUTPUT_DIR}")

# Optional: Generate category statistics
def analyze_logo_dataset():
    """Analyze the logo dataset and generate statistics by category"""
    # Find all SVG files in subdirectories
    svg_files_with_categories = find_all_svg_files(SVG_LOGOS_DIR)
    
    if not svg_files_with_categories:
        print(f"No SVG files found in {SVG_LOGOS_DIR} or its subfolders")
        return
    
    # Count logos by category
    category_counts = {}
    for _, category in svg_files_with_categories:
        if category in category_counts:
            category_counts[category] += 1
        else:
            category_counts[category] = 1
    
    # Print statistics
    print(f"Found {len(svg_files_with_categories)} total logos across {len(category_counts)} categories")
    print("\nCategory breakdown:")
    for category, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {category}: {count} logos")
    
    # Optionally, visualize as a bar chart
    plt.figure(figsize=(12, 8))
    categories = list(category_counts.keys())
    counts = list(category_counts.values())
    
    # Sort by count
    sorted_indices = sorted(range(len(counts)), key=lambda i: counts[i], reverse=True)
    categories = [categories[i] for i in sorted_indices]
    counts = [counts[i] for i in sorted_indices]
    
    plt.bar(categories, counts)
    plt.xticks(rotation=45, ha='right')
    plt.xlabel('Category')
    plt.ylabel('Number of Logos')
    plt.title('Logo Count by Category')
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, 'category_stats.png'))
    plt.show()

if __name__ == "__main__":
    # Analyze the dataset first (optional)
    # analyze_logo_dataset()
    
    # For SVGs, direct conversion is the best option
    # This will convert SVGs to PNGs while maintaining subfolder structure
    # main(convert_svgs=True, maintain_subfolders=True)
    
    # For training an ML model using SVGs as training data
    main(use_ml=True)
    
    # For white background JPEGs, use the simple thresholding method
    # main(use_ml=False, threshold=240)
    
    print("Choose a method by uncommenting one of the function calls in the main block.")