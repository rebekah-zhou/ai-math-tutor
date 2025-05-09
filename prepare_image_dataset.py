import os
import shutil
import pandas as pd
import numpy as np
import argparse
import torch
from PIL import Image
from torchvision import transforms
import h5py
from tqdm import tqdm
import json
import base64
from io import BytesIO

def create_dataset_archive(image_dir, labels_file, output_dir, format_type="zip", sample_size=None):
    """
    Create a dataset archive with images and labels for easy upload to cloud platforms.
    
    Parameters:
    -----------
    image_dir : str
        Directory containing the images
    labels_file : str
        CSV file with image labels
    output_dir : str
        Directory to save the output files
    format_type : str
        'zip' to create a ZIP archive of images
        'numpy' to create NumPy arrays
        'h5' to create HDF5 dataset
        'json' to create a JSON with base64 encoded images
    sample_size : int or None
        If specified, only include this many samples (for testing)
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Read the labels CSV file
    df = pd.read_csv(labels_file)
    print(f"Read {len(df)} entries from {labels_file}")
    
    # If sample size is specified, take a subset
    if sample_size is not None and sample_size < len(df):
        df = df.sample(sample_size, random_state=42)
        print(f"Using {sample_size} random samples")
    
    # Process based on format type
    if format_type == "zip":
        create_zip_archive(df, image_dir, output_dir)
    elif format_type == "numpy":
        create_numpy_arrays(df, image_dir, output_dir)
    elif format_type == "h5":
        create_h5_dataset(df, image_dir, output_dir)
    elif format_type == "json":
        create_json_dataset(df, image_dir, output_dir)
    else:
        print(f"Unknown format type: {format_type}")

def create_zip_archive(df, image_dir, output_dir):
    """Create a ZIP archive with images and a CSV with labels."""
    # Create a temporary directory to organize the files
    temp_dir = os.path.join(output_dir, "temp_dataset")
    images_dir = os.path.join(temp_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    
    # Copy the images to the temporary directory
    print("Copying images...")
    image_paths = []
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        # This assumes your CSV has an 'image_file' column - adjust as needed
        if 'image_file' in df.columns:
            image_file = row['image_file']
        else:
            # If no explicit image file column, use index or other identifier
            # Adjust file extension (.png, .jpg, etc.) as needed
            image_file = f"{idx}.png"
        
        src_path = os.path.join(image_dir, image_file)
        if os.path.exists(src_path):
            dest_path = os.path.join(images_dir, image_file)
            shutil.copy2(src_path, dest_path)
            image_paths.append(os.path.join("images", image_file))
        else:
            print(f"Warning: Image file not found: {src_path}")
            image_paths.append("")
    
    # Create a CSV with image paths and labels
    df['image_path'] = image_paths
    csv_path = os.path.join(temp_dir, "labels.csv")
    df.to_csv(csv_path, index=False)
    
    # Create a simple README file
    with open(os.path.join(temp_dir, "README.md"), 'w') as f:
        f.write("# Math Handwriting Dataset\n\n")
        f.write(f"This dataset contains {len(df)} images of handwritten mathematical expressions.\n\n")
        f.write("## Structure\n\n")
        f.write("- `images/`: Directory containing all image files\n")
        f.write("- `labels.csv`: CSV file with image paths and labels\n\n")
        f.write("## Usage in Python\n\n")
        f.write("```python\n")
        f.write("import pandas as pd\n")
        f.write("from PIL import Image\n\n")
        f.write("# Load the labels\n")
        f.write("df = pd.read_csv('labels.csv')\n\n")
        f.write("# Load an image\n")
        f.write("image = Image.open(df.iloc[0]['image_path'])\n")
        f.write("label = df.iloc[0]['label']  # Adjust column name as needed\n")
        f.write("```\n")
    
    # Create the ZIP archive
    archive_path = os.path.join(output_dir, "math_dataset.zip")
    shutil.make_archive(os.path.join(output_dir, "math_dataset"), 'zip', temp_dir)
    
    # Clean up temporary directory
    shutil.rmtree(temp_dir)
    
    print(f"Created dataset archive: {archive_path}")
    print(f"Upload this file to your online platform (e.g., Deepnote)")

def create_numpy_arrays(df, image_dir, output_dir):
    """Create NumPy arrays with images and labels."""
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])
    
    # Initialize arrays to store images and labels
    images = []
    labels = []
    
    print("Loading and processing images...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        # This assumes your CSV has an 'image_file' and 'label' column - adjust as needed
        if 'image_file' in df.columns:
            image_file = row['image_file']
        else:
            # If no explicit image file column, use index or other identifier
            image_file = f"{idx}.png"
        
        if 'label' in df.columns:
            label = row['label']
        else:
            label = row['latex'] if 'latex' in df.columns else str(idx)
        
        src_path = os.path.join(image_dir, image_file)
        if os.path.exists(src_path):
            img = Image.open(src_path).convert('L')  # Convert to grayscale
            img_tensor = transform(img).numpy()
            images.append(img_tensor)
            labels.append(label)
        else:
            print(f"Warning: Image file not found: {src_path}")
    
    # Convert to NumPy arrays
    images_array = np.stack(images) if images else np.array([])
    labels_array = np.array(labels, dtype=object)
    
    # Save arrays
    np.save(os.path.join(output_dir, "images.npy"), images_array)
    np.save(os.path.join(output_dir, "labels.npy"), labels_array)
    
    # Create a simple README file
    with open(os.path.join(output_dir, "README_numpy.md"), 'w') as f:
        f.write("# Math Handwriting Dataset (NumPy Format)\n\n")
        f.write(f"This dataset contains {len(images)} images of handwritten mathematical expressions.\n\n")
        f.write("## Files\n\n")
        f.write("- `images.npy`: NumPy array of shape (n_samples, 1, 128, 128) containing the images\n")
        f.write("- `labels.npy`: NumPy array of shape (n_samples,) containing the labels\n\n")
        f.write("## Usage in Python\n\n")
        f.write("```python\n")
        f.write("import numpy as np\n")
        f.write("import matplotlib.pyplot as plt\n\n")
        f.write("# Load the data\n")
        f.write("images = np.load('images.npy')\n")
        f.write("labels = np.load('labels.npy', allow_pickle=True)\n\n")
        f.write("# Display an image\n")
        f.write("plt.imshow(images[0][0], cmap='gray')\n")
        f.write("plt.title(labels[0])\n")
        f.write("plt.show()\n")
        f.write("```\n")
    
    print(f"Created NumPy arrays in {output_dir}:")
    print(f"- images.npy: {images_array.shape}")
    print(f"- labels.npy: {labels_array.shape}")

def create_h5_dataset(df, image_dir, output_dir):
    """Create an HDF5 dataset with images and labels."""
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])
    
    h5_path = os.path.join(output_dir, "math_dataset.h5")
    
    with h5py.File(h5_path, 'w') as f:
        # Create datasets
        image_shape = (len(df), 1, 128, 128)
        images_dataset = f.create_dataset('images', shape=image_shape, dtype='f')
        # Variable-length string dataset for labels
        dt = h5py.special_dtype(vlen=str)
        labels_dataset = f.create_dataset('labels', shape=(len(df),), dtype=dt)
        
        print("Loading and processing images...")
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            # This assumes your CSV has an 'image_file' and 'label' column - adjust as needed
            if 'image_file' in df.columns:
                image_file = row['image_file']
            else:
                # If no explicit image file column, use index or other identifier
                image_file = f"{idx}.png"
            
            if 'label' in df.columns:
                label = row['label']
            else:
                label = row['latex'] if 'latex' in df.columns else str(idx)
            
            src_path = os.path.join(image_dir, image_file)
            if os.path.exists(src_path):
                img = Image.open(src_path).convert('L')  # Convert to grayscale
                img_tensor = transform(img).numpy()
                images_dataset[idx] = img_tensor
                labels_dataset[idx] = label
            else:
                print(f"Warning: Image file not found: {src_path}")
                images_dataset[idx] = np.zeros((1, 128, 128))
                labels_dataset[idx] = "MISSING"
    
    # Create a simple README file
    with open(os.path.join(output_dir, "README_h5.md"), 'w') as f:
        f.write("# Math Handwriting Dataset (HDF5 Format)\n\n")
        f.write(f"This dataset contains {len(df)} images of handwritten mathematical expressions.\n\n")
        f.write("## Files\n\n")
        f.write("- `math_dataset.h5`: HDF5 file containing the images and labels\n\n")
        f.write("## Usage in Python\n\n")
        f.write("```python\n")
        f.write("import h5py\n")
        f.write("import matplotlib.pyplot as plt\n\n")
        f.write("# Load the data\n")
        f.write("with h5py.File('math_dataset.h5', 'r') as f:\n")
        f.write("    # Get the first image and label\n")
        f.write("    image = f['images'][0]\n")
        f.write("    label = f['labels'][0]\n\n")
        f.write("    # Display the image\n")
        f.write("    plt.imshow(image[0], cmap='gray')\n")
        f.write("    plt.title(label)\n")
        f.write("    plt.show()\n")
        f.write("```\n")
    
    print(f"Created HDF5 dataset: {h5_path}")

def create_json_dataset(df, image_dir, output_dir):
    """Create a JSON file with base64-encoded images and labels."""
    dataset = []
    
    print("Loading and encoding images...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        # This assumes your CSV has an 'image_file' and 'label' column - adjust as needed
        if 'image_file' in df.columns:
            image_file = row['image_file']
        else:
            # If no explicit image file column, use index or other identifier
            image_file = f"{idx}.png"
        
        if 'label' in df.columns:
            label = row['label']
        else:
            label = row['latex'] if 'latex' in df.columns else str(idx)
        
        src_path = os.path.join(image_dir, image_file)
        if os.path.exists(src_path):
            # Encode image to base64
            with open(src_path, 'rb') as img_file:
                img_data = img_file.read()
                b64_encoded = base64.b64encode(img_data).decode('utf-8')
            
            # Add to dataset
            dataset.append({
                'id': idx,
                'label': label,
                'image_b64': b64_encoded,
                'filename': image_file
            })
        else:
            print(f"Warning: Image file not found: {src_path}")
    
    # Save to JSON
    json_path = os.path.join(output_dir, "math_dataset.json")
    with open(json_path, 'w') as f:
        json.dump(dataset, f)
    
    # Create a simple README file
    with open(os.path.join(output_dir, "README_json.md"), 'w') as f:
        f.write("# Math Handwriting Dataset (JSON Format)\n\n")
        f.write(f"This dataset contains {len(dataset)} images of handwritten mathematical expressions.\n\n")
        f.write("## Files\n\n")
        f.write("- `math_dataset.json`: JSON file containing the images (base64-encoded) and labels\n\n")
        f.write("## Usage in Python\n\n")
        f.write("```python\n")
        f.write("import json\n")
        f.write("import base64\n")
        f.write("from io import BytesIO\n")
        f.write("from PIL import Image\n")
        f.write("import matplotlib.pyplot as plt\n\n")
        f.write("# Load the data\n")
        f.write("with open('math_dataset.json', 'r') as f:\n")
        f.write("    dataset = json.load(f)\n\n")
        f.write("# Get the first image and label\n")
        f.write("sample = dataset[0]\n")
        f.write("label = sample['label']\n\n")
        f.write("# Decode base64 image\n")
        f.write("img_data = base64.b64decode(sample['image_b64'])\n")
        f.write("img = Image.open(BytesIO(img_data))\n\n")
        f.write("# Display the image\n")
        f.write("plt.imshow(img, cmap='gray' if img.mode == 'L' else None)\n")
        f.write("plt.title(label)\n")
        f.write("plt.show()\n")
        f.write("```\n")
    
    print(f"Created JSON dataset: {json_path}")

def main():
    parser = argparse.ArgumentParser(description="Prepare image dataset for online platforms")
    parser.add_argument("--image_dir", default="data/images", help="Directory containing the images")
    parser.add_argument("--labels_file", default="data/images/labels.csv", help="CSV file with image labels")
    parser.add_argument("--output_dir", default="output", help="Directory to save the output files")
    parser.add_argument("--format", default="zip", choices=["zip", "numpy", "h5", "json"], 
                        help="Output format type")
    parser.add_argument("--sample", type=int, default=None, 
                        help="Number of samples to include (for testing)")
    args = parser.parse_args()
    
    create_dataset_archive(args.image_dir, args.labels_file, args.output_dir, args.format, args.sample)

if __name__ == "__main__":
    main()