import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import shutil

def create_dataset_dataframe(data_dir):
    """
    Create a DataFrame with image paths and labels
    
    Args:
        data_dir: Directory containing class subdirectories with images
        
    Returns:
        DataFrame with columns 'image_path' and 'label'
    """
    image_paths = []
    labels = []
    
    # Iterate through each class directory
    for class_name in os.listdir(data_dir):
        class_dir = os.path.join(data_dir, class_name)
        
        # Skip if not a directory
        if not os.path.isdir(class_dir):
            continue
        
        # Process all images in the class directory
        for img_name in os.listdir(class_dir):
            # Check if it's an image file
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(class_dir, img_name)
                image_paths.append(img_path)
                labels.append(class_name)
    
    # Create DataFrame
    df = pd.DataFrame({
        'image_path': image_paths,
        'label': labels
    })
    
    # Create numerical labels
    unique_labels = df['label'].unique()
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    df['label_idx'] = df['label'].map(label_to_idx)
    
    return df, label_to_idx

def split_dataset(df, test_size=0.2, val_size=0.2, random_state=42):
    """
    Split dataset into train, validation, and test sets
    
    Args:
        df: DataFrame with image paths and labels
        test_size: Proportion of data for testing
        val_size: Proportion of training data for validation
        random_state: Random seed for reproducibility
        
    Returns:
        train_df, val_df, test_df
    """
    # First split into train+val and test
    train_val_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=df['label']
    )
    
    # Then split train+val into train and val
    train_df, val_df = train_test_split(
        train_val_df, test_size=val_size, random_state=random_state, stratify=train_val_df['label']
    )
    
    return train_df, val_df, test_df

def organize_data_for_keras(df, target_dir, subset_name):
    """
    Organize images for use with Keras ImageDataGenerator
    
    Args:
        df: DataFrame with image paths and labels
        target_dir: Base directory for organized dataset
        subset_name: Name of the subset (train, val, test)
    """
    subset_dir = os.path.join(target_dir, subset_name)
    os.makedirs(subset_dir, exist_ok=True)
    
    for label in df['label'].unique():
        label_dir = os.path.join(subset_dir, label)
        os.makedirs(label_dir, exist_ok=True)
    
    for _, row in df.iterrows():
        src_path = row['image_path']
        label = row['label']
        filename = os.path.basename(src_path)
        dst_path = os.path.join(subset_dir, label, filename)
        shutil.copy(src_path, dst_path)
    
    return subset_dir

def create_data_generators(train_dir, val_dir, test_dir, batch_size=32, img_size=(224, 224)):
    """
    Create data generators for training, validation, and testing
    
    Args:
        train_dir: Directory containing training images organized by class
        val_dir: Directory containing validation images organized by class
        test_dir: Directory containing test images organized by class
        batch_size: Batch size for training
        img_size: Target image size
        
    Returns:
        train_generator, val_generator, test_generator, class_names
    """
    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    # Only rescaling for validation and test
    val_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    # Create generators
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='sparse'
    )
    
    val_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='sparse'
    )
    
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='sparse'
    )
    
    # Get class names
    class_names = list(train_generator.class_indices.keys())
    
    return train_generator, val_generator, test_generator, class_names

def visualize_samples(generator, class_names, num_samples=5):
    """
    Visualize sample images from the data generator
    
    Args:
        generator: Keras data generator
        class_names: List of class names
        num_samples: Number of samples to visualize
    """
    plt.figure(figsize=(15, num_samples * 3))
    
    # Get a batch of images
    x_batch, y_batch = next(generator)
    
    for i in range(min(num_samples, len(x_batch))):
        plt.subplot(num_samples, 1, i + 1)
        plt.imshow(x_batch[i])
        plt.title(f"Class: {class_names[int(y_batch[i])]}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('sample_images.png')
    plt.close()

def analyze_dataset(df):
    """
    Analyze the dataset distribution and image properties
    
    Args:
        df: DataFrame with image paths and labels
        
    Returns:
        DataFrame with analysis results
    """
    # Count samples per class
    class_counts = df['label'].value_counts().reset_index()
    class_counts.columns = ['Class', 'Count']
    
    # Image size analysis
    sizes = []
    for img_path in df['image_path'].sample(min(100, len(df))):
        try:
            img = plt.imread(img_path)
            sizes.append([img_path, img.shape[0], img.shape[1], img.shape[2]])
        except Exception as e:
            print(f"Error analyzing {img_path}: {e}")
    
    size_df = pd.DataFrame(sizes, columns=['image_path', 'height', 'width', 'channels'])
    
    # Basic statistics
    stats = {
        'total_images': len(df),
        'num_classes': len(df['label'].unique()),
        'min_height': size_df['height'].min(),
        'max_height': size_df['height'].max(),
        'min_width': size_df['width'].min(),
        'max_width': size_df['width'].max(),
        'channel_mode': size_df['channels'].mode()[0]
    }
    
    return class_counts, size_df, stats

if __name__ == "__main__":
    print("Data Processing Script for Metal Surface Defect Detection")
    print("="*60)
    
    # Example usage (uncomment to run)
    # data_dir = "NEU"
    # organized_data_dir = "organized_data"
    # 
    # # Create dataset DataFrame
    # df, label_to_idx = create_dataset_dataframe(data_dir)
    # print("Dataset created with", len(df), "images across", len(label_to_idx), "classes")
    # 
    # # Analyze dataset
    # class_counts, size_df, stats = analyze_dataset(df)
    # print("Dataset Statistics:")
    # for key, value in stats.items():
    #     print(f"- {key}: {value}")
    # 
    # # Split dataset
    # train_df, val_df, test_df = split_dataset(df)
    # print(f"Dataset split: {len(train_df)} train, {len(val_df)} validation, {len(test_df)} test")
    # 
    # # Organize data for Keras
    # train_dir = organize_data_for_keras(train_df, organized_data_dir, 'train')
    # val_dir = organize_data_for_keras(val_df, organized_data_dir, 'val')
    # test_dir = organize_data_for_keras(test_df, organized_data_dir, 'test')
    # 
    # # Create data generators
    # train_gen, val_gen, test_gen, class_names = create_data_generators(train_dir, val_dir, test_dir)
    # 
    # # Visualize samples
    # visualize_samples(train_gen, class_names) 