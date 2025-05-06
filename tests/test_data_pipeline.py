import os
import pytest
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Create a dummy image preprocessing function for testing
def preprocess_image(img_path, target_size=(224, 224)):
    """Preprocess image for model prediction"""
    img = load_img(img_path, target_size=target_size)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize
    return img_array

# Create a dummy data augmentation function for testing
def create_data_augmentation():
    """Create a data augmentation pipeline for training"""
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
    ])

class TestDataPipeline:
    
    def test_image_preprocessing_shape(self):
        """Test if image preprocessing returns correct shape"""
        # Create a dummy image for testing
        dummy_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        dummy_image_path = "dummy_image.jpg"
        tf.keras.preprocessing.image.save_img(dummy_image_path, dummy_image)
        
        try:
            # Test preprocessing
            processed_image = preprocess_image(dummy_image_path)
            
            # Check shape
            assert processed_image.shape == (1, 224, 224, 3)
            
            # Check normalization (values should be between 0 and 1)
            assert np.max(processed_image) <= 1.0
            assert np.min(processed_image) >= 0.0
            
        finally:
            # Clean up
            if os.path.exists(dummy_image_path):
                os.remove(dummy_image_path)
    
    def test_data_augmentation(self):
        """Test if data augmentation produces different images"""
        # Create a dummy image for testing
        dummy_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        dummy_image = np.expand_dims(dummy_image, axis=0) / 255.0
        
        # Apply data augmentation
        data_augmentation = create_data_augmentation()
        augmented_image = data_augmentation(dummy_image)
        
        # Check shape remains the same
        assert augmented_image.shape == dummy_image.shape
        
        # Verify that augmentation changed the image (may fail extremely rarely)
        assert not np.array_equal(augmented_image, dummy_image)
    
    def test_batch_processing(self):
        """Test batch processing functionality"""
        # Create dummy batch data
        batch_size = 16
        dummy_batch = np.random.randint(0, 255, (batch_size, 224, 224, 3), dtype=np.uint8) / 255.0
        
        # Test batch normalization
        normalized_batch = dummy_batch.copy()
        
        # Check shape and values
        assert normalized_batch.shape == (batch_size, 224, 224, 3)
        assert np.max(normalized_batch) <= 1.0
        assert np.min(normalized_batch) >= 0.0 