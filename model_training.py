import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50, VGG16, MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

def create_transfer_learning_model(base_model_name, input_shape=(224, 224, 3), num_classes=6, dropout_rate=0.5, l2_reg=0.001):
    """
    Create a transfer learning model using a pre-trained base model
    
    Args:
        base_model_name: Name of the base model to use ('resnet50', 'vgg16', or 'mobilenetv2')
        input_shape: Input shape for the model
        num_classes: Number of output classes
        dropout_rate: Dropout rate for regularization
        l2_reg: L2 regularization factor
        
    Returns:
        Keras model
    """
    # Choose base model
    if base_model_name.lower() == 'resnet50':
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    elif base_model_name.lower() == 'vgg16':
        base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    elif base_model_name.lower() == 'mobilenetv2':
        base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
    else:
        raise ValueError(f"Unsupported base model: {base_model_name}")
    
    # Freeze base model layers
    for layer in base_model.layers:
        layer.trainable = False
    
    # Add custom top layers
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dense(512, activation='relu', kernel_regularizer=l2(l2_reg))(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(256, activation='relu', kernel_regularizer=l2(l2_reg))(x)
    x = Dropout(dropout_rate)(x)
    predictions = Dense(num_classes, activation='softmax', kernel_regularizer=l2(l2_reg))(x)
    
    # Create model
    model = Model(inputs=base_model.input, outputs=predictions)
    
    return model

def fine_tune_model(model, num_layers_to_unfreeze=10):
    """
    Fine-tune a transfer learning model by unfreezing some of the top layers
    
    Args:
        model: Keras model with frozen base layers
        num_layers_to_unfreeze: Number of top layers to unfreeze for fine-tuning
        
    Returns:
        Model with unfrozen layers
    """
    # Unfreeze the top layers
    for layer in model.layers[-num_layers_to_unfreeze:]:
        if hasattr(layer, 'trainable'):
            layer.trainable = True
    
    return model

def train_model(model, train_generator, val_generator, epochs=50, fine_tuning_epochs=25, 
                initial_lr=0.001, fine_tuning_lr=0.0001, model_save_path='models'):
    """
    Train the model with a two-stage approach: transfer learning followed by fine-tuning
    
    Args:
        model: Keras model
        train_generator: Training data generator
        val_generator: Validation data generator
        epochs: Number of epochs for initial training
        fine_tuning_epochs: Number of epochs for fine-tuning
        initial_lr: Initial learning rate
        fine_tuning_lr: Learning rate for fine-tuning
        model_save_path: Directory to save models
        
    Returns:
        Training history
    """
    # Create model directory if it doesn't exist
    os.makedirs(model_save_path, exist_ok=True)
    
    # Callbacks
    checkpoint = ModelCheckpoint(
        os.path.join(model_save_path, 'best_model.h5'),
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )
    
    # Stage 1: Train the top layers with frozen base model
    model.compile(
        optimizer=Adam(learning_rate=initial_lr),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("Stage 1: Training the top layers with frozen base model")
    history1 = model.fit(
        train_generator,
        steps_per_epoch=len(train_generator),
        epochs=epochs,
        validation_data=val_generator,
        validation_steps=len(val_generator),
        callbacks=[checkpoint, early_stopping, reduce_lr]
    )
    
    # Stage 2: Fine-tuning
    model = fine_tune_model(model)
    
    model.compile(
        optimizer=Adam(learning_rate=fine_tuning_lr),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("Stage 2: Fine-tuning the model")
    history2 = model.fit(
        train_generator,
        steps_per_epoch=len(train_generator),
        epochs=fine_tuning_epochs,
        validation_data=val_generator,
        validation_steps=len(val_generator),
        callbacks=[checkpoint, early_stopping, reduce_lr]
    )
    
    # Save final model
    model.save(os.path.join(model_save_path, 'surface_defect_model.h5'))
    
    # Combine histories
    combined_history = {
        'accuracy': history1.history['accuracy'] + history2.history['accuracy'],
        'val_accuracy': history1.history['val_accuracy'] + history2.history['val_accuracy'],
        'loss': history1.history['loss'] + history2.history['loss'],
        'val_loss': history1.history['val_loss'] + history2.history['val_loss']
    }
    
    return combined_history

def evaluate_model(model, test_generator, class_names):
    """
    Evaluate the model and generate classification report and confusion matrix
    
    Args:
        model: Trained Keras model
        test_generator: Test data generator
        class_names: List of class names
        
    Returns:
        Test accuracy, classification report, and confusion matrix
    """
    # Reset the generator
    test_generator.reset()
    
    # Predict
    predictions = model.predict(test_generator, steps=len(test_generator), verbose=1)
    y_pred = np.argmax(predictions, axis=1)
    
    # True labels
    y_true = test_generator.classes[:len(y_pred)]
    
    # Classification report
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()
    
    # Calculate accuracy
    accuracy = sum(y_pred == y_true) / len(y_true)
    
    return accuracy, report, cm

def plot_training_history(history):
    """
    Plot training and validation accuracy and loss
    
    Args:
        history: Training history
    """
    # Accuracy plot
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['accuracy'])
    plt.plot(history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='lower right')
    
    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()

if __name__ == "__main__":
    print("Model Training Script for Metal Surface Defect Detection")
    print("="*60)
    
    # Example usage (uncomment to run)
    # from data_processing import create_data_generators
    # 
    # # Paths to organized dataset directories
    # train_dir = "organized_data/train"
    # val_dir = "organized_data/val"
    # test_dir = "organized_data/test"
    # 
    # # Create data generators
    # train_gen, val_gen, test_gen, class_names = create_data_generators(train_dir, val_dir, test_dir)
    # 
    # # Create model
    # model = create_transfer_learning_model('resnet50', num_classes=len(class_names))
    # 
    # # Train model
    # history = train_model(model, train_gen, val_gen)
    # 
    # # Plot training history
    # plot_training_history(history)
    # 
    # # Load best model for evaluation
    # best_model = tf.keras.models.load_model('models/best_model.h5')
    # 
    # # Evaluate model
    # accuracy, report, cm = evaluate_model(best_model, test_gen, class_names)
    # 
    # print(f"Test Accuracy: {accuracy:.4f}")
    # print("\nClassification Report:")
    # for class_name in class_names:
    #     print(f"{class_name}: F1-Score = {report[class_name]['f1-score']:.4f}")
    # 
    # print(f"\nOverall F1-Score: {report['weighted avg']['f1-score']:.4f}")
    # print(f"Saved confusion matrix to confusion_matrix.png") 