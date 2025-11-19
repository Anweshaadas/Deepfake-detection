import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix

# --- CONFIGURATION (Adjust these if your folder structure changes) ---
TARGET_SIZE = (224, 224) 
BATCH_SIZE = 32
EPOCHS = 10 
LEARNING_RATE = 1e-4 
PROCESSED_DATA_DIR = 'dataset/processed' # Directory containing FAKE/REAL subfolders
MODEL_SAVE_PATH = 'deepfake_detector_simplified.h5'

# --- MODEL BUILDING ---
def build_simple_model(input_shape=TARGET_SIZE + (3,)):
    # Load EfficientNetB0 pre-trained on ImageNet
    base_model = EfficientNetB0(
        weights='imagenet', 
        include_top=False, 
        input_shape=input_shape
    )
    # Freeze the base layers for transfer learning
    base_model.trainable = False 

    # Build the classification head using the Sequential API
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(), 
        Dropout(0.5), # Regularization layer
        Dense(1, activation='sigmoid') # Final binary classification layer
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE), 
        loss='binary_crossentropy', 
        metrics=['accuracy']
    )
    return model

# --- TRAINING AND EVALUATION ---
if __name__ == '__main__':
    # 1. Setup Data Generators 
    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2) 

    try:
        train_generator = datagen.flow_from_directory(
            PROCESSED_DATA_DIR,
            target_size=TARGET_SIZE,
            batch_size=BATCH_SIZE,
            class_mode='binary',
            subset='training',
            seed=42
        )

        validation_generator = datagen.flow_from_directory(
            PROCESSED_DATA_DIR,
            target_size=TARGET_SIZE,
            batch_size=BATCH_SIZE,
            class_mode='binary',
            subset='validation',
            shuffle=False, 
            seed=42
        )
    except Exception as e:
        print("\nFATAL ERROR: Could not initialize data generators.")
        print(f"Details: {e}")
        print(f"Path checked: {os.path.abspath(PROCESSED_DATA_DIR)}")
        print("Please verify the 'dataset/processed' directory exists and has FAKE/REAL subfolders.")
        exit()


    # 2. Final Data Check before Training
    print("\n--- DATA CHECK ---")
    print(f"Training samples found: {train_generator.samples}")
    print(f"Validation samples found: {validation_generator.samples}")
    print("------------------")

    # This is the IF/ELSE block that was causing the syntax error.
    # Note the correct indentation of the 'else:' statement.
    if train_generator.samples == 0:
        print("\nFATAL ERROR: Training data count is ZERO.")
        print("Ensure you have images in both FAKE and REAL folders under 'dataset/processed'.")
    else:
        # 3. Build Model
        deepfake_detector_model = build_simple_model()
        print("\n--- Model Summary ---")
        deepfake_detector_model.summary()
        
        # 4. Train Model
        print("\n--- Starting Model Training ---")
        steps_per_epoch = max(1, train_generator.samples // BATCH_SIZE)
        validation_steps = max(1, validation_generator.samples // BATCH_SIZE)

        H = deepfake_detector_model.fit(
            train_generator,
            steps_per_epoch=steps_per_epoch,
            epochs=EPOCHS,
            validation_data=validation_generator,
            validation_steps=validation_steps
        )

        # 5. Save Model
        deepfake_detector_model.save(MODEL_SAVE_PATH)
        print(f"\n✅ Model saved successfully to {MODEL_SAVE_PATH}")

        # 6. Evaluate Model
        print("\n--- Final Evaluation on Validation Set ---")
        
        # Predict on validation data
        Y_pred = deepfake_detector_model.predict(validation_generator)
        # Flatten predictions to 1D and convert to int labels (0/1)
        y_pred_binary = np.round(Y_pred).astype(int).flatten()

        # Get true labels, truncated to match the number of predictions
        num_predictions = len(y_pred_binary)
        y_true = validation_generator.classes[:num_predictions]

        # Get class names ordered by their index (ensures correct mapping)
        class_names = [name for name, idx in sorted(train_generator.class_indices.items(), key=lambda x: x[1])]
        
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred_binary, target_names=class_names))
        
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_true, y_pred_binary))