import os
import cv2
import numpy as np
from tensorflow.keras.models import Sequential, save_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

NUM_CATEGORIES = 43
IMG_SIZE = 30

def load_data(data_dir):
    """Load PPM images from directory structure"""
    images = []
    labels = []
    
    for label in range(NUM_CATEGORIES):
        dir_path = os.path.join(data_dir, str(label))
        if not os.path.isdir(dir_path):
            continue
            
        for file in os.listdir(dir_path):
            if not file.endswith('.ppm'):
                continue
                
            try:
                img_path = os.path.join(dir_path, file)
                img = cv2.imread(img_path)
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                img = img.astype('float32') / 255.0
                images.append(img)
                labels.append(label)
            except Exception as e:
                print(f"Error processing {file}: {str(e)}")
    
    return np.array(images), np.array(labels)

def build_model():
    """CNN model architecture"""
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
        MaxPooling2D((2,2)),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D((2,2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(NUM_CATEGORIES, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

if __name__ == "__main__":
    print("Loading data...")
    X, y = load_data("gtsrb")
    
    print("Building model...")
    model = build_model()
    
    print("Training...")
    model.fit(X, y, epochs=10, validation_split=0.2)
    
    print("Saving model...")
    save_model(model, "best_model.h5")
    print("Training complete! Model saved to best_model.h5")