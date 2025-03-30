import os
import sys
import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Constants
NUM_CATEGORIES = 43
IMG_WIDTH = 30
IMG_HEIGHT = 30

def load_data(data_dir):
    """
    Load image data from directory `data_dir`.
    
    Returns tuple (images, labels):
    - images: List of numpy arrays representing images
    - labels: List of integers representing categories
    """
    images = []
    labels = []
    
    # Loop through each category directory (0-42)
    for category in range(NUM_CATEGORIES):
        category_dir = os.path.join(data_dir, str(category))
        
        # Skip if directory doesn't exist
        if not os.path.isdir(category_dir):
            continue
            
        # Process each image in the category directory
        for filename in os.listdir(category_dir):
            # Skip hidden files
            if filename.startswith('.'):
                continue
                
            filepath = os.path.join(category_dir, filename)
            
            try:
                # Read and resize image
                img = cv2.imread(filepath)
                if img is None:
                    continue
                    
                img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
                
                # Normalize pixel values
                img = img.astype('float32') / 255.0
                
                images.append(img)
                labels.append(category)
            except Exception as e:
                print(f"Error processing {filepath}: {e}", file=sys.stderr)
    
    return (np.array(images), np.array(labels))

def get_model():
    """
    Returns a compiled convolutional neural network model.
    """
    model = Sequential([
        # First convolutional layer
        Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
        MaxPooling2D(pool_size=(2, 2)),
        
        # Second convolutional layer
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        
        # Flatten and dense layers
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        
        # Output layer
        Dense(NUM_CATEGORIES, activation='softmax')
    ])
    
    # Compile model
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def main():
    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python traffic.py data_directory [model.h5]")
    
    # Get image arrays and labels for all image files
    images, labels = load_data(sys.argv[1])
    
    # Split data into training and testing sets
    split = int(0.9 * len(images))
    x_train, y_train = images[:split], labels[:split]
    x_test, y_test = images[split:], labels[split:]
    
    # Get a compiled neural network
    model = get_model()
    
    # Fit model on training data
    model.fit(x_train, y_train, epochs=10)
    
    # Evaluate neural network performance
    model.evaluate(x_test, y_test, verbose=2)
    
    # Save model to file
    if len(sys.argv) == 3:
        filename = sys.argv[2]
        model.save(filename)
        print(f"Model saved to {filename}.")

if __name__ == "__main__":
    main()
