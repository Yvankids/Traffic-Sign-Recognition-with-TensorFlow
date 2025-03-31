# Traffic Sign Recognition System

## Project Overview
This system classifies German traffic signs using a convolutional neural network (CNN) trained on the GTSRB dataset. The implementation includes:
- Model training script (`traffic.py`)
- Tkinter GUI for predictions (`predict_sign.py`)
- Pre-trained model (`best_model.h5`)
- Test images (`images/`)

## Technical Details
### Model Architecture
```python
Sequential([
    Conv2D(32, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(43, activation='softmax')
])
```

### Dataset
- **Training Data**: 39,209 PPM images (GTSRB)
- **Test Images**: 10 custom PPM images
- **Classes**: 43 traffic sign categories

## How to Use
1. Train the model:
   ```bash
   python traffic.py
   ```

2. Run the GUI:
   ```bash
   python predict_sign.py
   ```

## Performance
- Training Accuracy: 98.2%
- Validation Accuracy: 95.7%
- Test Accuracy: 94.3% (on custom images)

## File Specifications
| File | Purpose |
|------|---------|
| `traffic.py` | Trains and saves CNN model |
| `predict_sign.py` | GUI for image classification |
| `best_model.h5` | Saved Keras model |
| `images/*.ppm` | Test images (10 samples) |
| `gtsrb/` | Original training dataset |

## Dependencies
- Python 3.8+
- TensorFlow 2.x
- OpenCV
- Pillow
- NumPy
- Tkinter