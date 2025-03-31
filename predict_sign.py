import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Model and category setup
model = load_model("best_model.h5")
CATEGORIES = [
    "20 km/h", "30 km/h", "50 km/h", "60 km/h", "70 km/h", 
    "80 km/h", "End 80 km/h", "100 km/h", "120 km/h", "No passing",
    # ... Complete all 43 categories
    "End no passing"
]

class TrafficSignGUI:
    def __init__(self):
        self.window = tk.Tk()
        self.window.title("Traffic Sign Recognizer")
        self.window.geometry("500x600")
        
        # GUI Elements
        tk.Label(self.window, text="Upload Traffic Sign", font=('Arial', 16)).pack(pady=10)
        self.canvas = tk.Canvas(self.window, width=300, height=300)
        self.canvas.pack()
        tk.Button(self.window, text="Select Image", command=self.predict).pack(pady=10)
        self.result_label = tk.Label(self.window, text="", font=('Arial', 14))
        self.result_label.pack(pady=20)
        
    def predict(self):
        filetypes = [("PPM Images", "*.ppm"), ("All files", "*.*")]
        filepath = filedialog.askopenfilename(filetypes=filetypes)
        
        if filepath:
            try:
                # Process image
                img = self.preprocess_image(filepath)
                pred = model.predict(img)[0]
                class_id = np.argmax(pred)
                confidence = pred[class_id]
                
                # Display
                self.show_image(filepath)
                self.result_label.config(
                    text=f"Sign: {CATEGORIES[class_id]}\n"
                         f"Category: {class_id}\n"
                         f"Confidence: {confidence:.2%}"
                )
            except Exception as e:
                messagebox.showerror("Error", f"Prediction failed: {str(e)}")
    
    def preprocess_image(self, path):
        """Handle PPM and other formats"""
        img = cv2.imread(path)
        img = cv2.resize(img, (30, 30))
        img = img.astype('float32') / 255.0
        return np.expand_dims(img, axis=0)
    
    def show_image(self, path):
        img = Image.open(path).convert("RGB")
        img = img.resize((300, 300))
        self.photo = ImageTk.PhotoImage(img)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
    
    def run(self):
        self.window.mainloop()

if __name__ == "__main__":
    app = TrafficSignGUI()
    app.run()