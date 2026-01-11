import cv2
import numpy as np
from tensorflow import keras
from PIL import Image

# Load model dan labels
print("Loading model...")
model = keras.models.load_model('model/keras_model.h5')

with open('model/labels.txt', 'r') as f:
    class_names = [line.strip() for line in f.readlines()]

print("Model loaded successfully!")
print(f"Classes: {class_names}")

# Function untuk preprocess frame
def preprocess_frame(frame):
    # Convert BGR (OpenCV) ke RGB
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Resize ke 224x224
    img = cv2.resize(img, (224, 224))
    
    # Normalize seperti Teachable Machine
    img_array = np.asarray(img, dtype=np.float32)
    normalized_image = (img_array / 127.5) - 1
    
    # Reshape untuk model
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image
    
    return data

# Function untuk predict
def predict_frame(frame):
    processed = preprocess_frame(frame)
    prediction = model.predict(processed, verbose=0)
    
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence = prediction[0][index]
    
    return class_name, confidence, prediction[0]

# Buka camera
print("Opening camera...")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Cannot open camera")
    exit()

print("\n=== CONTROLS ===")
print("Press 'q' to quit")
print("Press 's' to save screenshot")
print("Press SPACE to freeze/unfreeze")
print("================\n")

frozen = False
frozen_frame = None

while True:
    if not frozen:
        ret, frame = cap.read()
        if not ret:
            print("Error: Cannot read frame")
            break
        current_frame = frame.copy()
    else:
        current_frame = frozen_frame.copy()
    
    # Predict
    class_name, confidence, all_predictions = predict_frame(current_frame)
    
    # Ukuran frame
    height, width = current_frame.shape[:2]
    
    # Background untuk text (semi-transparent overlay)
    overlay = current_frame.copy()
    cv2.rectangle(overlay, (0, 0), (width, 180), (0, 0, 0), -1)
    current_frame = cv2.addWeighted(overlay, 0.6, current_frame, 0.4, 0)
    
    # Display prediction
    y_offset = 30
    cv2.putText(current_frame, f"Prediction: {class_name}", 
                (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 
                0.8, (0, 255, 0), 2)
    
    y_offset += 35
    cv2.putText(current_frame, f"Confidence: {confidence:.2%}", 
                (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, (255, 255, 255), 2)
    
    # Display all predictions
    y_offset += 40
    cv2.putText(current_frame, "All Predictions:", 
                (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 
                0.6, (200, 200, 200), 1)
    
    for i, (name, score) in enumerate(zip(class_names, all_predictions)):
        y_offset += 25
        color = (0, 255, 0) if i == np.argmax(all_predictions) else (150, 150, 150)
        cv2.putText(current_frame, f"{name}: {score:.1%}", 
                    (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, color, 1)
    
    # Status indicator
    if frozen:
        cv2.putText(current_frame, "FROZEN", 
                    (width - 120, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, (0, 255, 255), 2)
    
    # Show frame
    cv2.imshow('Skin Cancer Detection - Camera Test', current_frame)
    
    # Handle keyboard input
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('q'):
        print("Quitting...")
        break
    elif key == ord('s'):
        filename = f'capture_{class_name}_{confidence:.2f}.jpg'
        cv2.imwrite(filename, current_frame)
        print(f"Screenshot saved: {filename}")
    elif key == ord(' '):
        frozen = not frozen
        if frozen:
            frozen_frame = frame.copy()
            print("Frame frozen - press SPACE to unfreeze")
        else:
            print("Frame unfrozen")

# Cleanup
cap.release()
cv2.destroyAllWindows()
print("Camera closed.")