import os
import cv2
import numpy as np
import tensorflow as tf
from time import sleep
import keyboard

# Path to the model
MODEL_PATH = os.getcwd() + "\model.tflite"

# Class labels (one-hot encoding)
LABELS = ['No Bottle', 'Plastic', 'Paper', 'Glass', 'Metal']

# Input size of the model
INPUT_SIZE = (224, 224)

# Loading & allocating the model
interpreter = tf.lite.Interpreter(model_path = MODEL_PATH)
interpreter.allocate_tensors()

# Getting input & output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Starting the camera
cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1080)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while True:
    # Reading the input frame
    ret, frame = cam.read()
    
    # Saving the input frame
    cv2.imwrite('frame.jpg', frame)
    img = cv2.imread('frame.jpg')
    
    # Normalizing & resizing the input frame
    img = np.expand_dims(cv2.resize(frame, INPUT_SIZE) / 255.0, axis = 0).astype(np.float32)

    # Setting the input tensor
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()

    # Getting the output tensor & label index (reformatting the output)
    output = interpreter.get_tensor(output_details[0]['index'])
    label_idx = np.argmax(output)
    text = LABELS[label_idx] + f" ({output[0][label_idx] * 100:.2f})"
    
    # Putting the label on the saved image (labeling the image)
    img = cv2.putText(cv2.imread('frame.jpg'), text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
    cv2.imwrite('frame.jpg', img)
    
    if keyboard.is_pressed("q"):
        break
    
    sleep(1)

# Cleaning up
cam.release()
cv2.destroyAllWindows()
