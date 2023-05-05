import os
import cv2
import numpy as np
import tensorflow as tf
from time import sleep
import serial

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

def get_frame(cam):
    ret, frame = cam.read()
    return frame

def save_frame(frame):
    cv2.imwrite('frame.jpg', frame)
    return frame

def load_frame():
    return cv2.imread('frame.jpg')

def normalize_and_resize(frame):
    img = np.expand_dims(cv2.resize(frame, INPUT_SIZE) / 255.0, axis = 0).astype(np.float32)
    return img

def classify_frame(interpreter, img):
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    label_idx = np.argmax(output)
    text = LABELS[label_idx] + f" ({output[0][label_idx] * 100:.2f}"
    return text, label_idx

def draw_label(frame, text):
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    return frame

def send_to_arduino(label_idx):
    try:
        s = serial.Serial("COM9", 9600, parity = serial.PARITY_NONE, stopbits = serial.STOPBITS_ONE)
        sleep(2)
        s.reset_input_buffer()
        print("Serial communication has started.")
        s.write(f"{LABELS[label_idx]}\n".encode())
    except KeyboardInterrupt:
        print("Serial communication has ended.")
        s.close()

def process_frame(cam):
    frame = get_frame(cam)
    frame = save_frame(frame)
    img = load_frame()
    img = normalize_and_resize(img)
    text, label_idx = classify_frame(interpreter, img)
    frame = draw_label(frame, text)
    cv2.imshow('Classification', frame)
    send_to_arduino(label_idx)

while True:
    try:
        process_frame(cam)
        if cv2.waitKey(1) == ord('q'):
            break
        sleep(1)
    except KeyboardInterrupt:
        print("Program terminated.")
        break

# Cleaning up
cam.release()
cv2.destroyAllWindows()
