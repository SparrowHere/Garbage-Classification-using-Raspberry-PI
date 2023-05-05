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
    text = LABELS[label_idx] + f" ({output[0][label_idx] * 100:.2f}"
    
    # Drawing the label text on the frame
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Showing the frame
    cv2.imshow('Classification', frame)
    
    try:
        # Seting up & starting the serial communication between the Arduino & the PC
        s = serial.Serial(
            "COM9",
            9600,
            parity = serial.PARITY_NONE,
            stopbits = serial.STOPBITS_ONE
        )
        
        sleep(2)
    
        s.reset_input_buffer()
        print("Serial communication has started.")
        
        # Sending the label index to the Arduino
        s.write(f"{LABELS[label_idx]}\n".encode())
    # When the serial communication is interrupted, the serial port is closed (communication is ended)
    except KeyboardInterrupt:
        print("Serial communication has ended.")

        s.close()
    
    # When the 'q' key is pressed, the program is terminated
    if cv2.waitKey(1) == ord('q'):
        break
    
    sleep(1)

# Cleaning up
cam.release()
cv2.destroyAllWindows()