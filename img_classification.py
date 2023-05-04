import os
import cv2
import numpy as np
import tensorflow as tf
# from smbus2 import SMBus as smbus
from time import sleep
import serial

# Model dosyasının yolu
MODEL_PATH = os.getcwd() + "\model.tflite"

# Sınıf etiketleri (one-hot encoding)
LABELS = ['No Bottle', 'Plastic', 'Paper', 'Glass', 'Metal']

# Girdi görüntüsünün boyutları
INPUT_SIZE = (224, 224)

# Modeli yükleme
interpreter = tf.lite.Interpreter(model_path = MODEL_PATH)
interpreter.allocate_tensors()

# Girdi ve çıktı tensorlarını alma
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Kamera ayarları
cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1080)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while True:
    # Kameradan bir görüntü alma
    ret, frame = cam.read()
    
    # Girdi Görüntüsünü fotoğraf olarak kaydetme ve atama
    cv2.imwrite('frame.jpg', frame)
    img = cv2.imread('frame.jpg')
    
    # Girdi görüntüsünü model için uygun boyuta getirme ve normalizasyon
    img = np.expand_dims(cv2.resize(frame, INPUT_SIZE) / 255.0, axis = 0).astype(np.float32)

    # Girdi tensorunu modelde çalıştırma
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()

    # Çıktı tensorunu alma ve sınıf etiketini bulma
    output = interpreter.get_tensor(output_details[0]['index'])
    label_idx = np.argmax(output)
    text = LABELS[label_idx] + f" ({output[0][label_idx] * 100:.2f}"
    
    # Görüntüye sınıf etiketini ekleme
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Görüntüyü gösterme
    cv2.imshow('Classification', frame)
    
    try:
        # Arduino ile seri haberleşme başlatma
        s = serial.Serial(
            "COM9",
            9600,
            parity = serial.PARITY_NONE,
            stopbits = serial.STOPBITS_ONE
        )
        
        sleep(2)
        
        s.reset_input_buffer()
        print("Serial communication has started.")
        
        # Sınıf etiketini Arduino'ya gönderme
        s.write(f"{LABELS[label_idx]}\n".encode())
    except KeyboardInterrupt:
        print("Serial communication has ended.")
        
        # Herhangi bir klavye tuşuna basıldığında seri haberleşmeyi sonlandırma
        s.close()
    
    # Çıkış tuşuna basıldığında programı sonlandırma
    if cv2.waitKey(1) == ord('q'):
        break
    
    sleep(1)

# Temizleme
cam.release()
cv2.destroyAllWindows()