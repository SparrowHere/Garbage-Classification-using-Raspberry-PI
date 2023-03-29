import os
import cv2
import numpy as np
import tensorflow as tf

# Model dosyasının yolu
MODEL_PATH = '/path/to/your/model.tflite'

# Sınıf etiketleri
LABELS = ['class1', 'class2', 'class3']

# Girdi görüntüsünün boyutları
INPUT_SIZE = (224, 224)

# Modeli yükleme
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

# Girdi ve çıktı tensorlarını alma
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Kamera ayarları
cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    # Kameradan bir görüntü alma
    ret, frame = cam.read()

    # Girdi görüntüsünü model için uygun boyuta getirme ve normalizasyon
    img = cv2.resize(frame, INPUT_SIZE)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    img = img.astype(np.float32)

    # Girdi tensorunu modelde çalıştırma
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()

    # Çıktı tensorunu alma ve sınıf etiketini bulma
    output = interpreter.get_tensor(output_details[0]['index'])
    label_idx = np.argmax(output)
    label = LABELS[label_idx]

    # Görüntüye sınıf etiketini ekleme
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Görüntüyü gösterme
    cv2.imshow('Classification', frame)

    # Çıkış tuşuna basıldığında programı sonlandırma
    if cv2.waitKey(1) == ord('q'):
        break

# Temizleme
cam.release()
cv2.destroyAllWindows()
