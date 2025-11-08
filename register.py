# register.py
import cv2
import numpy as np
import json
import os

# Загрузка моделей
detector = cv2.FaceDetectorYN.create(
    model="face_detection_yunet_2023mar.onnx",
    config="",
    input_size=(320, 320)
)
recognizer = cv2.FaceRecognizerSF.create(
    model="face_recognition_sface_2021dec.onnx",
    config=""
)

# База данных
DB_FILE = "database.npy"
NAMES_FILE = "names.json"

# Загрузка базы
if os.path.exists(DB_FILE):
    db = np.load(DB_FILE, allow_pickle=True).item()
    names = db["names"]
    embeddings = db["embeddings"]
else:
    names = []
    embeddings = []

print("Введите имя для регистрации:")
name = input("Имя: ").strip()

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("Наведите лицо на камеру. Нажмите 'c' — сделать снимок, 'q' — выход.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Настройка размера для детектора
    detector.setInputSize((frame.shape[1], frame.shape[0]))
    _, faces = detector.detect(frame)

    if faces is not None:
        for face in faces:
            # Обрезка и выравнивание лица
            aligned = recognizer.alignCrop(frame, face)
            embedding = recognizer.feature(aligned)

            # Отображение
            x, y, w, h = map(int, face[:4])
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, "Gotovo", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("Registraciya", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('c') and faces is not None:
        aligned = recognizer.alignCrop(frame, faces[0])
        embedding = recognizer.feature(aligned)
        names.append(name)
        embeddings.append(embedding)
        print(f"✅ {name} сохранён")
    elif key == ord('q'):
        break

# Сохранение
db = {"names": names, "embeddings": np.array(embeddings)}
np.save(DB_FILE, db, allow_pickle=True)

# Сохранение имён
with open(NAMES_FILE, 'w', encoding='utf-8') as f:
    json.dump(names, f, ensure_ascii=False)

print("База сохранена.")
cap.release()
cv2.destroyAllWindows()