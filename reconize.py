# recognize.py
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

# Загрузка базы
if not os.path.exists("database.npy"):
    print("❌ Нет зарегистрированных лиц. Запустите register.py")
    exit()

db = np.load("database.npy", allow_pickle=True).item()
names = db["names"]
embeddings = db["embeddings"]

# Порог (0.4–0.6 для cosine)
THRESHOLD = 0.4

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("Распозnавание... Нажмите 'q' для выхода.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    detector.setInputSize((frame.shape[1], frame.shape[0]))
    _, faces = detector.detect(frame)

    if faces is not None:
        for face in faces:
            x, y, w, h = map(int, face[:4])
            aligned = recognizer.alignCrop(frame, face)
            embedding = recognizer.feature(aligned)

            # Поиск лучшего совпадения
            min_dist = 1.0
            best_idx = -1
            for i, saved_emb in enumerate(embeddings):
                dist = recognizer.match(embedding, saved_emb, cv2.FaceRecognizerSF_FR_COSINE)
                if dist < min_dist:
                    min_dist = dist
                    best_idx = i

            # Результат
            if min_dist < THRESHOLD:
                label = names[best_idx]
            else:
                label = "Neizvestno"

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} ({min_dist:.2f})", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("Raspoznavanie", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()