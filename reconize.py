import cv2
import json
import os
import sys
import pyttsx3  # Import pyttsx3

# Initialize the text-to-speech engine
engine = pyttsx3.init()

# Optionally set properties like rate and volume
engine.setProperty('rate', 150)    # Speed of speech
engine.setProperty('volume', 1.0)  # Volume level (0.0 to 1.0)

# --- Пути к файлам ---
MODEL_PATH = 'trained_model.yml'
LABELS_PATH = 'labels.json'

# --- Параметры ---
CONFIDENCE_THRESHOLD = 80
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# --- Загрузка ресурсов ---
def load_resources():
    print("🔍 Загрузка ресурсов...")

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    if not os.path.exists(MODEL_PATH):
        print(f"❌ Не найдена модель: {MODEL_PATH}")
        sys.exit(1)
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(MODEL_PATH)
    print(f"✅ Модель загружена: {MODEL_PATH}")

    if not os.path.exists(LABELS_PATH):
        print(f"⚠️  Не найдены метки: {LABELS_PATH} — используем пустой словарь")
        label_names = {}
    else:
        with open(LABELS_PATH, 'r', encoding='utf-8') as f:
            try:
                label_names = json.load(f)
                print(f"✅ Загружено меток: {len(label_names)}")
            except json.JSONDecodeError as e:
                print(f"❌ Ошибка в JSON: {e}")
                sys.exit(1)

    return face_cascade, recognizer, label_names

# --- Основная функция ---
def main():
    face_cascade, recognizer, label_names = load_resources()

    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        print("❌ Камера не открыта. Попробуй CAP_V4L вместо CAP_V4L2.")
        cap.open(0, cv2.CAP_V4L)
        if not cap.isOpened():
            sys.exit(1)

    print("🎥 Камера запущена. Нажмите 'q', чтобы выйти.")

    cv2.namedWindow('Распознавание лиц', cv2.WINDOW_AUTOSIZE)

    # Keep track of who has been greeted
    greeted = set()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️  Ошибка чтения кадра")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(60, 60),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        for (x, y, w, h) in faces:
            face_roi = gray[y:y+h, x:x+w]
            label_id, confidence = recognizer.predict(face_roi)

            if confidence < CONFIDENCE_THRESHOLD:
                name = label_names.get(str(label_id), "Неизвестно")
                text = f"{name} ({confidence:.1f})"
                color = (0, 255, 0)

                # Only greet if this person hasn't been greeted yet
                if name not in greeted:
                    greeting = f"Hello, {name}!"
                    print(greeting)
                    engine.say(greeting)
                    engine.runAndWait()
                    greeted.add(name)  # Mark as greeted
            else:
                text = "Неизвестно"
                color = (0, 0, 255)

            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, text, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

        cv2.imshow('Распознавание лиц', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    print("👋 Программа завершена.")

if __name__ == "__main__":
    main()