import cv2
import os

# Папка для фото
dataset_dir = 'dataset'
if not os.path.exists(dataset_dir):
    os.makedirs(dataset_dir)

# Каскад
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
if face_cascade.empty():
    print("❌ Не удалось загрузить каскад")
    exit()

# Камера
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("❌ Камера не открыта")
    exit()

cv2.namedWindow('Сбор данных', cv2.WINDOW_GUI_NORMAL)

# Ввод имени
user_id = input("Введите имя (латиница): ").strip()
user_id = ''.join(c for c in user_id if c.isalnum())
if not user_id:
    user_id = 'user'

count = 0
print("Ждём кадры... Нажмите 'c' — сделать снимок, 'q' — выход.")

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        print("❌ Проблема с кадром")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=3, minSize=(50, 50))

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    cv2.imshow('Сбор данных', frame)

    # 🔍 Отладка: покажем, какую клавишу нажали
    key = cv2.waitKey(30) & 0xFF  # увеличили задержку до 30 мс — лучше для отладки

    if key != 255:  # 255 = ничего не нажато
        print(f"⌨️ Нажата клавиша: {key} (символ: {chr(key) if key < 128 else '?'})")

    if key == 99 and len(faces) > 0:
        cv2.putText(frame, "Lico naideno!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        (x, y, w, h) = faces[0]
        face = gray[y:y+h, x:x+w]
        filename = f"{dataset_dir}/{user_id}_{count}.jpg"
        cv2.imwrite(filename, face)
        print(f"✅ Сохранено: {filename}")
        count += 1
    elif key == ord('q'):
        print("Выход...")
        break
    else:
        cv2.putText(frame, "Nety lica", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        print("⚠️ Лицо не обнаружено — не могу сохранить")

cap.release()
cv2.destroyAllWindows()