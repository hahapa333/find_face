import cv2
from ultralytics import YOLO
import os


# Папка для сохранения
os.makedirs("detections", exist_ok=True)

model = YOLO('yolov8n.pt')

# Целевые классы
target_classes = {
    'person': 'человек',
    'cell phone': 'телефон',
    'cup': 'кружка'
}

# Получаем индексы классов
class_to_id = {name: idx for idx, name in model.names.items()}
try:
    class_ids = [class_to_id[cls] for cls in target_classes if cls in class_to_id]
    print(f"✅ Отслеживаемые классы: {target_classes} → индексы: {class_ids}")
except KeyError as e:
    print(f"❌ Класс не найден: {e}")
    exit()

frame_count = 0
# Флаги: сохранено ли уже
saved_flags = {cls: False for cls in target_classes}

# Попробуй разные индексы и бэкенды

cap = cv2.VideoCapture(0)  # Linux/macOS

if not cap.isOpened():
    print("❌ Камера не открылась!")
    exit()

# Функция: проиграть звук и сказать текст


while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Не читается кадр")
        break



    # Уменьшаем размер для скорости
    frame = cv2.resize(frame, (640, 480))

    # Делаем инференс
    results = model(frame, classes=class_ids, stream=False, verbose=False)
    result = results[0]

    # Имена обнаруженных объектов
    detected_names = [model.names[int(box.cls)] for box in result.boxes]

    # Проверяем каждый класс
    for en_name, ru_name in target_classes.items():
        if en_name in detected_names and not saved_flags[en_name]:
            # Получаем координаты первого подходящего объекта
            for box in result.boxes:
                if model.names[int(box.cls)] == en_name:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cropped = frame[y1:y2, x1:x2]

                    # Папка для обрезанных
                    os.makedirs("detections/cropped", exist_ok=True)
                    crop_filename = f"detections/cropped/cropped_{en_name.replace(' ', '_')}_{frame_count:04d}.jpg"
                    cv2.imwrite(crop_filename, cropped)
                    print(f"✂️  {en_name.capitalize()} найден! Обрезано и сохранено: {crop_filename}")


                    saved_flags[en_name] = True
                    break  # Только один объект за раз
            # Разные частоты:

            saved_flags[en_name] = True  # Отмечаем как сохранённый


    annotated_frame = results[0].plot()

    cv2.imshow('YOLOv8', annotated_frame)

    if cv2.waitKey(1) == 27:  # ESC
        break
    frame_count += 1
cap.release()
cv2.destroyAllWindows()
