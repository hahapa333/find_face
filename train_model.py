import cv2
import os
import numpy as np

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

def load_data():
    faces = []
    labels = []
    label_map = {}
    current_id = 0

    dataset_dir = 'dataset'
    for filename in os.listdir(dataset_dir):
        if not filename.endswith('.jpg'):
            continue
        path = os.path.join(dataset_dir, filename)
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue

        # Извлекаем имя из имени файла: "alex_0.jpg" → "alex"
        name = filename.split('_')[0]

        if name not in label_map:
            label_map[name] = current_id
            current_id += 1

        label = label_map[name]
        faces.append(image)
        labels.append(label)

    return faces, np.array(labels), {v: k for k, v in label_map.items()}

print("Обучение модели...")
faces, labels, label_names = load_data()
recognizer.train(faces, labels)
recognizer.save('trained_model.yml')
print("Модель сохранена: trained_model.yml")

# Сохраним имена
import json
with open('labels.json', 'w') as f:
    json.dump(label_names, f)

print("Готово. Теперь можно распознавать лица.")