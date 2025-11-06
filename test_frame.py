import cv2


def test_camera(index):
    print(f"Проверяю камеру {index}...")
    cap = cv2.VideoCapture(index)

    if not cap.isOpened():
        print(f"❌ Камера {index} — не открылась")
        return False

    ret, frame = cap.read()
    if not ret:
        print(f"❌ Камера {index} — нет кадра")
        cap.release()
        return False

    print(f"✅ Камера {index} — работает! Размер: {frame.shape}")
    cv2.imshow(f"Камера {index}", frame)
    cv2.waitKey(1500)
    cv2.destroyAllWindows()

    cap.release()
    return True


# Пробуем обе
test_camera(0)
test_camera(1)