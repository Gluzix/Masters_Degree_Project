from PyQt6.QtCore import QDir
import cv2
import dlib
import numpy as np
import os


class LandmarkImagesCreator:

    @staticmethod
    def create_landmarks(path_to_images: str, path_to_predictor: str, path_to_haarcascade: str):
        image_count = 0

        dir = QDir()

        if not QDir(path_to_images).exists():
            return False

        path_train_src = dir.currentPath() + "/src/"
        path_train_tar = dir.currentPath() + "/tar/"

        if not QDir(path_train_src).exists():
            QDir().mkdir(path_train_src)

        if not QDir(path_train_tar).exists():
            QDir().mkdir(path_train_tar)

        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(path_to_predictor)

        face_cascade = cv2.CascadeClassifier(path_to_haarcascade)

        for filename in os.listdir(path_to_images):
            is_found = False
            f = os.path.join(path_to_images, filename)

            image = cv2.imread(f)
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            faces = face_cascade.detectMultiScale(
                gray_image,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE
            )

            if len(faces) == 0:
                continue

            for ZZ in (0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110):
                (x, y, w, h) = faces[0]

                if ZZ == 100:
                    cropped_image_gray = gray_image[y:y + h + 20, x - 50:x + w + 50]
                    cropped_image_color = image[y:y + h + 20, x - 50:x + w + 50]
                elif ZZ == 110:
                    cropped_image_gray = gray_image[y - 50:y + h + 50, x:x + w + 20]
                    cropped_image_color = image[y - 50:y + h + 50, x:x + w + 20]
                else:
                    cropped_image_gray = gray_image[y - ZZ:y + h + ZZ, x - ZZ:x + w + ZZ]
                    cropped_image_color = image[y - ZZ:y + h + ZZ, x - ZZ:x + w + ZZ]

                try:
                    resized_cropped_image_gray = cv2.resize(cropped_image_gray, (256, 256), interpolation=cv2.INTER_NEAREST)
                    resized_cropped_image_color = cv2.resize(cropped_image_color, (256, 256), interpolation=cv2.INTER_NEAREST)
                except Exception as e:
                    print(str(e))
                    continue

                rects = detector(resized_cropped_image_gray, 1)
                height, width, channel = resized_cropped_image_color.shape

                blank_image = np.zeros((height, width, 3), np.uint8)

                if rects:
                    is_found = True

                for rect in rects:
                    shape = predictor(resized_cropped_image_gray, rect)
                    shape_numpy_arr = np.zeros((68, 2), dtype='int')
                    for i in range(0, 68):
                        shape_numpy_arr[i] = (shape.part(i).x, shape.part(i).y)

                    for i, (x, y) in enumerate(shape_numpy_arr):
                        cv2.circle(blank_image, (x, y), 1, (255, 255, 255), -1)

                if is_found:
                    cv2.imwrite(f"{path_train_src}image_{image_count}.png", blank_image)
                    cv2.imwrite(f"{path_train_tar}image_{image_count}.png", resized_cropped_image_color)
                    image_count += 1
