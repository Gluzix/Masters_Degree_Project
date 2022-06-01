from PyQt6.QtCore import QDir
import cv2
import dlib
import numpy as np
import os

if __name__ == '__main__':
    image_count = 0
    dir = QDir()

    path_images = dir.currentPath() + "/images/"
    path_train_src = dir.currentPath() + "/src/"
    path_train_tar = dir.currentPath() + "/tar/"

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(
        'E:/Projekt Magisterski/resources/pre_trained_data/shape_predictor_68_face_landmarks.dat')

    cascPath = "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cascPath)

    for filename in os.listdir(path_images):
        is_found = False
        f = os.path.join(path_images, filename)

        image = cv2.imread(f)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(
            gray_image,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        if len(faces) == 0:
            continue

        (x, y, w, h) = faces[0]
        cropped_image_gray = gray_image[y - 70:y + h + 70, x - 70:x + w + 70]
        cropped_image_color = image[y - 70:y + h + 70, x - 70:x + w + 70]

        try:
            resized_cropped_image_gray = cv2.resize(cropped_image_gray, (256, 256), interpolation=cv2.INTER_CUBIC)
            resized_cropped_image_color = cv2.resize(cropped_image_color, (256, 256), interpolation=cv2.INTER_CUBIC)
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
                cv2.circle(blank_image, (x, y), 2, (255, 255, 255), -1)

        if is_found:
            cv2.imwrite(f"{path_train_src}image_{image_count}.png", blank_image)
            cv2.imwrite(f"{path_train_tar}image_{image_count}.png", resized_cropped_image_color)
            image_count += 1
