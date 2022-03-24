import cv2
import dlib
import numpy as np

if __name__ == '__main__':
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('E:/Projekt Magisterski/pre_trained_data/shape_predictor_68_face_landmarks.dat')
    cap = cv2.VideoCapture(0)

    while True:
        ret, image = cap.read()
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        rects = detector(gray_image, 1)

        for rect in rects:
            shape = predictor(gray_image, rect)
            shape_numpy_arr = np.zeros((68, 2), dtype='int')
            for i in range(0, 68):
                shape_numpy_arr[i] = (shape.part(i).x, shape.part(i).y)

            for i, (x, y) in enumerate(shape_numpy_arr):
                cv2.circle(image, (x, y), 1, (0, 0, 255), -1)

        cv2.imshow("Landmark Detection", image)

        if cv2.waitKey(10) == 27:
            break

    cap.release()

# code taken from: https://towardsdatascience.com/face-landmark-detection-using-python-1964cb620837
