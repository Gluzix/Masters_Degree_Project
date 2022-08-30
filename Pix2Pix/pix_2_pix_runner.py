from os import listdir
from numpy import asarray
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from keras.models import load_model
from matplotlib import pyplot
import cv2
import dlib
import numpy as np
from numpy import load
from Pix2Pix.pix_2_pix_model import Pix2PixModel
from Pix2Pix.pix_2_pix_trainer import Pix2PixTrainer


class Pix2PixRunner:
    def __init__(self):
        pass

    @staticmethod
    def load_real_samples(filename):
        data = load(filename)
        X1, X2 = data['arr_0'], data['arr_1']
        X1 = (X1 - 127.5) / 127.5
        X2 = (X2 - 127.5) / 127.5
        return [X1, X2]

    @staticmethod
    def load_images(path_to_src, path_to_tar, size=(256, 256)):
        src_list = list()
        tar_list = list()

        for filename in listdir(path_to_src):
            pixels = load_img(path_to_src+filename, target_size=size)
            pixels = img_to_array(pixels)
            src_list.append(pixels)

        for filename in listdir(path_to_tar):
            pixels = load_img(path_to_tar+filename, target_size=size)
            pixels = img_to_array(pixels)
            tar_list.append(pixels)

        return [asarray(src_list), asarray(tar_list)]

    @staticmethod
    def plot_images(filename):
        data = load(filename)
        src_images, tar_images = data['arr_0'], data['arr_1']
        print('Loaded: ', src_images.shape, tar_images.shape)
        n_samples = 3
        for i in range(n_samples):
            pyplot.subplot(2, n_samples, 1 + i)
            pyplot.axis('off')
            pyplot.imshow(src_images[i].astype('uint8'))
        for i in range(n_samples):
            pyplot.subplot(2, n_samples, 1 + n_samples + i)
            pyplot.axis('off')
            pyplot.imshow(tar_images[i].astype('uint8'))
        pyplot.show()

    @staticmethod
    def try_to_predict(path_to_dataset: str, path_to_model: str):
        dataset = Pix2PixRunner.load_real_samples(path_to_dataset)
        model = load_model(path_to_model)
        [X_realA, X_realB], _ = Pix2PixTrainer.generate_real_samples(dataset, 1, 1)
        X_realA = (X_realA + 1) / 2.0
        X_fakeB, _ = Pix2PixTrainer.generate_fake_samples(model, X_realA, 1)
        X_fakeB = (X_fakeB + 1) / 2.0

        fig, axs = pyplot.subplots(1, 3)

        axs[0].imshow(X_realB[0])
        axs[0].set_title('Real')
        axs[1].imshow(X_realA[0])
        axs[1].set_title('Landmark')
        axs[2].imshow(X_fakeB[0])
        axs[2].set_title('Fake')

        pyplot.show()

    @staticmethod
    def try_to_predict_same_image(path_to_dataset: str, path_to_model: str):
        dataset = Pix2PixRunner.load_real_samples(path_to_dataset)
        model = load_model(path_to_model)
        [X_realA, X_realB], _ = Pix2PixTrainer.generate_one_real_sample(dataset, 1)

        X_fakeB, _ = Pix2PixTrainer.generate_fake_samples(model, X_realA, 1)
        X_fakeB = (X_fakeB + 1) / 2.0
        X_realB = (X_realB + 1) / 2.0

        fig, axs = pyplot.subplots(1, 3)

        axs[0].imshow(X_realB[0])
        axs[0].set_title('Real')
        axs[1].imshow(X_realA[0])
        axs[1].set_title('Landmark')
        axs[2].imshow(X_fakeB[0])
        axs[2].set_title('Fake')

        pyplot.show()

    @staticmethod
    def try_to_predict_and_show_multiple_frames(path_to_dataset: str, path_to_model: str):
        dataset = Pix2PixRunner.load_real_samples(path_to_dataset)
        model = load_model(path_to_model)
        [X_realA, X_realB] = Pix2PixTrainer.generate_real_samples_array(dataset)
        counter = 0

        for realA, realB in zip(X_realA, X_realB):
            realA = realA[np.newaxis, ...]
            realA = (realA + 1) / 2.0
            X_fakeB, y = Pix2PixTrainer.generate_fake_samples(model, realA, 1)
            X_fakeB = (X_fakeB + 1) / 2.0
            cv2.imshow("name1", X_fakeB[0])
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            counter += 1

    @staticmethod
    def train(path_to_dataset: str):
        dataset = Pix2PixRunner.load_real_samples(path_to_dataset)
        image_shape = dataset[0].shape[1:]
        model = Pix2PixModel(image_shape)
        trainer = Pix2PixTrainer(model.discriminator, model.generator, model.gan, dataset)
        trainer.train()

    @staticmethod
    def create_dataset(path_to_source: str, path_to_target: str, filename: str):
        [src_images, tar_images] = Pix2PixRunner.load_images(path_to_source, path_to_target)
        print('Loaded: ', src_images.shape, tar_images.shape)
        np.savez_compressed(filename, src_images, tar_images)
        print('saved dataset: ', filename)
        Pix2PixRunner.plot_images(filename)

    @staticmethod
    def predict_from_webcam(predictor_path: str, model_path: str, haarcascade_path: str):
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(predictor_path)
        model = load_model(model_path)

        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + haarcascade_path)

        vid = cv2.VideoCapture(0)
        while True:
            ret, frame = vid.read()
            gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = face_cascade.detectMultiScale(gray_image, 1.1, 4)

            if len(faces) == 0:
                continue

            (x, y, w, h) = faces[0]

            cropped_image_gray = gray_image[y - 30:y + h + 30, x - 30:x + w + 30]
            cropped_image_color = frame[y - 30:y + h + 30, x - 30:x + w + 30]

            try:
                resized_cropped_image_gray = cv2.resize(cropped_image_gray, (256, 256), interpolation=cv2.INTER_NEAREST)
                resized_cropped_image_color = cv2.resize(cropped_image_color, (256, 256),
                                                         interpolation=cv2.INTER_NEAREST)
            except Exception as e:
                print(str(e))
                continue

            rects = detector(resized_cropped_image_gray, 1)

            if not rects:
                continue

            height, width, channel = resized_cropped_image_color.shape
            blank_image = np.zeros((height, width, 3), np.uint8)

            for rect in rects:
                shape = predictor(cropped_image_gray, rect)
                shape_numpy_arr = np.zeros((68, 2), dtype='int')
                for i in range(0, 68):
                    shape_numpy_arr[i] = (shape.part(i).x, shape.part(i).y)

                for i, (x, y) in enumerate(shape_numpy_arr):
                    cv2.circle(blank_image, (x, y), 1, (255, 255, 255), -1)

            realA = np.expand_dims(blank_image, axis=0)
            realA = (realA - 127.5) / 127.5

            X_fakeB, _ = Pix2PixTrainer.generate_fake_samples(model, realA, 1)
            X_fakeB = (X_fakeB + 1) / 2.0
            X_fakeB[0] = cv2.cvtColor(X_fakeB[0], cv2.COLOR_BGR2RGB)

            cv2.imshow('frame', X_fakeB[0])

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        vid.release()
        cv2.destroyAllWindows()

    @staticmethod
    def test_load_from_npz_image_and_predict(real_samples_path: str, model_path: str):
        dataset = Pix2PixRunner.load_real_samples(real_samples_path)
        model = load_model(model_path)
        trainA, trainB = dataset

        img = trainA[0]
        img = img[np.newaxis, ...]

        X_fakeB, _ = Pix2PixTrainer.generate_fake_samples(model, img, 1)

        X_fakeB = (X_fakeB + 1) / 2.0

        pyplot.imshow(X_fakeB[0])
        pyplot.show()
