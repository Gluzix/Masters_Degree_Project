from os import listdir
from numpy import asarray
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from keras.models import load_model
from matplotlib import pyplot
import cv2
import dlib
import numpy as np
from PIL import Image
from numpy import load
from Pix2Pix.pix_2_pix_model import Pix2PixModel
from Pix2Pix.pix_2_pix_trainer import Pix2PixTrainer


class Pix2Pix:
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
    def gather_image_and_try_predict():
        cascPath = "haarcascade_frontalface_default.xml"
        faceCascade = cv2.CascadeClassifier(cascPath)

        cap = cv2.VideoCapture(0)
        ret, image = cap.read()
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(
            gray_image,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        (x, y, w, h) = faces[0]

        cropped_image_gray = gray_image[y - 70:y + h + 70, x - 70:x + w + 70]
        cropped_image_color = image[y - 70:y + h + 70, x - 70:x + w + 70]

        try:
            resized_cropped_image_gray = cv2.resize(cropped_image_gray, (256, 256), interpolation=cv2.INTER_CUBIC)
            resized_cropped_image_color = cv2.resize(cropped_image_color, (256, 256), interpolation=cv2.INTER_CUBIC)
        except Exception as e:
            print(str(e))
            return

        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(
            'E:/Projekt Magisterski/resources/pre_trained_data/shape_predictor_68_face_landmarks.dat')
        rects = detector(resized_cropped_image_gray, 1)

        height, width, channel = resized_cropped_image_color.shape
        blank_image = np.zeros((height, width, 3), np.uint8)
        for rect in rects:
            shape = predictor(resized_cropped_image_gray, rect)
            shape_numpy_arr = np.zeros((68, 2), dtype='int')
            for i in range(0, 68):
                shape_numpy_arr[i] = (shape.part(i).x, shape.part(i).y)

            for i, (x, y) in enumerate(shape_numpy_arr):
                cv2.circle(blank_image, (x, y), 2, (255, 255, 255), -1)

        cap.release()

        image_pil = Image.fromarray(blank_image)
        test_ = image_pil.resize((256, 256), resample=Image.NEAREST)

        # Model loading...
        model = load_model("E:/Projekt Magisterski/resources/model_resources/model_071500.h5")

        arr = img_to_array(test_)
        arr = arr[np.newaxis, ...]

        X_fakeB, _ = Pix2PixTrainer.generate_fake_samples(model, arr, 1)
        X_fakeB = (X_fakeB + 1) / 2.0

        fig, axs = pyplot.subplots(2, 2)
        axs[0, 0].imshow(arr[0])
        axs[0, 0].set_title('from camera - landmark')
        axs[0, 1].imshow(X_fakeB[0])
        axs[0, 1].set_title('from camera - predicted')

        dataset = Pix2Pix.load_real_samples('E:/Projekt Magisterski/resources/model_resources/maps_256_2.npz')
        [X_realA_2, X_realB_2], _ = Pix2PixTrainer.generate_real_samples(dataset, 1, 1)
        X_realA_2 = (X_realA_2 + 1) / 2.0
        X_fakeB_2, _ = Pix2PixTrainer.generate_fake_samples(model, X_realA_2, 1)
        X_fakeB_2 = (X_fakeB_2 + 1) / 2.0

        axs[1, 0].imshow(X_realA_2[0])
        axs[1, 0].set_title('from dataset - landmark')
        axs[1, 1].imshow(X_fakeB_2[0])
        axs[1, 1].set_title('from dataset - predicted')

        pyplot.show()

    @staticmethod
    def try_to_predict(path_to_dataset: str, path_to_model: str):
        dataset = Pix2Pix.load_real_samples(path_to_dataset)
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

        cv2.imshow("test", X_fakeB[0])

        pyplot.show()

    @staticmethod
    def try_to_predict_and_show_multiple_frames(path_to_dataset: str, path_to_model: str):
        dataset = Pix2Pix.load_real_samples(path_to_dataset)
        model = load_model(path_to_model)
        [X_realA, X_realB] = Pix2PixTrainer.generate_real_samples_array(dataset)
        counter = 0

        for realA, realB in zip(X_realA, X_realB):
            realA = realA[np.newaxis, ...]
            realA = (realA + 1) / 2.0
            X_fakeB, y = Pix2PixTrainer.generate_fake_samples(model, realA, 1)
            X_fakeB = (X_fakeB + 1) / 2.0
            # vis = np.concatenate((realB, X_fakeB[0]), axis=1)
            cv2.imshow("name1", X_fakeB[0])
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            counter += 1

    @staticmethod
    def train(path_to_dataset: str):
        dataset = Pix2Pix.load_real_samples(path_to_dataset)
        image_shape = dataset[0].shape[1:]
        model = Pix2PixModel(image_shape)
        trainer = Pix2PixTrainer(model.discriminator, model.generator, model.gan, dataset)
        trainer.train()

    @staticmethod
    def create_dataset(path_to_source: str, path_to_target: str, filename: str):
        [src_images, tar_images] = Pix2Pix.load_images(path_to_source, path_to_target)
        print('Loaded: ', src_images.shape, tar_images.shape)
        np.savez_compressed(filename, src_images, tar_images)
        print('saved dataset: ', filename)
        Pix2Pix.plot_images(filename)


if __name__ == '__main__':
    # Pix2Pix.try_to_predict('E:/Projekt Magisterski/Pix2Pix/maps_256_kamil_256_.npz',
    #                        'E:/Projekt Magisterski/Pix2Pix/model_120000.h5')

    # Pix2Pix.gather_image_and_try_predict()

    # Pix2Pix.train('maps_256_trump_256_at_begin.npz')

    # Pix2Pix.try_to_predict('maps_256_kamil.npz',
    #                        'model_062560.h5')

    # Pix2Pix.create_dataset('src/',
    #                        'tar/',
    #                        'maps_256_duda_256_.npz')

    # Pix2Pix.try_to_predict_and_show_multiple_frames('maps_256_kamil_256_.npz', 'model_142800.h5')
    # Pix2Pix.try_to_predict_and_show_multiple_frames('maps_256_kamil_256_.npz', 'model_120000.h5')

    # Pix2Pix.plot_images("maps_256_trump_256_at_begin.npz")
