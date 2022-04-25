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


def load_real_samples(filename):
    data = load(filename)
    X1, X2 = data['arr_0'], data['arr_1']
    X1 = (X1 - 127.5) / 127.5
    X2 = (X2 - 127.5) / 127.5
    return [X1, X2]


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


def gather_image_and_try_predict():
    cap = cv2.VideoCapture(0)
    ret, image = cap.read()
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(
        'E:/Projekt Magisterski/pre_trained_data/shape_predictor_68_face_landmarks.dat')
    rects = detector(gray_image, 1)
    height, width, channel = image.shape
    blank_image = np.zeros((height, width, 3), np.uint8)
    for rect in rects:
        shape = predictor(gray_image, rect)
        shape_numpy_arr = np.zeros((68, 2), dtype='int')
        for i in range(0, 68):
            shape_numpy_arr[i] = (shape.part(i).x, shape.part(i).y)

        for i, (x, y) in enumerate(shape_numpy_arr):
            cv2.circle(image, (x, y), 2, (255, 255, 255), -1)
            cv2.circle(blank_image, (x, y), 2, (255, 255, 255), -1)

    pyplot.imshow(image)

    image_pil = Image.fromarray(blank_image)
    test_ = image_pil.resize((256, 256))
    pyplot.imshow(test_)
    pyplot.show()

    # Model loading...
    model = load_model("E:/Projekt Magisterski/Pix2Pix/model_020040.h5")
    # pixels = img_to_array(blank_image)

    img = Image.fromarray(blank_image)
    resized_image = img.resize((256, 256))
    pyplot.imshow(img)
    pyplot.show()
    arr = img_to_array(img)
    # arr = arr[np.newaxis, ...]
    pyplot.imshow(arr)

    # X_fakeB, _ = generate_fake_samples(model, arr, 1)
    # X_fakeB = (X_fakeB + 1) / 2.0
    # pyplot.imshow(X_fakeB[0])
    # pyplot.show()


def try_to_predict(path_to_dataset: str, path_to_model: str):
    dataset = load_real_samples(path_to_dataset)
    model = load_model(path_to_model)
    [X_realA, X_realB], _ = Pix2PixTrainer.generate_real_samples(dataset, 1, 1)
    X_realA = (X_realA + 1) / 2.0
    X_fakeB, _ = Pix2PixTrainer.generate_fake_samples(model, X_realA, 1)
    X_fakeB = (X_fakeB + 1) / 2.0
    pyplot.imshow(X_fakeB[0])
    pyplot.show()


def train(path_to_dataset: str):
    dataset = load_real_samples(path_to_dataset)
    image_shape = dataset[0].shape[1:]
    model = Pix2PixModel(image_shape)
    trainer = Pix2PixTrainer(model.discriminator, model.generator, model.gan, dataset)
    trainer.train()


def create_dataset(path_to_source: str, path_to_target: str, filename: str):
    [src_images, tar_images] = load_images(path_to_source, path_to_target)
    print('Loaded: ', src_images.shape, tar_images.shape)
    np.savez_compressed(filename, src_images, tar_images)
    print('saved dataset: ', filename)
    plot_images(filename)


if __name__ == '__main__':
    # try_to_predict('E:/Projekt Magisterski/resources/model_resources/maps_256_2.npz',
    #                'E:/Projekt Magisterski/resources/model_resources/model_020040.h5')

    # train('E:/Projekt Magisterski/resources/model_resources/maps_256_2.npz')

    # create_dataset('E:/Projekt Magisterski/resources/model_resources/train_2/src',
    #                'E:/Projekt Magisterski/resources/model_resources/train_2/tar',
    #                'maps_256_2.npz')

    # gather_image_and_try_predict()