from os import listdir
from numpy import asarray
from numpy import load
from matplotlib import pyplot
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from numpy import savez_compressed


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


def plot_images():
    data = load('maps_256.npz')
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


if __name__ == '__main__':
    path_src = 'train/src/'
    path_tar = 'train/tar/'
    [src_images, tar_images] = load_images(path_src, path_tar)
    print('Loaded: ', src_images.shape, tar_images.shape)
    filename = 'maps_256.npz'
    savez_compressed(filename, src_images, tar_images)
    print('saved dataset: ', filename)

    plot_images()
