from glob import glob
from skimage.filters import threshold_otsu
from skimage.color import rgb2gray
from skimage import img_as_ubyte

import skimage.io as io
import numpy as np
import os


def get_classes(path):
    '''
    Returns a list of classes
    '''
    try:
        classes_file = open(path, 'r')
    except FileNotFoundError:
        raise Exception('Classes text file not found')

    classes = [line.strip()[4:] for line in classes_file]
    return classes


def load_classes_data_paths(classes, images_dir):
    '''
    Loads images paths for each class.
    '''
    classes_images = dict()

    for idx, class_name in enumerate(classes):
        classes_images[class_name] = glob(
            images_dir + str(idx + 1) + '/*.jpg')

    return classes_images


def preprocess_image(image):
    '''
    Applies thresholding and color correction to the image.
    '''
    gray_scale_image = rgb2gray(image)
    binary_image = gray_scale_image > threshold_otsu(gray_scale_image)

    rows, columns = binary_image.shape
    rows -= 1
    columns -= 1

    corners = [binary_image[0, 0], binary_image[0, columns],
               binary_image[rows, 0], binary_image[rows, columns]]

    if np.argmax(np.bincount(corners)) == 1:
        binary_image = np.invert(binary_image)

    return binary_image


def save_preprocessed_images(output_dir, classes_images):
    '''
    Saves preprocessed images to the disk.
    '''
    for idx, class_images in enumerate(classes_images.values()):
        dir_name = output_dir + str(idx + 1) + '/'

        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        else:
            raise Exception('Directory already exists')

        for image_path in class_images:
            image = io.imread(image_path)

            transformed_image = preprocess_image(image)

            image_name = image_path.split('\\')[-1]

            io.imsave(dir_name + image_name,
                      img_as_ubyte(transformed_image))


def split_train_test_validation_sets(classes_images, train_ratio, validation_ratio):
    '''
    Splits images into train, test and validation sets.
    '''
    train_images = dict()
    test_images = dict()
    validation_images = dict()

    for class_name, class_images in classes_images.items():
        n_images = len(class_images)
        n_train_images = int(n_images * train_ratio)
        n_validation_images = int(n_images * validation_ratio)

        train_images[class_name] = class_images[:n_train_images]
        test_images[class_name] = class_images[n_train_images:n_train_images +
                                               n_validation_images]
        validation_images[class_name] = class_images[n_train_images +
                                                     n_validation_images:]

    return train_images, test_images, validation_images
