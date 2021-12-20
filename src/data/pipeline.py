import fire
from preprocess_data import get_classes, load_classes_data_paths
from preprocess_data import save_preprocessed_images, split_train_test_validation_sets


def run_data_preprocessing_pipeline(input_dir, output_dir, train_ratio, validation_ratio):
    '''
    Loads images from the input directory, preprocesses, splits and saves them to the output directory.
    '''
    classes = get_classes(input_dir + 'names.txt')
    classes_images = load_classes_data_paths(classes, input_dir)

    train_classes_images, test_classes_images, val_classes_images = split_train_test_validation_sets(
        classes_images, train_ratio, validation_ratio)

    train_images_dir = output_dir + 'train/'
    test_images_dir = output_dir + 'test/'
    val_images_dir = output_dir + 'validation/'

    save_preprocessed_images(train_images_dir, train_classes_images)
    save_preprocessed_images(test_images_dir, test_classes_images)
    save_preprocessed_images(val_images_dir, val_classes_images)

    classes_images_train = load_classes_data_paths(classes, train_images_dir)
    classes_images_test = load_classes_data_paths(classes, test_images_dir)
    classes_images_validation = load_classes_data_paths(
        classes, val_images_dir)

    return classes, classes_images_train, classes_images_test, classes_images_validation


def cli_data_preprocessing_pipeline(input_dir, output_dir, train_ratio=0.7, validation_ratio=0.1):
    '''
    Loads images from the input directory, preprocesses, splits 
    and saves them to the output directory.

    :param input_dir: directory with images
    :param output_dir: directory to save preprocessed images
    :param train_ratio: ratio of train images
    :param validation_ratio: ratio of validation images
    '''
    print('Preprocessing data...')

    run_data_preprocessing_pipeline(
        input_dir, output_dir, train_ratio, validation_ratio)

    return 'Done preprocessing data ✅'


if __name__ == '__main__':
    fire.Fire(cli_data_preprocessing_pipeline)
