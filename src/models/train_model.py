from sklearn import svm

from skimage.exposure import histogram
from skimage import io
from skimage.filters import threshold_otsu
from skimage.color import rgb2gray

from scipy.signal import convolve2d

import time
import numpy as np
import os
import fire
import pickle


def preprocessing_image(img):
    '''
    DESCRIPTION:
    Preprocess an image.
        1. Grayscale
        2. OTSU Threshold
        3. Binarization
        4. Checking image binary is 0 or 1
    RETURN:
    Preprocessd Image
    '''
    grayscale_image = rgb2gray(img)
    if grayscale_image.max() <= 1:
        grayscale_image = (grayscale_image * 255)
    grayscale_image = grayscale_image.astype(np.uint8)

    global_threshold = threshold_otsu(grayscale_image)
    binary_image = np.where(grayscale_image > global_threshold, 255, 0)

    image_histogram = np.asarray(histogram(binary_image, nbins=256))
    if image_histogram.argmax() <= 150:
        binary_image = 255 - binary_image
    binary_image = np.where(binary_image > 0, 0, 1)
    binary_image = binary_image.astype(np.uint8)

    return binary_image


def feature_extraction_lpq(img, winSize=3, freqestim=2):

    # alpha in STFT approaches (for Gaussian derivative alpha=1)
    STFTalpha = 1/winSize
    # Sigma for STFT Gaussian window (applied if freqestim==2)
    sigmaS = (winSize-1)/4
    # Sigma for Gaussian derivative quadrature filters (applied if freqestim==3)
    sigmaA = 8/(winSize-1)

    # Compute descriptor responses only on part that have full neigborhood.
    # Use 'same' if all pixels are included (extrapolates np.image with zeros).
    convmode = 'valid'

    # Convert np.image to double
    img = np.float64(img)
    # Get radius from window size
    r = (winSize-1)/2
    # Form spatial coordinates in window
    x = np.arange(-r, r+1)[np.newaxis]

    u = np.arange(1, r+1)

    # STFT uniform window
    if freqestim == 1:
        #  Basic STFT filters
        w0 = np.ones_like(x)
        w1 = np.exp(-2*np.pi*x*STFTalpha*1j)
        w2 = np.conj(w1)
    # STFT Gaussian window (equals to Gaussian quadrature filter pair)
    elif freqestim == 2:
        # Basic STFT filters
        w0 = (x * 0 + 1)
        w1 = np.exp(-2*np.pi*x*STFTalpha*1j)
        w2 = np.conj(w1)
        # Gaussian window
        gs = np.exp(- 0.5 * (x / sigmaS) ** 2) / \
            (np.multiply(np.sqrt(2 * np.pi), sigmaS))
        # Windowed filters
        w0 = np.multiply(gs, w0)
        w1 = np.multiply(gs, w1)
        w2 = np.multiply(gs, w2)
        # Normalize to zero mean
        w1 = w1 - np.mean(w1)
        w2 = w2 - np.mean(w2)
    # Gaussian derivative quadrature filter pair
    elif freqestim == 3:

        G0 = np.exp(- x ** 2 * (np.sqrt(2) * sigmaA) ** 2)

        G1_zeros = np.concatenate((np.zeros(len(u)), np.asarray([0])))
        G1 = np.asarray(
            [np.concatenate((G1_zeros, u * np.exp(- u ** 2 * sigmaA ** 2)))])

        # Normalize to avoid small numerical values (do not change the phase response we use)
        G0 = G0 / np.max(np.abs(G0))
        G1 = G1 / np.max(np.abs(G1))

        # Compute spatial domain correspondences of the filters
        w0 = np.real(np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(G0))))
        w1 = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(G1)))
        w2 = np.conj(w1)

        # Normalize to avoid small numerical values (do not change the phase response we use)
        w0 = w0 / \
            np.max(
                np.abs(np.array([np.real(np.max(w0)), np.imag(np.max(w0))])))
        w1 = w1 / \
            np.max(
                np.abs(np.array([np.real(np.max(w1)), np.imag(np.max(w1))])))
        w2 = w2 / \
            np.max(
                np.abs(np.array([np.real(np.max(w2)), np.imag(np.max(w2))])))

    # Run filters to compute the frequency response in the four points. Store np.real and np.imaginary parts separately
    # Run first filter
    filterResp1 = convolve2d(convolve2d(img, w0.T, convmode), w1, convmode)
    filterResp2 = convolve2d(convolve2d(img, w1.T, convmode), w0, convmode)
    filterResp3 = convolve2d(convolve2d(img, w1.T, convmode), w1, convmode)
    filterResp4 = convolve2d(convolve2d(img, w1.T, convmode), w2, convmode)

    # Initilize frequency domain matrix for four frequency coordinates (np.real and np.imaginary parts for each frequency).
    freqResp = np.dstack([filterResp1.real, filterResp1.imag,
                          filterResp2.real, filterResp2.imag,
                          filterResp3.real, filterResp3.imag,
                          filterResp4.real, filterResp4.imag])

    # Perform quantization and compute LPQ codewords
    inds = np.arange(freqResp.shape[2])[np.newaxis, np.newaxis, :]
    LPQdesc = ((freqResp > 0)*(2**inds)).sum(2)

    LPQdesc = np.histogram(LPQdesc.flatten(), range(256))[0]

    LPQdesc = LPQdesc/LPQdesc.sum()

    return LPQdesc


def get_training_features(base_directory):
    '''
    DESCRIPTION:
    Get array of images at specific directory. Then preprocess, extract features.
    RETURN:
    (X, Y)
    Array of features of training data, array of true classification of training data
    X: (M, N)
    Y: (M, 1)
    '''

    X = []
    Y = []
    for folder_num in range(1, 10):
        folder_path = base_directory + "\\" + str(folder_num)
        filenames = os.listdir(folder_path)

        for fn in filenames:
            path = os.path.join(folder_path, fn)
            img = io.imread(path)
            preprocessed_image = preprocessing_image(img)
            image_features = feature_extraction_lpq(
                preprocessed_image, winSize=3, freqestim=2)
            X.append(image_features)
            Y.append(folder_num)

    return np.asarray(X), np.asarray(Y)


def train(training_directory, output_directory, verbose=None):
    '''
    Takes training_directory and create model then saves it in the output_directory named as model.sav
    '''
    # CHOOSEN CLASSIFIER
    clf = svm.SVC(kernel='poly', C=1, gamma=53, probability=True)

    # PREPROCESSING & FEATURE EXTRACTION
    if verbose != None and verbose > 0:
        print("--> Starting preprocessing and feature extraction...")
    start = time.time()
    X, Y = get_training_features(training_directory)
    end = time.time()
    preprocessing_extraction_time = end - start
    if verbose != None and verbose > 1:
        print("--> Finished preprocessing and feature extraction on training directory in",
              preprocessing_extraction_time, "seconds")
    elif verbose != None and verbose > 0:
        print("--> Finished preprocessing and feature extraction on training directory.")

    # FITTING TRAINING FEATURES
    if verbose != None and verbose > 0:
        print("--> Starting fitting training data")
    start = time.time()
    clf.fit(X, Y)
    end = time.time()
    fitting_time = end - start
    if verbose != None and verbose > 1:
        print("--> Finished fitting training data in", fitting_time, "seconds")
    elif verbose != None and verbose > 0:
        print("--> Finished fitting training data.")

    pickle.dump(clf, open(output_directory + "\\model.sav", 'wb'))


if __name__ == '__main__':
    fire.Fire(train)