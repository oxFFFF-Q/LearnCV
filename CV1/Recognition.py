import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from PIL import Image
import glob
import math
from numpy import linalg as LA
from scipy.ndimage import convolve


def load_data(path):
    '''
    Load data from folder data, face images are in the folder facial_images, face features are in the folder facial_features.


    Args:
        path: path of folder data

    Returns:
        imgs: list of face images as numpy arrays
        feats: list of facial features as numpy arrays
    '''

    imgs = []
    feats = []

    path_ima = os.path.join(path, 'facial_images')
    path_fea = os.path.join(path, 'facial_features')
    list_ima = [os.path.join(path_ima, f) for f in os.listdir(path_ima)]
    list_fea = [os.path.join(path_fea, f) for f in os.listdir(path_fea)]
    imgs = [np.array(Image.open(f)) for f in list_ima]
    feats = [np.array(Image.open(f)) for f in list_fea]
    return imgs, feats


def gaussian_kernel(fsize, sigma):
    '''
    Define a Gaussian kernel

    Args:
        fsize: kernel size
        sigma: sigma of Gaussian kernel

    Returns:
        The Gaussian kernel
    '''

    m, n = (fsize,fsize)
    # Just make matrixes
    x = np.arange(-m/2+0.5,m/2)
    y = np.arange(-n/2+0.5,n/2)
    # https://numpy.org/doc/stable/reference/generated/numpy.meshgrid.html
    # Expand rank
    X,Y = np.meshgrid(x, y, sparse = True)
    g = np.exp(-(X**2 + Y**2)/(2*sigma**2))
    # Normalization
    return g/np.sum(g)


def downsample_x2(x, factor=2):
    '''
    Downsampling an image by a factor of 2

    Args:
        x: image as numpy array (H * W)

    Returns:
        downsampled image as numpy array (H/2 * W/2)

    '''

    #
    # TODO
    #
    d_size1 = (x.shape[0] + 1) // factor
    d_size2 = (x.shape[1] + 1) // factor
    downsample = np.zeros((d_size1, d_size2))
    sample_matrix = np.full(x.shape, True, dtype=bool)
    sample_matrix[:, 1::2] = False
    sample_matrix[1::2, ::2] = False
    mitZerosSample = x * sample_matrix
    t = mitZerosSample[mitZerosSample != 0]
    downsample = t.reshape(d_size1, d_size2)
    return downsample



def gaussian_pyramid(img, nlevels, fsize, sigma):
    '''
    A Gaussian pyramid is constructed by combining a Gaussian kernel and downsampling.
    Tips: use scipy.signal.convolve2d for filtering image.

    Args:
        img: face image as numpy array (H * W)
        nlevels: number of levels of Gaussian pyramid, in this assignment we will use 3 levels
        fsize: Gaussian kernel size, in this assignment we will define 5
        sigma: sigma of Gaussian kernel, in this assignment we will define 1.4

    Returns:
        GP: list of Gaussian downsampled images, it should be 3 * H * W
    '''
    GP = []
    k = gaussian_kernel(fsize, sigma)
    g_in = img
    GP.append(g_in)
    for i in range(nlevels - 1):
        g_out = convolve2d(g_in, k, boundary='symm')
        g_in = downsample_x2(g_out)
        GP.append(g_in)
    return GP


def template_distance(v1, v2):
    '''
    Calculates the distance between the two vectors to find a match.
    Browse the course slides for distance measurement methods to implement this function.
    Tips:
        - Before doing this, let's take a look at the multiple choice questions that follow.
        - You may need to implement these distance measurement methods to compare which is better.

    Args:
        v1: vector 1
        v2: vector 2

    Returns:
        Distance
    '''
    # dot product
    '''
    dot_product = np.dot(v1,v2)
    norm_v1 = LA.norm(v1,ord = 2)
    norm_v2 = LA.norm(v2,ord = 2)
    cos_v1v2 = dot_product/(norm_v1*norm_v2)
    distance = np.arccos(cos_v1v2)
    '''
    # SSD
    distance = np.sqrt(v1 - v2).sum()
    return distance


def sliding_window(img, feat, step=1):
    '''
    A sliding window for matching features to windows with SSDs. When a match is found it returns to its location.

    Args:
        img: face image as numpy array (H * W)
        feat: facial feature as numpy array (H * W)
        step: stride size to move the window, default is 1
    Returns:
        min_score: distance between feat and window
    '''
    min_score = None
    win = np.zeros(feat.shape)
    winH = win.shape[0]
    winW = win.shape[1]
    l_sub = []
    scores = []
    score = 0
    index = 0
    s = step - 1

    for i in range(img.shape[0] - winH):
        for j in range(img.shape[1] - winW):
            win = img[i + s:i + winH, j + s:j + winW]
            l_sub.append(win)
            index = index + 1
            score = template_distance(win, feat)
            scores.append(score)
    try:
        min_score = min(scores)
    except ValueError:  # len(scores) = 0,when the feats size greater than win shape
        pass
    return min_score


class Distance(object):
    # choice of the method
    METHODS = {1: 'Dot Product', 2: 'SSD Matching'}

    # choice of reasoning
    REASONING = {
        1: 'it is more computationally efficient',
        2: 'it is less sensitive to changes in brightness.',
        3: 'it is more robust to additive Gaussian noise',
        4: 'it can be implemented with convolution',
        5: 'All of the above are correct.'
    }

    def answer(self):
        '''Provide your answer in the return value.
        This function returns one tuple:
            - the first integer is the choice of the method you will use in your implementation of distance.
            - the following integers provide the reasoning for your choice.
        Note that you have to implement your choice in function template_distance

        For example (made up):
            (1, 1) means
            'I will use Dot Product because it is more computationally efficient.'
        '''

        return (None, None)  # TODO


def find_matching_with_scale(imgs, feats):
    '''
    Find face images and facial features that match the scales

    Args:
        imgs: list of face images as numpy arrays
        feats: list of facial features as numpy arrays
    Returns:
        match: all the found face images and facial features that match the scales: N * (score, g_im, feat)
        score: minimum score between face image and facial feature
        g_im: face image with corresponding scale
        feat: facial feature
    '''
    match = []
    (score, g_im, feat) = (None, None, None)
    for fes in feats:
        for img in imgs:
            GP_img = gaussian_pyramid(img, nlevels=3, fsize=5, sigma=1.4)
            for i in range(len(GP_img)):
                if i == 0:  # original picture
                    (score, g_im, feat) = sliding_window(GP_img[i], fes), GP_img[i], fes
                else:
                    (score_, g_im_, feat_) = sliding_window(GP_img[i], fes), GP_img[i], fes
                    if score != None and score_ != None:
                        if score_ < score:
                            (score, g_im, feat) = (score_, g_im_, feat_)
            match.append((score, g_im, feat))
    return match