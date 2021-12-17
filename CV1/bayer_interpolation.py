import numpy as np
from scipy.ndimage import convolve


def loaddata(path):
    """ Load bayerdata from file
    取图像
    Args:
        Path of the .npy file
    Returns:
        Bayer data as numpy array (H,W)
    """

    #
    # You code here
    #
    bayer = np.load(path)
    return bayer


def separatechannels(bayerdata):
    """ Separate bayer data into RGB channels so that
    each color channel retains only the respective
    values given by the bayer pattern and missing values
    are filled with zero
    分离RGB
    Args:
        Numpy array containing bayer data (H,W)
    Returns:
        red, green, and blue channel as numpy array (H,W)
    """

    #
    # You code here
    #
    # make the shape of r,g,b matrix same as bayerdata.
    # the shape of bayerdata is (512, 448)

    r = np.zeros_like(bayerdata)
    g = np.zeros_like(bayerdata)
    b = np.zeros_like(bayerdata)

    # for red value: every two row from first row and every two column from second column.
    r[::2, 1::2] = bayerdata[::2, 1::2]
    # for blue value: every two row from secomd row and every two column from first column.
    b[1::2, ::2] = bayerdata[1::2, ::2]
    # for green value: is the combination of rea and blue value.
    g[::2, ::2] = bayerdata[::2, ::2]
    g[1::2, 1::2] = bayerdata[1::2, 1::2]

    return r, g, b

def assembleimage(r, g, b):
    """ Assemble separate channels into image
    拼接图像
    Args:
        red, green, blue color channels as numpy array (H,W)
    Returns:
        Image as numpy array (H,W,3)
    """

    #
    # You code here
    #
    img = np.stack((r, g, b), axis= -1) # axis: 0,1,2,3,…是从外开始剥，-n,-n+1,…,-3,-2,-1是从里开始剥
    return img


def interpolate(r, g, b):
    """ Interpolate missing values in the bayer pattern
    by using bilinear interpolation
    #  通过插值补全缺失的两个色彩
    Args:
        red, green, blue color channels as numpy array (H,W)
    Returns:
        Interpolated image as numpy array (H,W,3)
    """

    #
    # You code here
    #
    # scipy.ndimage.convolve(input, weights, output=None, mode='reflect', cval=0.0, origin=0)
    '''
            rb各四分之一
            g占array的一半
            g的九宫格里面 四角还是g 这个不能算。
            rb九宫格不会再有同色
        '''
    gfilter = np.array([
        [0, 1 / 4, 0],
        [1 / 4, 1, 1 / 4],
        [0, 1 / 4, 0]
    ])

    rbfilter = np.array([
        [1 / 4, 1 / 2, 1 / 4],
        [1 / 2, 1, 1 / 2],
        [1 / 4, 1 / 2, 1 / 4]
    ])

    r = convolve(r, rbfilter, mode="mirror")
    g = convolve(g, gfilter, mode="mirror")
    b = convolve(b, rbfilter, mode="mirror")
    img = assembleimage(r, g, b)
    return img
