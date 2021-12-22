import numpy as np
from scipy.ndimage import convolve
import matplotlib.pyplot as plt


def gauss2d(sigma, fsize):
    """
    Args:
      sigma: width of the Gaussian filter
      fsize: dimensions of the filter

    Returns:
      g: *normalized* Gaussian filter
    """

    #
    m, n = (fsize, fsize)
    x = np.arange(-m / 2 + 0.5, m / 2)
    y = np.arange(-n / 2 + 0.5, n / 2)
    xx, yy = np.meshgrid(x, y, sparse=True)
    g = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    return g / np.sum(g)
    #


def createfilters():
    """
    Returns:
      fx, fy: filters as described in the problem assignment
    """

    #
    g = gauss2d(0.9, 3)
    derxy = (1 / 2) * np.array([1, 0, -1])
    fx = g * derxy
    fy = np.outer((derxy.T), (g.T))  # 外积，对应相乘
    fx_ = np.array([[0.5, 0, -0.5]]) * g
    fy_ = fx.transpose() * g  # 转置

    return fx_, fy_


def filterimage(I, fx, fy):
    """ Filter the image with the filters fx, fy.
    You may use the ndimage.convolve scipy-function.

    Args:
      I: a (H,W) numpy array storing image data
      fx, fy: filters

    Returns:
      Ix, Iy: images filtered by fx and fy respectively
    """

    #
    Ix = convolve(I, fx, mode='constant', cval=0.0)
    Iy = convolve(I, fy, mode='constant', cval=0.0)
    return Ix, Iy
    #



if __name__ == "__main__":
    # load image
    img = plt.imread("data/a1p4.png")

    # create filters
    fx, fy = createfilters()

    # filter image
    imgx, imgy = filterimage(img, fx, fy)

    # show filter results
    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax1.imshow(imgx, "gray", interpolation="none")
    ax1.set_title("x derivative")
    ax1.axis("off")
    ax2 = plt.subplot(122)
    ax2.imshow(imgy, "gray", interpolation="none")
    ax2.set_title("y derivative")
    ax2.axis("off")


