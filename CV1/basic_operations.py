import numpy as np
import matplotlib.pyplot as plt

'''
绘制图像
numpy专用的npy(二进制格式) 存/取
图像水平/竖直翻转
图像水平/竖直拼接


'''

def display_image(img):
    """ Show an image with matplotlib:

    Args:
        Image as numpy array (H,W,3)
    """

    #
    # You code here
    #
    plt.figure()
    plt.imshow(img)
    plt.show()


def save_as_npy(path, img):
    """ Save the image array as a .npy file:

    Args:
        Image as numpy array (H,W,3)
    """

    #
    # You code here
    #
    np.save(path, img)


def load_npy(path):
    """ Load and return the .npy file:

    Args:
        Path of the .npy file
    Returns:
        Image as numpy array (H,W,3)
    """

    #
    # You code here
    #
    return np.load(path)     # np.load('x.npz')['n']  读取x.npz文件的z列

def mirror_horizontal(img):
    """ Create and return a horizontally mirrored image:

    Args:
        Loaded image as numpy array (H,W,3)

    Returns:
        A horizontally mirrored numpy array (H,W,3).
    """

    #
    # You code here
    #
    return np.fliplr(img) # np.fliplr(img): 左右翻转    np.flipud(img):上下翻转


def display_images(img1, img2):
    """ display the normal and the mirrored image in one plot:

    Args:
        Two image numpy arrays
    """

    #
    # You code here
    #
    img = np.hstack([img1, img2]) # np.vstack():竖直方向堆叠  np.hstack():水平方向堆叠
    plt.figure()
    plt.imshow(img)
    plt.show()