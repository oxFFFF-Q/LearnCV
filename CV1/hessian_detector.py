import numpy as np
from scipy.ndimage import convolve, maximum_filter

'''
用Hessian检测兴趣点，采用极大值抑制，最后输出兴趣点xy坐标
'''


def gauss2d(sigma, fsize):
    """ Create a 2D Gaussian filter

    Args:
        sigma: width of the Gaussian filter
        fsize: (w, h) dimensions of the filter
    Returns:
        *normalized* Gaussian filter as (h, w) np.array
    """
    m, n = fsize
    x = np.arange(-m / 2 + 0.5, m / 2)
    y = np.arange(-n / 2 + 0.5, n / 2)
    xx, yy = np.meshgrid(x, y, sparse=True)
    g = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    return g / np.sum(g)


def derivative_filters():
    """ Create derivative filters for x and y direction

    Returns:
        fx: derivative filter in x direction
        fy: derivative filter in y direction
    """
    fx = np.array([[0.5, 0, -0.5]])
    fy = fx.transpose()
    return fx, fy


def compute_hessian(img, gauss, fx, fy):
    """ Compute elements of the Hessian matrix

    Args:
        img:
        gauss: Gaussian filter
        fx: derivative filter in x direction
        fy: derivative filter in y direction

    Returns:
        I_xx: (h, w) np.array of 2nd derivatives in x direction
        I_yy: (h, w) np.array of 2nd derivatives in y direction
        I_xy: (h, w) np.array of 2nd derivatives in x-y direction
    """

    img_G = convolve(img,gauss,mode = 'mirror')
    I_x = convolve(img_G,fx,mode='mirror')
    I_y = convolve(img_G,fy,mode = 'mirror')
    I_xx = convolve(I_x,fx,mode = 'mirror')           # 注意二阶导
    I_yy = convolve(I_y,fy,mode = 'mirror')
    I_xy = convolve(I_x,fy,mode='mirror')
    return I_xx,I_yy,I_xy


def compute_criterion(I_xx, I_yy, I_xy, sigma):
    """ Compute criterion function

    Args:
        I_xx: (h, w) np.array of 2nd derivatives in x direction
        I_yy: (h, w) np.array of 2nd derivatives in y direction
        I_xy: (h, w) np.array of 2nd derivatives in x-y direction
        sigma: scaling factor

    Returns:
        criterion: (h, w) np.array of scaled determinant of Hessian matrix
    """
    det = (I_xx * I_yy) - (I_xy * I_xy)
    p = sigma ** 4
    return p*det


def nonmaxsuppression(criterion, threshold):
    """ Apply non-maximum suppression to criterion values
        and return Hessian interest points
        去掉冗余点
        Args:
            criterion: (h, w) np.array of criterion function values
            threshold: criterion threshold
        Returns:
            rows: (n,) np.array with y-positions of interest points
            cols: (n,) np.array with x-positions of interest points
    """
    if len(criterion)==0:                              # 防止报错
        return np.empty((0, 0)),np.empty((0, 0))

    max = maximum_filter(criterion, (5, 5))     # 返回指定窗口内最大值组成的相同大小矩阵
    r, c = np.nonzero(criterion >= max)
    a = np.full(criterion.shape, threshold)
    a[r, c] = criterion[r, c]
    rows, cols = np.nonzero(a > threshold)


    return rows,cols
