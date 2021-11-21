import math
import numpy as np
from scipy import ndimage


def gauss2d(sigma, fsize):
  """
  Args:
    sigma: width of the Gaussian filter
    fsize: dimensions of the filter

  Returns:
    g: *normalized* Gaussian filter
  """

  #
  kernel = np.zeros((fsize, fsize))
  center = fsize // 2
  if sigma <= 0:
    sigma = ((fsize - 1) * 0.5 - 1) * 0.3 + 0.8
  s = sigma ** 2
  sum_val = 0
  for i in range(fsize):
    for j in range(fsize):
      x, y = i - center, j - center
      kernel[i, j] = np.exp(-(x ** 2 + y ** 2) / 2 * s)
      sum_val += kernel[i, j]
  g = kernel / sum_val
  return g
  #


def createfilters():
  """
  Returns:
    fx, fy: filters as described in the problem assignment
  """

  #
  fx = np.array((0.5, 0, -0.5))
  fy = gauss2d(0.9, 3)[:, 0]
  return  fx, fy
  #



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
  Ix = ndimage.convolve(fx, I)
  Iy = ndimage.convolve(fy, I)
  return Ix, Iy
  #


def detectedges(Ix, Iy, thr):
  """ Detects edges by applying a threshold on the image gradient magnitude.

  Args:
    Ix, Iy: filtered images
    thr: the threshold value

  Returns:
    edges: (H,W) array that contains the magnitude of the image gradient at edges and 0 otherwise
  """

  #
  # You code here
  #


def nonmaxsupp(edges, Ix, Iy):
  """ Performs non-maximum suppression on an edge map.

  Args:
    edges: edge map containing the magnitude of the image gradient at edges and 0 otherwise
    Ix, Iy: filtered images

  Returns:
    edges2: edge map where non-maximum edges are suppressed
  """

  # handle top-to-bottom edges: theta in [-90, -67.5] or (67.5, 90]

  # You code here

  # handle left-to-right edges: theta in (-22.5, 22.5]

  # You code here

  # handle bottomleft-to-topright edges: theta in (22.5, 67.5]

  # Your code here

  # handle topleft-to-bottomright edges: theta in [-67.5, -22.5]

  # Your code here
