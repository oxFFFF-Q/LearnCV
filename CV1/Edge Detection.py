import math
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
  fx = np.array([[0.5, 0, -0.5]]) * g
  fy = fx.T  # 转置

  return fx, fy




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


def detectedges(Ix, Iy, thr):
  """ Detects edges by applying a threshold on the image gradient magnitude.

  Args:
    Ix, Iy: filtered images
    thr: the threshold value

  Returns:
    edges: (H,W) array that contains the magnitude of the image gradient at edges and 0 otherwise
  """

  #
  edges = np.sqrt(Ix ** 2 + Iy ** 2)
  edges[edges < thr] = 0
  print(edges)
  return edges
  #


def nonmaxsupp(edges, Ix, Iy):
  """ Performs non-maximum suppression on an edge map.

  Args:
    edges: edge map containing the magnitude of the image gradient at edges and 0 otherwise
    Ix, Iy: filtered images

  Returns:
    edges2: edge map where non-maximum edges are suppressed
  """
  # Store result here
  edges2 = np.ones_like(edges)

  # Expand by one pixel
  padedges = np.pad(edges, 1)

  # Edge orientation in [-90, 90]
  orientation = np.arctan(Iy / (Ix + 1e-24))

  # Store indices
  r = edges.shape[0]
  c = edges.shape[1]
  # Suppress non-maxima in 4 directions
  pi8 = math.pi / 8

  # handle top-to-bottom edges: theta in [-90, -67.5] or (67.5, 90]

  # You code here
  is_theta = np.logical_or((orientation <= -3 * pi8), (orientation > 3 * pi8))
  is_nonmax = (padedges[1:r + 1, 1:c + 1] < np.maximum(padedges[0:r, 1:c + 1], padedges[2:r + 2, 1:c + 1]))
  edges2[np.logical_and(is_theta, is_nonmax)] = 0

  # handle left-to-right edges: theta in (-22.5, 22.5]

  # You code here
  is_theta = np.logical_or((orientation > -pi8), (orientation <= pi8))      # 逻辑或
  is_nonmax = (padedges[1:r + 1, 1:c + 1] < np.maximum(padedges[1:r + 1, 0:c], padedges[1:r + 1, 2:c + 2]))
  edges2[np.logical_and(is_theta, is_nonmax)] = 0

  # handle bottomleft-to-topright edges: theta in (22.5, 67.5]

  # Your code here
  is_theta = np.logical_or((orientation > pi8), (orientation <= 3 * pi8))
  is_nonmax = (padedges[1:r + 1, 1:c + 1] < np.maximum(padedges[0:r, 0:c], padedges[2:r + 2, 2:c + 2]))
  edges2[np.logical_and(is_theta, is_nonmax)] = 0

  # handle topleft-to-bottomright edges: theta in [-67.5, -22.5]

  # Your code here
  is_theta = np.logical_or((orientation > -3 * pi8), (orientation <= -pi8))
  is_nonmax = (padedges[1:r + 1, 1:c + 1] < np.maximum(padedges[2:r + 2, 0:c], padedges[0:r, 2:c + 2]))
  edges2[np.logical_and(is_theta, is_nonmax)] = 0

  return edges2 * edges


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

    # show gradient magnitude
    plt.figure()
    plt.imshow(np.sqrt(imgx ** 2 + imgy ** 2), "gray", interpolation="none")
    plt.axis("off")
    plt.title("Derivative magnitude")

    # threshold derivative
    threshold = 1.0  # change threshold
    edges = detectedges(imgx, imgy, threshold)
    plt.figure()
    plt.imshow(edges > 0, "gray", interpolation="none")
    plt.axis("off")
    plt.title("Binary edges")

    # non maximum suppression
    edges2 = nonmaxsupp(edges, imgx, imgy)
    plt.figure()
    plt.imshow(edges2 > 0, "gray", interpolation="none")
    plt.axis("off")
    plt.title("Non-maximum suppression")

    plt.show()