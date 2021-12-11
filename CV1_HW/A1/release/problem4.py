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
  offset = fsize // 2
  g = np.zeros((3, 1), dtype=np.float)
  for i in range(-offset, -offset + fsize):
    g[i + offset] = np.exp(-(i ** 2) / (2 * sigma ** 2))
  g = g / g.sum()
  return g
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
  fy = np.outer((derxy.T), (g.T))
  return fx, fy
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
  Ix = ndimage.convolve(I, fx, mode='constant', cval=0.0)
  Iy = ndimage.convolve(I, fy, mode='constant', cval=0.0)
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
  edges = np.zeros(Ix.shape)
  sos = np.sqrt(Ix ** 2 + Iy ** 2)
  mask = (sos >= (thr * 0.1))
  edges[mask] = sos[mask]
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
  edges2 = np.zeros(Ix.shape)
  edpad = np.pad(edges, pad_width=1, mode='constant', constant_values=0)
  for i in range(1, 1 + edges.shape[0]):
    for j in range(1, 1 + edges.shape[1]):
      if  edges[i-1][j-1]==0:
        continue
      if Ix[i - 1][j - 1] == 0:
        Ix[i - 1][j - 1] += 0.00001
      theta = (np.arctan(Iy[i - 1][j - 1] / Ix[i - 1][j - 1]) / np.pi) * 180
  # handle top-to-bottom edges: theta in [-90, -67.5] or (67.5, 90]
      if (-90 <= theta <= -67.5) or (67.5 < theta <= 90):
        if edpad[i][j] > edpad[i - 1][j] and edpad[i][j] > edpad[i + 1][j]:
          edges2[i - 1][j - 1] = edpad[i][j]
  # handle left-to-right edges: theta in (-22.5, 22.5]
      elif -22.5 < theta <= 22.5:
        if edpad[i][j] > edpad[i][j - 1] and edpad[i][j] > edpad[i][j + 1]:
          edges2[i - 1][j - 1] = edpad[i][j]
  # handle bottomleft-to-topright edges: theta in (22.5, 67.5]
      elif (22.5 < theta <= 67.5):
        if edpad[i][j] > edpad[i + 1][j - 1] and edpad[i][j] > edpad[i - 1][j + 1]:
          edges2[i - 1][j - 1] = edpad[i][j]
  # handle topleft-to-bottomright edges: theta in [-67.5, -22.5]
      elif (-67.5 <= theta <= -22.5):
        if edpad[i][j] > edpad[i - 1][j - 1] and edpad[i][j] > edpad[i + 1][j + 1]:
          edges2[i - 1][j - 1] = edpad[i][j]
  return edges2