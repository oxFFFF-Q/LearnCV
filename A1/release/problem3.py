import numpy as np
import scipy.linalg


def load_points(path):
    '''
    Load points from path pointing to a numpy binary file (.npy). 
    Image points are saved in 'image'
    Object points are saved in 'world'

    Returns:
        image: A Nx2 array of 2D points form image coordinate 
        world: A N*3 array of 3D points form world coordinate
    '''
    #
    # Your Code Here
    #

def create_A(x, X):
    """Creates (2*N, 12) matrix A from 2D/3D correspondences
    that comes from cross-product
    
    Args:
        x and X: N 2D and 3D point correspondences (homogeneous)
        
    Returns:
        A: (2*N, 12) matrix A
    """

    N, _ = x.shape
    assert N == X.shape[0]
    #
    # Your Code Here
    #

def homogeneous_Ax(A):
    """Solve homogeneous least squares problem (Ax = 0, s.t. norm(x) == 0),
    using SVD decomposition as in the lecture.

    Args:
        A: (2*N, 12) matrix A
    
    Returns:
        P: (3, 4) projection matrix P
    """
    #
    # Your Code Here
    #


def solve_KR(P):
    """Using th RQ-decomposition find K and R 
    from the projection matrix P.
    Hint 1: you might find scipy.linalg useful here.
    Hint 2: recall that K has 1 in the the bottom right corner.
    Hint 3: RQ decomposition is not unique (up to a column sign).
    Ensure positive element in K by inverting the sign in K columns 
    and doing so correspondingly in R.

    Args:
        P: 3x4 projection matrix.
    
    Returns:
        K: 3x3 matrix with intrinsics
        R: 3x3 rotation matrix 
    """
    #
    # Your Code Here
    #


def solve_c(P):
    """Find the camera center coordinate from P
    by finding the nullspace of P with SVD.

    Args:
        P: 3x4 projection matrix
    
    Returns:
        c: 3x1 camera center coordinate in the world frame
    """
    #
    # Your Code Here
    #
