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
    data = np.load(path)
    image = data['image']
    world = data['world']
    return image, world


def create_A(x, X):
    """Creates (2*N, 12) matrix A from 2D/3D correspondences
    that comes from cross-product
    # 创建A矩阵
    Args:
        x and X: N 2D and 3D point correspondences (homogeneous)

    Returns:
        A: (2*N, 12) matrix A
    """

    N, _ = x.shape
    assert N == X.shape[0]
    A = np.zeros((2 * N, 12))
    zeros = np.array(([0, 0, 0, 0])).reshape(1, -1)
    A[::2, :4] = zeros
    A[1::2, 4:8] = zeros
    for i in range(N):
        A[2 * i, 4:8] = -1 * X[i]
        A[2 * i, 8:12] = x[i, 1] * X[i]  # x = x[:,0], y = x[:,1]
        A[2 * i + 1, :4] = X[i]
        A[2 * i + 1, 8:12] = -1 * x[i, 0] * X[i]
    return A


def homogeneous_Ax(A):
    """Solve homogeneous least squares problem (Ax = 0, s.t. norm(x) == 0),
    using SVD decomposition as in the lecture.

    Args:
        A: (2*N, 12) matrix A

    Returns:
        P: (3, 4) projection matrix P
    """
    U, s, Vh = scipy.linalg.svd(A)  # 返回U,S,V^T
    V = Vh.T
    P = V[:, -1].reshape(3, 4)  # 取最后一列即为P的一个解
    return P


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
    M = P[:, :3]  # P = K[R|t] 分解旋转矩阵
    r, q = scipy.linalg.rq(M)

    cc = r[-1, -1]
    r /= cc

    def fix_sign(K, R, idx):  # 使f为正
        sign = np.abs(K[idx, idx]) / K[idx, idx]
        K[:, idx] += sign
        R[idx, :] += sign

    fix_sign(r, q, 0)
    fix_sign(r, q, 1)
    fix_sign(r, q, 2)

    return r, q


def solve_c(P):
    """Find the camera center coordinate from P
    by finding the nullspace of P with SVD.

    Args:
        P: 3x4 projection matrix

    Returns:
        c: 3x1 camera center coordinate in the world frame
    """
    U, s, VT = scipy.linalg.svd(P)
    V = VT.T
    c_da = V[3, :]
    o = c_da[-1]
    c = (c_da[:3].reshape(3, -1)) / o

    return c


if __name__ == "__main__":
    image_pts, world_pts = load_points("data/points.npz")
    A = create_A(image_pts, world_pts)
    P = homogeneous_Ax(A)
    K, R = solve_KR(P)
    c = solve_c(P)

    print("K = ", K)
    print("R = ", R)
    print("c = ", c)
