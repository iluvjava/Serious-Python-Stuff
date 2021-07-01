"""
Conjugate Gradient can be used to reverse the affect of a diffusion stencil.

Investigate: Is the sharpen kernel really the inverse of whatever that does the
blurring?

"""
from Conjugate_Gradient.core import cg_core
import scipy.sparse as scipy_sparse
import numpy as np
import matplotlib.pyplot as plt

def Laplacian2D(m, n, delta=1):
    """
        Get the matrix that take the laplacian on an flattened m by n grid.
        Boundary Conditions:
            * Periodic
        Assumption:
            * Image is vectorized by stack m rows into the vector. first n
            element is the first row.
            * Standard matrix indexing and horizontal is x, vertical is y.

    :param m:
        The height of the image
    :param n:
        The width of the image.
    """
    def MakeDoubleDiff(k):
        ddx  = np.diag([1 for __ in range(k - 1)], 1)
        ddx += np.diag([1 for __ in range(k - 1)], -1)
        ddx += np.diag([-2 for __ in range(k)], 0)
        ddx[0, -1] = ddx[-1, 0] = 1
        ddx  = scipy_sparse.csr_matrix(ddx)
        return ddx

    ddx = MakeDoubleDiff(n)
    ddy = MakeDoubleDiff(m)
    Ix = scipy_sparse.eye(n)
    Iy = scipy_sparse.eye(m)
    L = scipy_sparse.kron(Iy, ddx) + scipy_sparse.kron(ddy, Ix)
    return L/delta**2


def main():
    L = Laplacian2D(3, 4)
    print(L.toarray())

    def SinCosineTest():
        delta = 0.01
        x = np.arange(-10, 10, delta)[:, np.newaxis]
        y = np.arange(-8, 8, delta)[:, np.newaxis]
        xgrid = x.T + np.zeros((len(y), len(x)));
        ygrid = y + np.zeros((len(y), len(x)))
        x = xgrid; y = ygrid
        z = np.sin(np.pi*y) + np.cos(np.pi*x)
        # Apply Laplacian
        L = Laplacian2D(z.shape[0], z.shape[1], delta)
        dz = L@(z.reshape(-1)[:, np.newaxis])
        plt.matshow(z); plt.title("z")
        plt.show()
        plt.matshow(dz.reshape(z.shape)); plt.title("dz")
        plt.show()
        # Reverse Laplacian
        Sol = cg_core.ConjugateGradient(L, dz)
        recovered = Sol.X[-1].reshape(z.shape)
        plt.matshow(recovered); plt.title("recovered z")
        plt.show()
        print(f"Number of Itr for GC: {Sol.NumberItr}")

    SinCosineTest()



if __name__ == "__main__":
    import os
    import sys
    print(f"Runner: {sys.executable}")
    print(f"wdir: {os.curdir}")
    print(f"cwd: {os.getcwd()}")
    main()