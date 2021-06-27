"""
This file is gonna demonstrate 2 of the powerful algorithm used for solving sparse matrix
vector equation.

"""

import numpy as np
norm = np.linalg.norm
rand = np.random.rand
import scipy.sparse as scipy_sparse

class TheSolution:

    def __init__(this):
        this.NumberItr = 0
        this.X = []
        this.flag = 0
        this.RNorm = []
    def append(this, x):
        this.X.append(x)

def SteepestDescend(A, b, x0=None, tol=1e-4, maxitr=30000):
    """
        Warning: only works for symmetric matrix.
    :param A:
    :param b:
    :param x0:
    :param tol:
    :param maxitr:
    :return:
    """
    assert A.ndim == 2, "Matrix A has to be two dimensional"
    assert b.ndim == 2, "Matrix b has to be two dimensional"
    m, n = A.shape
    assert m == n, "Matrix Gonna be squared"
    assert tol > 0, "Tolerance is a non negative number."
    x = x0 if x0 is not None else np.ones((n, 1))
    r = b - A@x
    Sol = TheSolution()
    Sol.append(x)
    Itr = 0
    while norm(r, np.inf) > tol and Itr < maxitr:
        Itr += 1
        alpha = norm(r)**2/np.sum(r*A@r)
        x += alpha*r
        r = b - A@x
        Sol.append(x)
    if Itr == maxitr: Sol.flag = 1
    return Sol


def ConjugateGradient(A, b, maxitr=None, x0=None, tol=1e-4):
    """
        Warning: Only works for symmetric matrix
    :param A:
    :param b:
    :param x0:
    :param tol:
    :return:
    """
    assert A.ndim == 2, "Matrix A has to be two dimensional"
    m, n = A.shape
    assert m == n, "Matrix Gonna be squared"
    assert tol > 0, "Tolerance is a non negative number."
    maxitr = 2*m if maxitr is None else maxitr
    x = x0 if x0 is not None else np.ones((n, 1))
    d = r = b - A @ x
    Sol = TheSolution()
    Sol.append(x)
    Itr = 0
    while norm(r, np.inf) > tol and Itr < maxitr:
        alpha = np.sum(r*r)/np.sum(d*A@d)
        x += alpha*d
        if Itr % 5:
            rnew = b - A@x
        else:
            rnew = r - alpha*A@d
        beta = norm(rnew)**2/norm(r)**2
        d = rnew + beta * d
        r = rnew
        Sol.append(x)
        Itr += 1

    if Itr == maxitr: Sol.flag = 1
    return Sol



def main():
    def SimpleTest(solver:callable, N=30):
        A, b = rand(N, N), rand(N, 1)
        M = A.T@A
        c = A.T@b
        Sol = solver(M, c)
        x = Sol.X[-1]
        print(f"The ||A.T@A - A.T@b|| is: {norm(c - M@x, 1)}")
        print(f"The ||A@x - b|| is: {norm(A@x - b, 1)}")
        print(f"The flag is: {Sol.flag}")

    print("======== Steepest Descend =======")
    for __ in range(10):
        SimpleTest(SteepestDescend)
    print("======= Conjugate Gradient ======")
    for __ in range(10):
        SimpleTest(ConjugateGradient)


if __name__ == "__main__":
    import sys
    import os
    print(f"curdir: {os.curdir}")
    print(f"cwd: {os.getcwd()}")
    print(f"exec: {sys.executable}")
    main()

