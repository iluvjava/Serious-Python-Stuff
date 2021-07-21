"""
    Some functions that randomly generates convex problems in high dimension.
    1. problem parameters.
    2. first order drivative.
    3. proximal operator if non-smoothness is involved.
"""
__all__ = ["MatrixVector"]

import numpy as np
npl = np.linalg

class MatrixVector:
    """
        This is for benchmarking some of the optimization routines.
        min_x {||Ax - b||}
    """

    def __init__(this, A, b):
        """
            Note: Supports numpy tensor, torch tensor might not be supported.
        :param A:
            A matrix.
        :param b:
            A matrix or vector.
        """
        assert A.ndim == 2, "A needs to be a matrix"
        assert A.shape[0] == b.shape[0], "A's first dimension should match d"
        if b.ndim == 1:
            b = b[:, np.newaxis]
        this.A = A
        this.b = b
        this.Atrans = A.T
        this.AtransA = A.T@A

    def f(this, x):
        return npl.norm(this.A.dot(x) - this.b)**2

    def df(this, x):
        return this.Atrans.dot(this.A.dot(x) - this.b)






