"""
This file is gonna demonstrate 2 of the powerful algorithm used for solving sparse matrix
vector equation.

"""

import numpy as np
norm = np.linalg.norm
rand = np.random.rand
import warnings


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
    return AbstractConjugateGradient(lambda x: A@x, b, maxitr=maxitr, x0=x, tol=tol)


def AbstractConjugateGradient(
                        A:callable,
                        b:np.ndarray,
                        x0:np.ndarray,
                        maxitr=None,
                        tol=1e-4,
                        verbose:bool=False):
    """
        This method is just Conjugate Gradient but the matrix vector process has been
        abstracted out.
        Assumption:
            the abstracted A is still a square matrix.
    :param A:
        A callable that can perform linear transformation on some tensor
    :param b:
        This is the abstract vector on the rhs of the linear system.
    :param x0:
        This is an initial guess, and it has to be provided
    :param maxitr:
        maximal number of iterations
    :tol tolerance:
        The tolerance allowed to terminate the iterations.
    """
    assert tol > 0, "Tolerance is a non negative number."
    n = x0.shape[0]
    maxitr = 2 * n if maxitr is None else maxitr
    x = x0 if x0 is not None else np.ones((n, 1))
    d = r = b - A(x)
    Sol = TheSolution()
    Sol.append(x)
    Sol.NumberItr = 0
    while np.max(np.abs(r)) > tol and Sol.NumberItr < maxitr:
        alpha = np.sum(r * r) / np.sum(d * A(d))
        x += alpha * d
        if Sol.NumberItr % 5:
            rnew = b - A(x)
        else:
            rnew = r - alpha * A(d)
        beta = np.sum(rnew*rnew) / np.sum(r*r)
        d = rnew + beta * d
        r = rnew
        Sol.append(x)
        Sol.NumberItr += 1
        if verbose:
            print(f"Itr: {Sol.NumberItr}; Residual Inf Norm: {np.max(np.abs(r))}")
    if Sol.NumberItr == maxitr: Sol.flag = 1
    return Sol


def AbstractCGGenerate(A:callable, b, x0, maxitr:int=100):
    """
        A generator for the Conjugate Gradient method, so whoever uses it
        has the pleasure to collect all the quantities at runtime.
    :param A:
    :param b:
    :param x0:
    :return:
        itr, r, x
    """
    x = x0
    d = r = b - A(x)
    Itr = 0
    while Itr < maxitr:
        alpha = np.sum(r * r) / np.sum(d * A(d))
        if alpha < 0:
            warnings.warn(f"CG negative energy norm! Matrix Not symmetric and SPD! Convergence might fail.")
        x += alpha * d
        # rnew = r - alpha * A(d)
        rnew = b - A(x)
        beta = np.sum(rnew * rnew) / np.sum(r * r)
        d = rnew + beta * d
        r = rnew
        Itr += 1
        yield Itr, r, x


class CGAnalyzer:
    """
        See your Conjugate Gradient Algorithm at runtime and collect data from
        it while running.

    """
    def __init__(this, operator:callable, b:np.ndarray, initialGuess:np.ndarray):
        this._X0 = initialGuess
        this._RHS = b
        this._A = operator
        this._ResidualsNorm = []

        r = this.RHS - this._A(this._X0)
        r = np.max(np.abs(r))

        this._ResidualsNorm.append(r)
        this._BestSolution = None
        this._BestSolutionNorm = float("inf")
        this.Itr = 0

    @property
    def X0(this):
        return this._X0.copy()

    @X0.setter
    def X0(this, x0):
        type(this._A(x0)) is np.ndarray
        this._X0 = x0

    @property
    def RHS(this):
        return this._RHS.copy()

    @property
    def BestSolution(this):
        if this._BestSolution is None:
            return this.X0
        return this._BestSolution.copy()

    @property
    def ResidualNorm(this):
        return this._ResidualsNorm

    @RHS.setter
    def RHS(this, b):
        this._RHS = b
    
    def Generate(this, maxitr:int=100):
        """
            Get the residual (inf norm), and x for each iteration (In that order).
        :param maxitr:
        :return:
        """

        for Itr, R, X in AbstractCGGenerate(this._A,
                                            this._RHS,
                                            this._X0,
                                            maxitr=maxitr):
            this.Itr = Itr
            Residual = np.max(np.abs(R))
            this._ResidualsNorm.append(Residual)
            if Residual < this._BestSolutionNorm:
                this._BestSolution = X
                this._BestSolutionNorm = Residual
            yield X, Residual

def main():
    def SimpleTest(solver:callable, N=30):
        A, b = rand(N, N), rand(N, 1)
        M = A.T@A
        c = A.T@b
        Sol = solver(M, c)
        x = Sol.X[-1]
        print(f"The ||A.T@Ax - A.T@b|| is: {norm(c - M@x, 1)}")
        print(f"The ||A@x - b|| is: {norm(A@x - b, 1)}")
        print(f"The flag is: {Sol.flag}")
    print("======== Steepest Descend =======")
    for __ in range(10):
        SimpleTest(SteepestDescend)
    print("======= Conjugate Gradient ======")
    for __ in range(10):
        SimpleTest(ConjugateGradient)
    print("======= Analyizer Testing =======")
    def AnalyzerTesting():
        print("Non Symmetric")
        from tqdm import tqdm
        N = 128
        A, b = rand(N, N), rand(N, 1)
        Analyzer = CGAnalyzer(lambda x: A.T@A@x, A.T@b, A.T@b)
        for x, r in  tqdm(Analyzer.Generate(N*N)):
            Res =np.max(np.abs(A.T@A@x - A.T@b))
            if Res< 1e-5:
                print(Res)
                break
        from scipy.sparse.linalg import cg
        print("Scipy: ")
        x_new, flag = cg(A.T@A, A.T@b, tol=1e-5)
        print(f"||A.T@A@x - A.T@b|| = {np.max(np.abs(A.T@A@x_new - A.T@b))}; flag: {flag}")
    AnalyzerTesting()
    print("====== Analyzer Testing =======")
    from tqdm import tqdm
    import matplotlib.pyplot as plt
    N = 128
    A, b = rand(N, N), rand(N, 1)
    A = A.T@A
    CorrectX = np.linalg.pinv(A)@b
    Analyzer = CGAnalyzer(lambda x: A@x, b, np.ones(b.shape))
    EnergyNorm = []
    for x, r in  tqdm(Analyzer.Generate(N*N)):
        E = x - CorrectX
        EnergyNorm.append(np.sum(E*(A@E)))
        if r < 1e-5:
            break
    x = Analyzer.BestSolution
    print(f"error: {np.linalg.norm(x - CorrectX)}")
    print(f"||b - Ax_{{k}}|| = {np.linalg.norm(b - A@x)}")
    plt.plot(EnergyNorm)
    plt.title("$\\Vert e\\Vert_A $")
    plt.show()
    plt.plot(Analyzer.ResidualNorm)
    plt.show()


if __name__ == "__main__":
    import sys
    import os
    print(f"curdir: {os.curdir}")
    print(f"cwd: {os.getcwd()}")
    print(f"exec: {sys.executable}")
    main()