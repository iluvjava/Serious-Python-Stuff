"""
We can also do Quantile regression with CVXOPT.
And maybe reguarlized with shrinkage too I don't know.

"""

import cvxopt as cvx
import cvxopt.modeling as cvxm
from cvxopt import solvers
solvers.options["show_progress"] = False
solvers.options["abstol"] = 1e-15
solvers.options["reltol"] = 1e-10
solvers.options["refinement"] = 10

import numpy as np

def QuantileRegRegularized(
        V:np.ndarray,
        y:np.ndarray,
        alpha:float=0.5,
        l:float=0.001):
    """
        It does a quantile regression, given the vandermond matrix (It can be in any basis)
    :param V:
        vandermonde
    :param y:
        The labels for the regression
    :param alpha:
        The quantile we are interested in, 0.95 means that, the output is the 0.975 quantile estimation of the
        value of the function.
    :return:
    """
    assert V.ndim == 2 and V.shape[0] == y.shape[0]
    assert 0 < alpha < 1
    m, n = V.shape
    A = cvx.matrix(V)
    x = cvxm.variable(n)
    b = cvx.matrix(y)
    Epsilon = cvxm.variable(m)
    Delta = cvxm.variable(n)
    Lp = cvxm.op(cvxm.sum(Epsilon))
    Lp.addconstraint(-alpha*Epsilon <= b - A*x)
    Lp.addconstraint(b-A*x <= Epsilon*(1 - alpha))
    Lp.addconstraint(-Delta <= l*x <= Delta)
    Lp.solve()
    return (np.array(x.value)).reshape(-1)


class PolyRegression:
    """

    """
    def __init__(this):
        return


def main():
    N = 30                         # number of sampling.
    f = lambda x: np.cos(np.pi* x) # Grond truth
    random = np.random.random
    x = random(N)
    epsilon = random()

    pass


if __name__ == "__main__":
    import os
    import sys
    print(f"curdir: {os.curdir}")
    print(f"cwd: {os.getcwd()}")
    print(f"executable: {sys.executable}")
    main()