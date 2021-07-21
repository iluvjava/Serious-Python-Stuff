from enum import Enum
from utils import *
import numpy as np
import numpy.linalg as npl
import scipy.optimize as opt

class GradOpt(Enum):
    """
        Choose Multiple.
        1. Use ArmijoLineaSearch Subroutine for all descend directions
        2. Keep the smallest learning rate from linesearch and accumulate them.
        3. Set velocity to zero for accelerated gradient, if objective increases.
        4. Store all objective value, and usage data on f, df, for benchmarking.
        5. Store all choices of the algorithm in detail as string in the field.
    """
    ArmijoLineSearch = 1
    SteepestDescend = 2

    KeepSmallestLearningRate = 3
    GradientRestart = 4
    Diagnostic = 5
    KeepReports = 6


class GradAlg (Enum):
    """
        Choose ONE.
        1. Fixed learning rate gradient method.
            - Can be used with line search and minimal lr
        2. Classic accelerations.w
            - Can be used with line search and minimal lr
            - support gradient restart.
        3. Nesterov accelerated gradient method.
            - Can be used with line search and minimal lr
            - support gradient restart.
        4. Conjugate Gradient Method.
            ????

    """

    Vanilla = 1
    ClassicAcc = 2
    NesterovAcc = 3
    ConjugateGrad = 4


class GradientMethod:
    """
        An interactive class for Gradient Descend Subroutines.
        Call the solver to get the next step in the optimization procedures.

        * It's slow because too many queries and branching
        * It's for analyzing behaviors of gradient method under different
        line search policies and parameters.
    """

    def __init__(
            this,
            f:callable,
            df:callable,
            x0,
            eta:float=0.1,
            momentum:float=0.2,
            algo:GradAlg=GradAlg.Vanilla,
            **kargs):
        """

        :param f:
        :param df:
        :param x0:
        :param eta:
            Maximum step size.
        :param algo:
        :param kargs:
            key: opts; values: list of gradient options from enum class GradOpt
            key: box; values: tuples of box constrains.

        """
        this.f, this.df = f, df
        this.eta, this.m = eta, momentum

        #Parameters for optimization:
        this._v = 0 # velocity
        this._Xpre = x0
        this._ObjvalPre = f(x0)
        assert type(this._ObjvalPre) is float or type(this.ObjVal) is int, "f, should be a scalar function."
        this._Gpre = df(x0)
        assert x0.shape == x0, "the gradient and the input vector for objective should have the same shape."

        # Misc parameters
        this.Report = ""
        this.Xs = [x0]
        this.ObjVal = [this.ObjVal]
        this.Gs = [this._Gpre]
        this.Kargs = kargs
        this.Options = kargs["opts"] if "opts" in kargs else {}
        this.Box = kargs["box"] if "box" in kargs else {}
        this.Algo = algo

    def _Report(this, mesg):
        if GradOpt.Diagnostic in this.Options:
            this.Report += mesg

    def _LineSearch(this, p=None, eta=None):
        """
        :param p:
            Search direction. A vector. the length of this vector and eta
            determines the range for the line search subroutines.
        :param eta:
            Maximal step size.
        :return:
            next guess for function f.
        """
        xk, f, df, g = this._Xpre, this.f, this.df, this._Gpre
        eta = this.eta if eta is None else eta
        p = -g if p is None else p
        assert p.dot(g) < 0, "Search direction is increasing the objective"
        # setup Xnex depending on Policies.
        if GradOpt.ArmijoLineSearch in this.Options:
            eta = ArmijoLineSearch(f, df, xk, eta)
            Xnex = xk + eta*p
        elif GradOpt.SteepestDescend in this.Options:
            OptRes = opt.minimize_scalar(lambda t: f(xk + t*p), bounds=(0, eta))
            eta = OptRes.x
            Xnex = xk + eta*p
        else:
            # faith full step with maximal step size on direction p
            Xnex = xk + eta*p
        return Xnex

    def _VanillaGrad(this):
        """
            Gradient search direction, stepsize is learning rate.
        :return:
            pass
        """
        Xnex = this._LineSearch()
        return Xnex

    def _ClassicAcc(this):
        v, m = this._v, this.m
        xk, f, df, g = this._Xpre, this.f, this.df, this._Gpre
        eta = this.eta
        p = m*v - eta*g
        if p.dot(g) < 0: # only line search when it's in the decreasing direction
            Xnex = this._LineSearch(p=p, eta=1)
        else:
            if GradOpt.GradientRestart in this.Options:
                p = -eta*g
                Xnex = this._LineSearch(p=p, eta=1)
            else:
                Xnex = xk + p
        this._v = Xnex - xk # update momentum.
        return Xnex

    def _NesterovAcc(this):
        v, m = this._v, this.m
        xk, f, df, g = this._Xpre, this.f, this.df, this._Gpre
        eta = this.eta

        pass

    def __call__(this, proximal:callable=None):
        """
            Run specified optimization subroutine and then return the
            next guess
            * Will update the guess before next guess
            * Warning, box contraints will be added in addition to the given
            proximal operators.
        :param proximal
            Proximal operator for non-smooth/constraints part of the objective.
        :return:
            The next guess, a vector.
        """


        pass

    def __repr__(this):

        pass

    def Generate(this, maxitr):
        """
            Generate a sequence of guesses using this optimizer, it yield them
            one by one.
        :return:

        """

        pass

    def Reset(this, x0=None):
        """
            Reset the optimizer at current point, or a new point.
            * Clear momentum
            * reset maximal step size
            * Clear all diagnostic
            * clear all stored parameters.
        :return:
            None
        """
        pass


def test():
    pass


if __name__ == "__main__":
    import os
    print(f"{os.curdir}")
    print(f"{os.getcwd()}")
    test()

