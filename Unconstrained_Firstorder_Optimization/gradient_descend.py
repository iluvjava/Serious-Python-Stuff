__all__ = ["GradOpt", "GradAlg", "GradientMethod"]

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
        4. GradientRestart:
            - Restart the velocity of the momentum method's search direction is
            not increasing the objective on the gradient of the previous point.

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
            x0:NpTorch,
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
        assert momentum > 0 and eta > 0 and momentum < 1, "Eta, maximal step size should be positive and momentum should be betwen 0 and 1. "
        this.eta, this.m = eta, momentum


        this._Setup(x0, algo, kargs)


    def _Setup(this,
               f:callable,
               df:callable,
               x0:NpTorch,
               algo:GradAlg,
               **kargs):

        # Parameters for optimization:
        this._v = 0  # velocity
        this._Xpre = x0
        this._ObjvalPre = f(x0)
        assert isinstance(this._ObjvalPre, float) or isinstance(this._ObjvalPre, int), "f, should be a scalar function."
        this._Gpre = df(x0)
        assert x0.shape == this._Gpre.shape, "the gradient and the input vector for objective should have the same shape."
        # Misc parameters
        this.Kargs = kargs
        this.Options = kargs["opts"] if "opts" in kargs else {}
        this.Box = kargs["box"] if "box" in kargs else None
        this.Algo = algo

        # diagonostic params
        this.Report = ""
        this.Xs = []
        this.ObjVal = []
        this.Gs = []
        if GradOpt.Diagnostic in this.Options:
            this.Xs.append(x0)
            this.ObjVal.append([this.ObjVal])
            this.Gs.append([this._Gpre])

    def _Report(this, mesg:str):
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
        assert (p.T).dot(g) <= 0, "Search direction is increasing the objective"
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
        if (p.T).dot(g) < 0: # only line search when it's in the decreasing direction
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
        # TODO: IMPLEMENT

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

        if GradAlg.Vanilla is this.Algo:
            Xnex = this._VanillaGrad()
        elif GradAlg.ClassicAcc is this.Algo:
            Xnex = this._ClassicAcc()
        elif GradAlg.NesterovAcc is this.Algo:
            # TODO: IMPLEMENT
            raise Exception("Method Not implemented")
        elif GradAlg.ConjugateGrad is this.Algo:
            # TODO: IMPLEMENT
            raise Exception("Method Not implemented")
        else:
            raise Exception("Method Not implemented")
        if this.Box is not None:
            # TODO: IMPLEMENT
            pass
        if GradOpt.Diagnostic in this.Options:
            # TODO: IMPLEMENT
            pass

        this._Xpre = Xnex
        this._Gpre = this.df(Xnex)
        this.ObjVal = this.f(Xnex)
        return Xnex



    def __repr__(this):
        # TODO: IMPLEMENT
        pass

    def Iterator(this, maxitr):
        # TODO: Implement
        pass

    def Generate(this, maxitr):
        """
            Generate a sequence of guesses using this optimizer, it yield them
            one by one.
            * Intended to be used as an iterator.
        :return :

        """

        # TODO: Implement
        pass


    def Reset(
            this,
            x0:NpTorch,
            eta:float=0.1,
            momentum:float=0.2,
            algo:GradAlg=GradAlg.Vanilla,
            **kargs):
        """
            Reset the optimizer at current point, or a new point.
            * Clear momentum
            * reset maximal step size
            * Clear all diagnostic
            * clear all stored parameters.
        :return:
            None
        """
        # TODO: Implement
        this._Setup(this.f, this.df, x0, eta, momentum, algo, kargs)

        pass


def test1():
    norm = npl.norm
    A = np.array([[1, 2], [2, 1]])
    b = np.array([[1], [1]])
    x0 = np.array([[0], [0]])
    f = lambda x: norm(A.dot(x) - b)**2
    df = lambda x: 2*(A.T).dot(A.dot(x) - b)
    print("Creating a simple problem...")
    print("Testing Default Settings...")
    Subject = GradientMethod(f, df, x0, eta=1/(4*norm(A.T@A)))
    for _ in tqdm(range(1000)):
        Subject()
    FinalVal = f(Subject())
    assert FinalVal < 1e-16, f"didn't converge in 1000 iterations, finalval {FinalVal}"
    print("Test Finished")
    print("Reset and Adding Armijo linesearch")





if __name__ == "__main__":
    from tqdm import tqdm
    import matplotlib as plt
    import os
    print(f"{os.curdir}")
    print(f"{os.getcwd()}")
    test1()

