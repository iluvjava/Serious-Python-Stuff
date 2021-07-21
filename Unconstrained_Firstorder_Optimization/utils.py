"""
    Common subroutines in constrained smooth optimizations.
    1. Line search subroutines
    2. Misc Method that help with shit, it's gonna be messy.

"""

import numpy as np

def ArmijoLineSearch(
        f:callable,
        df:callable,
        xk:np.ndarray,
        p:np.ndarray,
        eta:float=1,
        c1:float=0.9,
        decay:float=0.5,
        diagnostic:bool=False):
    """

    :param f:
    :param df:
    :param p:
    :param eta:
    :param c1:
    :param decay:
    :return:
        The stepsize chosen. Not the next guess!
    """
    FuncQ = 0
    while f(xk + eta*p) > xk + c1*eta*(df(xk).T.dot(p)):
        eta *= decay
        FuncQ += 1
    if diagnostic: return eta, FuncQ
    return eta



def BoxProject(x:np.ndarray, boxconstraints):
    """
        Given vector x, project it onto hyper dimensional set of box
        constraints.
    :param x:
        a vector
    :param boxconstraints:
        list of tuples
    :return:
        projected vector.
    """

    pass