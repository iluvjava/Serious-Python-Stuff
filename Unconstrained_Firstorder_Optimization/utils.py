"""
    Common subroutines in constrained smooth optimizations.
    1. Line search subroutines
    2. Misc Method that help with shit, it's gonna be messy.

"""
from typing import *
import torch as torch
import numpy as np
NpTorch = Union[np.ndarray, torch.Tensor]
Box = List[Tuple[Union[int, float]]]

def ArmijoLineSearch(
        f:callable,
        df:callable,
        xk:NpTorch,
        p:NpTorch,
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



def ProximalBoxConstraints(
        x:NpTorch,
        boxconstraints:List[Tuple]
):
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




