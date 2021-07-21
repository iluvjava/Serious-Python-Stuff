"""
    Common subroutines in constrained smooth optimizations.
    1. Line search subroutines
    2. Misc Method that help with shit, it's gonna be messy.

"""


def ArmijoLineSearch(f, df, xk, p, eta:float=1, c1=0.9, decay=0.5, diagnostic:bool=False):
    """

    :param f:
    :param df:
    :param p:
    :param eta:
    :param c1:
    :param decay:
    :return:
    """
    FuncQ = 0
    while f(xk + eta*p) > xk + c1*eta*(df(xk).T@p):
        eta *= decay
        FuncQ += 1

    if diagnostic: return eta, FuncQ
    return eta



