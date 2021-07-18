"""
    First order optimizer.
    Use finite diff to approximate Hessian.
    Use both newton's or accelerated gradient, which ever is better.
    When gradient is small, it start to random sample using local hessian and the sequence of descned parameters.
"""

import numpy as np

class HybridOptimizer:
    """
        A class that binds together:
         * a Newton's method using hessian approximated by the gradient
         * A nesterov accelerated gradient method.

    """
    def __init__(
            _,
            f:callable,     # objective function
            df:callable,    # derivative of the function
            x0:np.ndarray,  # initial guess for optimization
            eta:float,      # the learning rate.
            momentum:float):
        """

        :param f:
        :param df:
        :param x0:
        :param eta:
        :param momentum:
        :return:
        """
        _.f, _.df = f, df
        _._Momentum = momentum
        _._Eta = eta

        # running parameters for optimization:
        _._Xpre = x0
        _._Xnex = None
        _._Gpre = df(x0)
        _._Gnex = None
        _._Velocity = 0
        _._H = None # Hessian using 2 gradients.

        # Running parameters for benchmarking and analyzing.
        _.Xs = [_._Xpre]
        _.Report = ""
        _._Initialize()

    def _Initialize(_):
        _._Velocity = _._Momentum * _._Velocity - _.df(_._Xpre)
        _._Xnex = _._Xpre + _._Velocity
        _._Gnex = _.df(_._Xnex)
        _._H = (_._Gnex - _._Gpre.T) / (_._Xnex - _._Xpre.T)
        _.Xs.append(_._Xnex)

    def _UpdateRunningParams(_, Xnex):
        _._Xpre, _._Xnex = _._Xnex, Xnex
        _.Xs.append(Xnex)
        _._G, _._Gnex = _._Gnex, _.df(Xnex)
        _._H = (_._Gnex - _._Gpre.T)/(_._Xnex - _._Xpre.T)

    def _TrySecant(_):
        df1, df2 = _._Gpre, _._Gnex
        assert df2.ndim == 2, "expected gradient to be 2d np array. "
        pinv = np.linalg.pinv
        H = _._H
        Xnex = _._Xpre - pinv(H)@df2
        return Xnex

    def _TryAccGradient(_):
        x1, x2 = _._Xpre, _._Xnex
        eta = _._Eta
        v, m = _._Velocity, _._Momentum
        _._Velocity = m * v - eta*_.df(x2 + v)  # Update velocity!
        Xnex = x2 + _._Velocity
        return Xnex

    def __call__(_):
        """
            Call on this function to get the next point that it will step into.
        :return:
            The point that is chosen next
        """
        df1, df2 = _._Gpre, _._Gnex
        f = _.f
        norm = np.linalg.norm
        Xnex = _._TryAccGradient()
        if norm(df2) < norm(df1) and norm(df2) > 1e-8 and norm(_._H) > 1e-8:   # Try newton
            XnexNewton = _._TrySecant()
            _._Velocity = 0         # reset velocity.
            if norm(f(XnexNewton)) < norm(f(Xnex)):
                Xnex = XnexNewton
        _._UpdateRunningParams(Xnex)
        return Xnex

    def Generate(_):

        pass




def main():
    f, df = lambda x: x**2, lambda x: 2*x
    optim = HybridOptimizer(f, df, np.array([[10], [10]]), 0.001, 0.2)
    for I in range(3):
        print(optim())



if __name__ == "__main__":
    import os
    print(f"{os.curdir}")
    print(f"{os.getcwd()}")
    main()





