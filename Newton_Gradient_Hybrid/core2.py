"""
    First order optimizer.
    Use finite diff to approximate Hessian.
    Use both newton's or accelerated gradient, which ever is better.
    When gradient is small, it start to random sample using local hessian and the sequence of descned parameters.
"""

import numpy as np
import matplotlib.pyplot as plt


class HybridOptimizer:

    ApproxHessian = 1     # Getting the Hessian with BFGS's algorithm
    FiniteDiffHessian = 2 # Getting the Hessian by finite diff on gradient

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
        assert isinstance(f(x0), float) or isinstance(f(x0), int)
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

    @property
    def H(this):
        return this._H.copy()

    def _Initialize(_):
        while _._Xnex is None or _.f(_._Xnex) > _.f(_._Xpre):
            _._Velocity = _._Momentum * _._Velocity - _._Eta*_.df(_._Xpre)
            _._Xnex = _._Xpre + _._Velocity
            if not(_._Xnex is None): _._Eta /= 2
        _._Gnex = _.df(_._Xnex)
        _._H = _._UpdateHessian()
        _.Xs.append(_._Xnex)


    def _UpdateHessian(_):
        """
            Use Finite diff to find the Hessian at the point Xnex.
        :return:
            Hessian at point Xnex
        """
        x1, x2 = _._Xpre, _._Xnex
        n = x1.shape[0]
        df = _.df
        h = np.mean(x2 - x1)
        H = np.zeros((n, n))

        def e(i):
            x = np.zeros((n, 1))
            x[i, 0] = 1
            return x

        for I in range(n):
            df1 = df(x2 + h*e(I))
            df2 = df(x2 - h*e(I))
            diff = df1 - df2
            if np.min(abs(diff)) == 0:
                return None
            g = (diff)/(2*h)
            H[:, [I]] = g
        return H

    def _UpdateRunningParams(_, Xnex):
        """
            Update all the running parameters for the class.
        :param Xnex:
            next step to take.
        :return:
        """
        _._Xpre, _._Xnex = _._Xnex, Xnex
        _.Xs.append(Xnex)
        _._G, _._Gnex = _._Gnex, _.df(Xnex)
        _._H = _._UpdateHessian()

    def _TrySecant(_):
        df1, df2 = _._Gpre, _._Gnex
        assert df2.ndim == 2, "expected gradient to be 2d np array. "
        pinv = np.linalg.pinv
        H = _._H
        Xnex = _._Xnex - pinv(H)@df2
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
            Only try newton's method when the gradient is decreasing and the
            Hessian has positive determinant.
        :return:
            The point that is chosen next
        """

        df1, df2 = _._Gpre, _._Gnex
        f = _.f
        det, norm = np.linalg.det, np.linalg.norm
        Xnex = _._TryAccGradient()
        Which = "G;"
        if not _._H is None and ((norm(df2) < norm(df1) and norm(df2) > 1e-8) or det(_._H) > 0):
            XnexNewton = _._TrySecant()
            Which += "N"
            if f(XnexNewton) < f(Xnex):
                _._Velocity = 0              # reset velocity.
                _._Eta = 1/(2*norm(_._H)**2) # reset learning rate too, using the Hessian
                Xnex = XnexNewton
                Which += ". N < G"
            else:
                Which += ". G < N"

        _._UpdateRunningParams(Xnex)
        return Xnex, Which

    def Generate(_):

        pass


def main():

    def TestWithSmoothConvex():
        norm = np.linalg.norm
        N = 100
        A = np.random.rand(N,N)
        x0 = np.random.rand(N, 1)*20
        f, df = lambda x: norm(A@x)**2, lambda x: 2*(A.T@A)@x
        eta = 1/(4*norm(A.T@A))
        optim = HybridOptimizer(f, df, x0, eta, 0.2)
        ObjectivVals = []
        for I in range(10):
            v, w = optim()
            ObjVal = f(v)
            print(f"{v[0, 0]}; {v[1, 0]}, objval: {ObjVal}, {w}")
            ObjectivVals.append(ObjVal)
            if len(ObjectivVals) > 2 and (ObjectivVals[-1] - ObjectivVals[-2]) > 1e-8:
                break

        plt.plot(ObjectivVals); plt.title("objective value")
        plt.show()

    def TestWithAFuncFromWiki():
        g = lambda x, y: (1.5 - x + x*y)**2 + (2.25 - x + x*y**2)**2 + (2.625 - x + x*y**3)**2
        dfx = lambda x, y: 2*(y - 1)*(1.5 - x + x*y) + \
                         2*(y**2 - 1)*(2.25 - x + x*y**2) + \
                         2*(y**3 - 1)*(2.625 - x + x*y**3)
        dfy = lambda x, y: 2*x*(1.5 -x + x*y) + \
                           4*x*y*(2.25 - x + x*y**2) + \
                           (6*x*y**2)*(2.625 - x + x*y**3)
        df = lambda x: np.array(
            [
                [dfx(x[0, 0], x[1, 0])],
                [dfy(x[0, 0], x[1, 0])]
            ])
        f = lambda x: g(x[0, 0], x[1, 0])
        x0 = np.random.rand(2,1)*1

        eta = 0.01
        optim = HybridOptimizer(f, df, x0, eta, 0.5)
        ObjectivVals = []
        x = x0
        for I in range(240):
            v, w = optim()
            ObjVal = f(v)
            print(f"{v[0, 0]}; {v[1, 0]}, objval: {ObjVal}, {w}")
            ObjectivVals.append(ObjVal)
            if np.linalg.norm(v - x) < 1e-10:
                break
            x = v

        plt.plot(ObjectivVals);
        plt.title("objective value")
        plt.show()


    TestWithAFuncFromWiki()


if __name__ == "__main__":
    import os
    print(f"{os.curdir}")
    print(f"{os.getcwd()}")
    main()





