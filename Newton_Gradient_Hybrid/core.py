"""
Blow are 2 optimizer that can overcome inflection points, because of the mixed use of Newton's method and
accelerated gradient method.


"""

import numpy as np
import math as math

def HybridNewtonGradient_Scalar(
        f: callable,
        df:callable,
        ddf:callable,
        x0:float,
        eta:float,
        momentum:float = 0.2,
        maxitr=10
    ):
    Xpre = x0
    velocity = 0
    G = df(Xpre)
    for I in range(maxitr):
        velocity = momentum*velocity - eta*df(Xpre + momentum*velocity)
        XGradient = Xpre + velocity
        # Try newton if gradient is accelerating, else don't try newton.
        Gnex = df(XGradient)
        if abs(Gnex) < abs(G):
            ddfXpre = ddf(Xpre)
            Xnewton = Xpre - G/ddfXpre
            if f(Xnewton) < f(XGradient) and abs(ddfXpre) > 1e-8:
                Xnext = Xnewton
                Which = "Tried Gradient and Newton, Newton is better"
                G = df(Xnext)
                velocity = 0
            else:
                Xnext = XGradient
                Which = "Tried Gradient and Newton, Gradient is better"
                G = Gnex

        else:
            Which = "Gradient, did't try Newton"
            Xnext = XGradient
            G = Gnex
        yield Xnext, Which
        Xpre = Xnext


def SecantGradient_Scalar(
        f:callable,
        df:callable,
        x0:float,
        eta:float=0.1,
        momentum:float=0.2,
        maxitr:int=100
    ):
    def finiteDiff(y1, y2, dx):
        return (y2 - y1)/dx
    # Gradient first
    Xpre = x0
    Gpre = df(Xpre)
    Xnext = Xpre - eta*Gpre
    Gnext = df(Xnext)
    velocity = Xnext - Xpre
    yield Xnext
    for _ in range(maxitr):
        velocity = momentum*velocity - eta*df(Xpre + momentum*velocity)
        Xnext2 = Xnext + velocity
        if Xnext2 - Xnext == 0:
            return Xnext2
        H = finiteDiff(Gpre, Gnext, Xnext - Xpre)
        if abs(Gnext) < abs(Gpre) and H > 1e-8:
            Xnewton = Xnext - Gnext/H
            if f(Xnewton) < f(Xnext2):
                Xnext2 = Xnewton
                velocity = 0
            yield Xnext2
        else:
            yield Xnext2
        Xpre, Xnext = Xnext, Xnext2
        Gpre, Gnext = Gnext, df(Xnext2)


def main():
    a = 1
    f = lambda x: (math.sin(x) + a*x)**2
    df = lambda x: 2*(math.sin(x) + a*x)*(math.cos(x) + a)
    ddf = lambda x: 2*(math.sin(x) + a*x)*(-math.sin(x)) + 2*(math.cos(x) + a)**2
    x0 = 10.2*math.pi
    eta = 0.2
    momentum = 0.3
    for I, (X, W) in enumerate(HybridNewtonGradient_Scalar (
        f, df, ddf, x0, eta, momentum=momentum, maxitr=1000
    )):
        print(f"X: {X}, Strategy Chose: {W}")
        if f(X) < 1e-14:
            break
    print(f"Gradient Newton's Hybrid Iterations Count: {I}, Objval: {f(X)}")

    for I, X in enumerate(SecantGradient_Scalar(
        f, df, x0, eta, maxitr=100, momentum=momentum
    )):
        print(f"X: {X}")
        if f(X) < 1e-14:
            break
    print(f"Secant Gradient Hybrid Iterations Count: {I}, Objval: {f(X)}")


if __name__ == "__main__":
    import os
    print(f"{os.curdir}")
    print(f"{os.getcwd()}")
    main()
