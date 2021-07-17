import numpy as np
import math as math



def HybridNewtonGradient_Scalar(
        f: callable,
        df:callable,
        ddf:callable,
        x0:float,
        eta:float,
        maxitr=10
    ):
    Xpre = x0
    G = df(Xpre)
    for I in range(maxitr):
        XGradient = Xpre - eta*G
        Xnext = Xpre - G/ddf(Xpre)
        Which = "Newton"
        if f(Xnext) > f(XGradient):
            Which = "Gradient"
            Xnext = XGradient
        yield Xnext, Which
        Xpre = Xnext
        G = df(Xnext)

def SecantGradient_Scalar(
        f:callable,
        df:callable,
        x0:float,
        eta:float=0.1,
        maxitr:int=100
    ):
    def finiteDiff(y1, y2, dx):
        return (y2 - y1)/dx
    # Gradient first
    Xpre = x0
    Gpre = df(Xpre)
    Xnext = Xpre - eta*Gpre
    Gnext = df(Xnext)
    yield Xnext

    for _ in range(maxitr):
        Xnext2 = Xnext - eta*Gnext
        Xnewton = Xnext - Gnext/finiteDiff(Gpre, Gnext, Xnext - Xpre)
        if f(Xnewton) < f(Xnext2):
            Xnext2 = Xnewton
        yield Xnext2
        Xpre, Xnext = Xnext, Xnext2
        Gpre, Gnext = Gnext, df(Xnext2)



def main():
    f = lambda x: (math.sin(x) + x)**2
    df = lambda x: 2*(math.sin(x) + x)
    ddf = lambda x: 2*(math.sin(x) + x)*(-math.sin(x)) + 2*(math.cos(x) + 1)**2
    x0 = 100.5*math.pi
    eta = 0.1
    for I, (X, W) in enumerate(HybridNewtonGradient_Scalar(
        f, df, ddf, x0, eta, maxitr=1000
    )):
        print(f"X: {X}, Strategy Chose: {W}")
        if f(X) < 1e-14:
            break
    print(f"Gradient Newton's Hybrid Iterations Count: {I}, Objval: {f(X)}")

    for I, X in enumerate(SecantGradient_Scalar(
        f, df, x0, eta, maxitr=1000
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
