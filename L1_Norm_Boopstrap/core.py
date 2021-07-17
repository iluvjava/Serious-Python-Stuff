import matplotlib.pyplot as plt
import numpy as np
import cvxopt as cvx
import cvxopt.modeling as cvxm
from cvxopt import solvers
solvers.options["show_progress"] = False
solvers.options["abstol"] = 1e-15
solvers.options["reltol"] = 1e-10
solvers.options["refinement"] = 10

from tqdm import tqdm

def L1RegFit(X: np.ndarray, y:np.ndarray):
    assert X.ndim == 2 and X.shape[0] == y.shape[0]
    m, n = X.shape
    A = cvx.matrix(X)
    x = cvxm.variable(n)
    b = cvx.matrix(y)
    Epsilon = cvxm.variable(m)
    Lp = cvxm.op(cvxm.sum(Epsilon))
    Lp.addconstraint(-Epsilon <= b - A*x)
    Lp.addconstraint(b-A*x <= Epsilon)
    Lp.solve()
    return (np.array(x.value)).reshape(-1)


def L2RegFit(X, y):
    return (np.linalg.pinv(X.T@X)@X.T@y).reshape(-1)


def RegBoopStrap(
            X:np.ndarray,   # regression data
            y:np.ndarray,   # labels
            fxn:callable,   # model training function
            queries=None,   # the points to query the confident interval
            trials=None,    # number of times to make the boostrap the model
            boostrapSize:int=None,
            alpha:float=5 # confidence band interval
        ):
    """
        Performs non-parametric boopstrap for regression models
    :param X:
    :param y:
    :param fxn:
    :param queries:
    :param trials:
    :param boostrapSize:
    :return:
    """
    assert X.ndim == 2 and X.shape[0] == y.shape[0] and y.ndim == 1
    assert X.ndim == 2
    assert alpha < 50 and alpha > 0
    m, n = X.shape
    trials = X.shape[0] if trials is None else trials
    queries = X[:, 1] if queries is None else queries
    queries.sort()
    bootstrapSize = m if boostrapSize is None else boostrapSize
    assert trials > 20
    assert bootstrapSize <= m, "don't recommend bootstrap size to be larger than sample size"
    BoostrapYhat = []
    for _ in tqdm(range(trials)):
        Selection = np.random.randint(0, m, bootstrapSize)
        Coefficient = fxn(X[Selection], y[Selection])
        V = np.vander(queries, Coefficient.shape[0])
        Yhat = (V@Coefficient[:, np.newaxis]).reshape(-1)
        BoostrapYhat.append(Yhat)
    OutputMatirx = np.array(BoostrapYhat)
    ConfidenceBand = np.percentile(OutputMatirx, [alpha/2, 100 - alpha/2], axis=0)
    return ConfidenceBand


def main():
    x = np.arange(-1, 1, 0.025)
    X = np.vander(x, 5)
    f = lambda x: np.cos(np.pi*x)
    y = f(x) + np.random.randn(X.shape[0]) * 0.2
    for II in range(2, len(y) - 2):
        if np.random.rand() < 0.1:
            y[II] = -3

    def RobustOutlier():
        Alpha1 = L1RegFit(X, y)
        Alpha2 = L2RegFit(X, y)
        yhat1 = X@Alpha1[:, np.newaxis]
        yhat2 = X@Alpha2[:, np.newaxis]
        plt.plot(x, y, "x")
        plt.plot(x, yhat1)
        plt.plot(x, yhat2)
        plt.legend(["data", "L1", "L2"])
        plt.show()
        return Alpha1, Alpha2

    Alpha1, Alpha2 = RobustOutlier()

    def Bootstrap():
        Queries = np.arange(-1, 1, 0.01)
        ConfidenceBand1 = RegBoopStrap(
            X,
            y,
            L1RegFit,
            queries= Queries,
            alpha=1,
            trials=300
        )
        ConfidenceBand2 = RegBoopStrap(
            X,
            y,
            L2RegFit,
            queries= Queries,
            alpha=1,
            trials=300
        )
        yhat1 = X@Alpha1[:, np.newaxis]
        yhat2 = X@Alpha2[:, np.newaxis]
        plt.plot(x, f(x))
        plt.plot(x, yhat1)
        plt.plot(x, yhat2)
        plt.plot(x, y, "x")
        plt.fill_between(Queries,
                         ConfidenceBand1[0],
                         ConfidenceBand1[1],
                         alpha=0.2,
                         color="tab:orange")
        plt.fill_between(Queries,
                         ConfidenceBand2[0],
                         ConfidenceBand2[1],
                         alpha=0.2,
                         color='tab:green')
        plt.title("Confidence Band 1, 2 Norm Regression")
        plt.legend(["Ground Truth", "Model L1 Loss", "Model L2 Loss"])
        plt.savefig("bootstrap-l1-l2.png", dpi=400)
        plt.show()
    Bootstrap()


if __name__ == "__main__":
    import os
    print(f"{os.curdir}")
    print(f"{os.getcwd()}")
    main()
