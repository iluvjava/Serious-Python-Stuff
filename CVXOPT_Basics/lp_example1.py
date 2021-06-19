import cvxopt as cvx
import cvxopt.modeling as cvxm
from cvxopt import solvers
solvers.options["show_progress"] = True
solvers.options["abstol"] = 1e-15
solvers.options["reltol"] = 1e-10
solvers.options["refinement"] = 10
import numpy as np
from typing import *


def WeightedIntervalSchedule(
            intervals:List[Tuple[Union[float, int]]],
            weights:List[float]
        ):
    assert len(intervals) == len(weights)
    Ts = set()
    for t1, t2 in intervals:
        Ts.add(t1); Ts.add(t2)
    TimeFrames = sorted(list(Ts))
    m, n = len(Ts), len(weights)
    A = np.zeros((m, n))
    for (II, JJ), Val in np.ndenumerate(A):
        if TimeFrames[II] <= intervals[JJ][1] \
                and TimeFrames[II] >= intervals[JJ][0]:
            A[II, JJ] = 1
        else:
            A[II, JJ] = 0
    A = cvx.matrix(A)
    x = cvxm.variable(n)
    b = cvx.matrix(np.ones(m))
    c = cvx.matrix(-np.array(weights, dtype=np.float64))
    Constraint1 = (A*x <= b)
    Constraint2 = (x >= 0)
    Lp = cvxm.op(cvxm.dot(c, x), [Constraint1, Constraint2])
    Lp.solve()
    return [round(II, 8) for II in x.value]


def main():
    Intervals = [(0, 2), (1, 3), (3, 5), (2, 6)]
    Weights = [1, 1, 1, 1]
    Res = WeightedIntervalSchedule(Intervals, Weights)
    print(f"Results is {Res}")


if __name__ == "__main__":
    import os
    print(f"{os.curdir}")
    print(f"{os.getcwd()}")
    main()
