from enum import Enum

class GradOpt(Enum):
    """
        Choose Multiple.
        1. Use ArmijoLineaSearch Subroutine for all descend directions
        2. Keep the smallest learning rate from linesearch and accumulate them.
        3. Set velocity to zero for accelerated gradient, if objective increases.
        4. Store all objective value, and usage data on f, df, for benchmarking.
        5. Store all choices of the algorithm in detail as string in the field.
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
        2. Classic accelerations.
            - Can be used with line search and minimal lr
            - support gradient restart.
        3. Nesterov accelerated gradient method.
            - Can be used with line search and minimal lr
            - support gradient restart.
    """

    Vanilla = 1
    ClassicAcc = 2
    NesterovAcc = 3
    Steepest = 4

class GradientMethod:
    """
        An interactive class for Gradient Descend Subroutines.
    """

    def __init__(
            this,
            f,
            df,
            x0,
            eta:float=1,
            algo:GradAlg=GradAlg.Vanilla,
            **kargs):
        """

        :param f:
        :param df:
        :param x0:
        :param eta:
        :param algo:
        :param kargs:
            key: opts; values: list of GradOpt.
        """
        pass


    def __call__(self):
        pass






def test():
    pass


if __name__ == "__main__":
    import os
    print(f"{os.curdir}")
    print(f"{os.getcwd()}")
    test()

