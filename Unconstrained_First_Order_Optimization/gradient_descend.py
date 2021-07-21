from enum import Enum

class GradientMethodOptions(Enum):
    """
        Choose Multiple.
    """
    ArmijoLineSearch = 1
    KeepSmallestLearningRate = 2
    GradientRestart = 3
    Diagnostic = 4

class GradientAlgorithmOption (Enum):
    """
        Choose ONE.
    """

    Vanilla = 1
    ClassicAcc = 2
    NesterovAcc = 3
    Steepest = 4

class GradientMethod:


    def __init__(
            this,
            f,
            df,
            x0,
            eta:float=1,
            **kargs):
        pass

    def __call__(self):
        pass
