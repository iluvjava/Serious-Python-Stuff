import core.cg_core
AbstractConjugateGradient = core.cg_core.AbstractConjugateGradient
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import scipy.sparse as sparse
from tqdm import tqdm
from scipy import signal


def ReadImageAsNumpy(file:str):
    img = Image.open(file)
    numpydata = np.asarray(img, dtype=np.float32)
    return numpydata


def BoxBlur(Image:np.ndarray, boxsize=5):
    assert Image.ndim == 2 or Image.ndim == 3, \
         "Should be a matrix, or a 3d tensor. "
    if Image.ndim == 3:
        assert Image.shape[2] == 3, \
            "Image should have 3 color channels on the last axis. "
    Kernel = np.ones((boxsize, boxsize)) / boxsize ** 2
    if Image.ndim == 2:
        return signal.convolve2d(Image, Kernel, boundary='wrap', mode='same')
    else:
        Blurred = np.zeros(Image.shape)
        for II in range(Image.shape[2]):
            Blurred[..., II] = \
                signal.convolve2d(Image, Kernel, boundary='wrap', mode='same')
        return Blurred



def Sharpen(Image:np.ndarray):
    Kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    return signal.convolve2d(Image, Kernel, boundary='wrap', mode='same')


def SuccessiveBlurr(Image: np.ndarray, boxsize=5, times=3):
    Blurred = Image
    for __ in range(times):
        Blurred = BoxBlur(Blurred, boxsize=boxsize)
    return Blurred


def main():
    TheImage = ReadImageAsNumpy(".\\data\\image2.png")
    TheImage /= 255
    TheImage = np.mean(TheImage, axis=2)
    Matrix = TheImage[0:-1:5, 0:-1:5]

    def MoreInvolved():
        N = 2
        BoxSize= 5
        Blurred = SuccessiveBlurr(Matrix, times=N, boxsize=BoxSize)
        plt.matshow(Blurred); plt.title(f"Kernel Size {BoxSize}, Iteration: {N}")
        plt.show()
        Sol = AbstractConjugateGradient(lambda x: SuccessiveBlurr(x, times=N, boxsize=BoxSize),
                                        Blurred,
                                        np.ones(Matrix.shape),
                                        tol=1e-4,
                                        maxitr=50*50,
                                        verbose=True)
        Deblurred = Sol.X[-1]
        plt.matshow(Deblurred); plt.title(f"Deblurred Image")
        plt.show()
        plt.matshow(Matrix); plt.title("Original Image")
        plt.show()

    MoreInvolved()


if __name__ == "__main__":
    import os
    import sys
    print(f"curdir: {os.curdir}")
    print(f"cwd: {os.getcwd()}" )
    print(f"exec: {sys.executable}")
    main()
