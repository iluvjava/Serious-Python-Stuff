import core.core
AbstractConjugateGradient = core.core.AbstractConjugateGradient
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
    Kernel = np.ones((boxsize, boxsize))/boxsize**2
    return signal.convolve2d(Image, Kernel, boundary='fill', mode='same')


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

    # Resize the matrix


    def SimpleTrial():
        Blurred = BoxBlur(Matrix)
        plt.matshow(Blurred); plt.title("Boxblurred")
        plt.show()
        Deblurred = AbstractConjugateGradient(BoxBlur,
                                              Blurred,
                                              np.ones(Matrix.shape),
                                              tol=1e-4,
                                              verbose=True)
        plt.matshow(Deblurred.X[-1]); plt.title("Deblurred by GC")
        plt.show()
        SharpepenedDeblurred = Sharpen(Blurred)
        plt.matshow(SharpepenedDeblurred); plt.title("Deblurred by Sharpen")
        plt.show()

    def MoreInvolved():
        N = 3
        BoxSize= 11
        Blurred = SuccessiveBlurr(Matrix, times=N, boxsize=BoxSize)
        plt.matshow(Blurred)
        plt.show()
        Sol = AbstractConjugateGradient(lambda x: SuccessiveBlurr(x, times=N, boxsize=BoxSize),
                                        Blurred,
                                        np.ones(Matrix.shape),
                                        tol=1e-4,
                                        maxitr=50*50,
                                        verbose=True)
        Deblurred = Sol.X[-1]
        plt.matshow(Deblurred)
        plt.show()
        plt.matshow(SuccessiveBlurr(Deblurred, times=N))
        plt.show()
    SimpleTrial()
    MoreInvolved()
    # EvenMoreInvolved()


if __name__ == "__main__":
    import os
    import sys
    print(f"curdir: {os.curdir}")
    print(f"cwd: {os.getcwd()}" )
    print(f"exec: {sys.executable}")
    main()
