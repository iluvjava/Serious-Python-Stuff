import core.laplacian2d
Laplacian2D = core.laplacian2d.Laplacian2D
import core.cg_core
ConjugateGradient = core.cg_core.ConjugateGradient
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import scipy.sparse as sparse
from tqdm import tqdm


def ReadImageAsNumpy(file:str):
    img = Image.open(file)
    numpydata = np.asarray(img, dtype=np.float32)
    return numpydata


def ApplyLaplacian(matrix):
    m, n = matrix.shape
    L = Laplacian2D(m, n)
    return (L@(matrix.reshape(-1))).reshape(matrix.shape)

def SuccessiveBlurring(matrix, iteration:int=1, deltaT=1):
    # assert deltaT < 0.5, "A value larger than 0.5 makes it Unstable, it gives bad CFL number."
    Blurred = matrix
    for __ in tqdm(range(iteration)):
        Blurred += deltaT*ApplyLaplacian(matrix)
    return Blurred

def SuccessiveDeblurring(matrix, iteration:int=10, deltaT=0.01):
    m, n = matrix.shape
    L = Laplacian2D(m, n)
    Identity = sparse.eye(m * n)
    Deblurred = matrix
    for __ in range(iteration):
        Sol = ConjugateGradient(
            (deltaT*L + Identity),
            Deblurred.reshape(-1)[:, np.newaxis],
            maxitr=200
        )
        Deblurred = Sol.X[-1]
        print(f"CG number of Itr: {Sol.NumberItr}, image max value: {np.max(Deblurred)}")


    return Deblurred.reshape(matrix.shape)



def main():
    TheImage = ReadImageAsNumpy("./data/image2.png")
    TheImage /= 255
    print("Image read. ")
    Blurred = SuccessiveBlurring(TheImage[..., 0])
    plt.imshow(Blurred)
    plt.show()

    Deblurred = SuccessiveDeblurring(Blurred)
    plt.imshow(Deblurred)
    plt.show()

    Deblurred = SuccessiveDeblurring(TheImage[..., 0])
    plt.imshow(Deblurred)
    plt.show()




if __name__ == "__main__":
    import os
    import sys
    print(f"curdir: {os.curdir}")
    print(f"cwd: {os.getcwd()}" )
    print(f"exec: {sys.executable}")
    main()
