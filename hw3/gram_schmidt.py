import matplotlib.pyplot as plt
import numpy as np
from scipy import linalg
import math

def gram_schmidt(A):
    """
    Orthonormalize the matrix A via modified gram-schmidt
    """

    Q = np.zeros_like(A)

    for i in range(A.shape[1]):
        
        # select column
        q = A[:,i]
        
        # make orthogonal to all previous cols
        proj = np.zeros_like(q)
        for j in np.arange(i):
            u = Q[:,j]
            proj += (np.inner(u, q) / np.inner(u, u)) * u
        
        q -= proj
        
        # normalize
        q = q / linalg.norm(q)
        Q[:,i] = q
    
    return Q

def main():

    return

if __name__ == "__main__":
    main()