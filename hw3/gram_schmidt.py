import matplotlib.pyplot as plt
import numpy as np
from scipy import linalg
import math

def qr_decomp(A):
    """
    Perform QR decomposition of A using gram-shmidt
    """
    Q = gram_schmidt(A)
    R = build_R(A, Q)

    return Q, R

def gram_schmidt(A):
    """
    Orthonormalize the matrix A via modified gram-schmidt
    """

    Q = np.zeros_like(A)

    for i in range(A.shape[1]):
        
        # select column
        q = np.array(A[:,i])
        
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

def build_R(A, Q):
    """
    Build upper triangular R from A = QR decomp
    """
    m, n = Q.shape
    R = np.zeros((n,n))

    # build
    for i in range(n):
        for j in range(n):
            R[i,j] = np.inner(Q[:,i], A[:,j])

    return R

def main():

    A = np.array([
        [1, 0, -2],
        [0.5, 1, 0],
        [0.3, -1, -6],
        [0, 1.2, 0]
    ])

    Q, R = qr_decomp(A)

    return

if __name__ == "__main__":
    main()