import matplotlib.pyplot as plt
import numpy as np
import math
import matlab.engine

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
        q = q / np.linalg.norm(q)
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

def compare_qr(eng, samples, m, n, ill=False):

    diff_map = {
        "gs": [],
        "np": [],
        "m": []
    }

    cond_arr = []

    for s in range(samples):
        
        # init matrix
        A = np.random.rand(m,n)

        if ill:
            A[:,-1] = A[:,0]

        A_m = matlab.double(A.tolist())
        
        # condition
        cond_arr.append(np.linalg.cond(A))

        # decompose w/ gram-schmidt, numpy, qrfactor.m
        Q_gs, R_gs = qr_decomp(A)
        Q_np, R_np = np.linalg.qr(A)
        Q_m, R_m = eng.qrfactor(A_m, nargout=2)
        Q_m = np.array(Q_m)
        R_m = np.array(R_m)
        
        # evaluate decomposition qualities
        diff_map["gs"].append(np.linalg.norm(A - Q_gs@R_gs))
        diff_map["np"].append(np.linalg.norm(A - Q_np@R_np))
        diff_map["m"].append(np.linalg.norm(A - Q_m@R_m))

    # plot
    err_kw = dict(lw=1, capsize=15, capthick=1)
    fig, ax = plt.subplots(figsize=(8,5))
    ax.axhline(y=0, color="k", linestyle="--", alpha=0.2)
    x, width = 0, 1/len(diff_map.keys())
    labels, ticks = [], []
    for k, v in diff_map.items():
        
        yval = np.mean(v)
        yerr = np.std(v) * 1.98
        labels.append(k)
        ticks.append(x)
        ax.bar(x, yval, width, yerr=yerr, label=k, 
            error_kw=err_kw)
        
        ax.set_xlabel("QR method")
        ax.set_ylabel("norm(A - A_reconstructed)")
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels)
        ax.set_title(f"Mean condition: {np.mean(cond_arr)}")

        x += 0.5

    plt.show()

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