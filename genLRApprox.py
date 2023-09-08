import nimfa
import pandas as pd
import numpy as np
from numpy import count_nonzero

def generate_low_rank_approximate(dataset_name, low, high, iter, inc=100):
    f'''
    This function generates low rank approximates of the adjacency matrix of the given dataset.
    - generates low rank approximates inside BMF$dataset_name$ folder
    :param dataset_name: folder name where we get the data from
    :param low: lowest rank
    :param high: highest rank
    :param iter: stop criteria for Bmf
    :param inc: default to 100
    '''

    MAT = pd.read_csv(f'{dataset_name}/A.txt', header=None, sep='\t').to_numpy()
    if high > MAT.shape[0]:
        high = MAT.shape[0]
    for rankval in range(low, high, inc):
        bmf = nimfa.Bmf(MAT, seed="nndsvd", rank=rankval, max_iter=iter, lambda_w=1.1, lambda_h=1.1)
        bmf_fit = bmf()
        W = bmf_fit.basis()
        H = bmf_fit.coef()
        T = np.dot(W, H)
        T = T.tolist()
        for i, x in enumerate(T):
            for j, y in enumerate(x):
                if T[i][j] > .5:
                    T[i][j] = 1
                else:
                    T[i][j] = 0
        df = pd.DataFrame(T)
        df.to_csv(f"{dataset_name}/BMF/{rankval}.txt", header=False, index=False)
        R = abs(T-MAT)
        print(str(rankval)+" "+str(count_nonzero(R)))