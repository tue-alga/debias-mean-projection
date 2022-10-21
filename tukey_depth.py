from msilib.schema import ProgId
import numpy as np
import pandas as pd


def make_vectors_from_p_to_all(p, S):
    vecs = []
    for s in S:
        v = s - p
        vecs.append(v)
    
    return vecs


def make_vectors_in_S(S, m):
    vecs = []
    n = S.shape[0]
    while len(vecs) < m:
        i, j = np.random.randint(0,n, 2)
        if i==j: continue   
        v = S[i] - S[j]
        vecs.append(v)
    return vecs


def calculate_tukey_depth(median, points):
    # vecs = make_vectors_in_S(points, 2000)
    vecs = make_vectors_from_p_to_all(median, points)
    
    p_index = 0
    all_points = np.concatenate((median, points), axis=0)
    depths = []
    n_points = all_points.shape[0]
    indecies = [i for i in range(n_points)]

    for v in vecs:
        v = v.reshape(-1, 1)
        d = all_points.dot(v).reshape(n_points,)
        df = pd.DataFrame({"1D": d, "indices": indecies})
        df = df.sort_values("1D")
        depth = df.loc[:p_index].shape[0]
        depths.append(np.min([depth, df.shape[0] - depth]))
    
    return depths



if __name__ == "__main__":
    median = np.load("median_fem.npy")
    # fem_vecs = np.load("masc_vecs.npy")
    points = np.load("fem_X_train.npy")

    fem = np.concatenate((median, points))

    depths = calculate_tukey_depth(median, points)
    print(np.min(depths))