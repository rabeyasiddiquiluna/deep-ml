import numpy as np

def avg_pool_2d(input_matrix, pool_size):
    x = np.asarray(input_matrix, dtype=float)
    H, W = x.shape
    return x.reshape(H // pool_size, pool_size,
                     W // pool_size, pool_size).mean(axis=(1, 3))