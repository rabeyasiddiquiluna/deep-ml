import numpy as np

def group_normalization(X: np.ndarray, gamma: np.ndarray, beta: np.ndarray, num_groups: int, epsilon: float = 1e-5) -> np.ndarray:
    # Your code here

  B, C, H, W = X.shape
  assert C % num_groups==0 ,"C must be divisible by num_groups"

  # B, C, H, W -> B, G,C//G H, W

  Xg = X.reshape(B,num_groups, C//num_groups,H,W)
  mean = Xg.mean(axis=(2,3,4),keepdims = True)
  var = Xg.var(axis=(2,3,4),keepdims = True)

  Xg_norm = (Xg - mean)/np.sqrt(var + epsilon)

  Xg_norm = Xg_norm.reshape(B,C,H,W)

  return gamma.reshape(1, C, 1, 1) * Xg_norm + beta.reshape(1, C, 1, 1)

