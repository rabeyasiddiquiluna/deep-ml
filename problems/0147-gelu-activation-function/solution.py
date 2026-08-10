import numpy as np

def GeLU(x: np.ndarray) -> np.ndarray:
	# Your code here
	cdf = 0.5 * ( 1.0 + np.tanh(np.sqrt(2.0/np.pi )*(x +.044715* np.power(x,3))))
	return np.round(x *cdf, 4)